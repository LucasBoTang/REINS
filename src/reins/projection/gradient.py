"""
Gradient-based feasibility projection.
"""

import torch


class GradientProjection:
    """
    Gradient-based feasibility projection.

    Args:
        rounding_components: List of rounding modules.
        constraints: List of Constraint objects.
        target_keys: Keys for relaxed variables to project.
        num_steps: Maximum projection iterations.
        step_size: Initial step size for gradient descent.
        decay: Step size decay factor per iteration.
        tolerance: Stop if max violation < tolerance.
    """

    def __init__(self, rounding_components, constraints, target_keys,
                 num_steps=10000, step_size=0.01, decay=0.999,
                 tolerance=1e-6):
        self.rounding_components = rounding_components
        self.constraints = constraints
        self.target_keys = target_keys
        self.num_steps = num_steps
        self.step_size = step_size
        self.decay = decay
        self.tolerance = tolerance
        self.num_iters = 0

    def __call__(self, data):
        """
        Project relaxed variables towards feasibility.

        Args:
            data: Dictionary containing variable tensors.

        Returns:
            Updated dictionary with projected and rounded solution.
        """
        # Clone and enable grad for all target variables
        xs = {k: data[k].clone().requires_grad_(True) for k in self.target_keys}
        # Save originals as ultimate fallback
        xs_orig = {k: data[k].clone() for k in self.target_keys}
        batch_size = next(iter(xs.values())).shape[0]
        device = next(iter(xs.values())).device
        d = 1.0

        # Per-sample step multiplier for backtracking
        sample_d = torch.ones(batch_size, device=device)
        # Per-sample iteration tracking
        sample_iters = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Last known-good state per sample (for backtracking)
        xs_prev = {k: xs[k].detach().clone() for k in self.target_keys}

        # Cache constraint violation keys
        viol_keys = [con.output_keys[2] for con in self.constraints]

        # Build temp data once, update in-place each iteration
        temp_data = {**data}
        num_iters = 0
        for _ in range(self.num_steps):
            num_iters += 1
            temp_data.update(xs)

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Accumulate total violation directly (avoid list + stack)
            total_viol = torch.zeros(batch_size, device=device)
            for con, vk in zip(self.constraints, viol_keys):
                out = con(temp_data)
                total_viol = total_viol + out[vk].reshape(batch_size, -1).sum(dim=1)

            finite_mask = torch.isfinite(total_viol)
            unsettled = sample_iters == 0

            # Backtracking: revert diverged samples to previous state, halve step
            diverged = unsettled & ~finite_mask
            if diverged.any():
                if num_iters == 1:
                    # First evaluation: no gradient step taken yet, nothing to backtrack to
                    sample_iters[diverged] = num_iters
                else:
                    xs = {
                        k: torch.where(diverged.unsqueeze(-1), xs_prev[k], xs[k])
                            .detach().requires_grad_(True)
                        for k in self.target_keys
                    }
                    sample_d[diverged] *= 0.5
                    # Settle samples whose step multiplier is too small
                    sample_iters[diverged & (sample_d < 1e-8)] = num_iters

            # Mark converged (feasible) samples as settled
            converged = unsettled & finite_mask & (total_viol < self.tolerance)
            sample_iters[converged] = num_iters

            # Active: unsettled, finite, still infeasible, not just-diverged
            active_mask = (sample_iters == 0) & finite_mask & (total_viol >= self.tolerance)

            # Stop if all samples are settled
            if not (sample_iters == 0).any():
                break
            # Skip gradient step if only diverged-and-reverted samples remain
            if not active_mask.any():
                continue

            # Save current good state before gradient step
            for k in self.target_keys:
                xs_prev[k][active_mask] = xs[k][active_mask].detach()

            # Backprop only through active (infeasible) samples
            grads = torch.autograd.grad(
                total_viol[active_mask].sum(), list(xs.values()),
                allow_unused=True,
            )
            xs = {
                k: (xs[k] - (sample_d * d * self.step_size).unsqueeze(-1) * g
                     ).detach().requires_grad_(True)
                if g is not None else xs[k]
                for k, g in zip(self.target_keys, grads)
            }
            d = self.decay * d

        # Samples that never settled get the total iteration count
        sample_iters[sample_iters == 0] = num_iters
        self.num_iters = num_iters
        data["_proj_iters"] = sample_iters

        # Revert non-finite (NaN/inf) samples to originals
        for k in self.target_keys:
            projected = xs[k].detach()
            bad_mask = ~torch.isfinite(projected).all(dim=-1)
            if bad_mask.any():
                projected[bad_mask] = xs_orig[k][bad_mask]
            data[k] = projected

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
