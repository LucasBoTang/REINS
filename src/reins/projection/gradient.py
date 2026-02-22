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
        # Save originals for NaN fallback
        xs_orig = {k: data[k].clone() for k in self.target_keys}
        batch_size = next(iter(xs.values())).shape[0]
        d = 1.0

        # Per-sample iteration tracking
        sample_iters = torch.zeros(batch_size, dtype=torch.long)

        # Build temp data once, update in-place each iteration
        temp_data = {**data}
        num_iters = 0
        for _ in range(self.num_steps):
            num_iters += 1
            temp_data.update(xs)

            # Round through components
            for comp in self.rounding_components:
                temp_data.update(comp(temp_data))

            # Compute total violation from all constraints at once
            viols = []
            for con in self.constraints:
                out = con(temp_data)
                viol_key = con.output_keys[2]
                viols.append(out[viol_key].reshape(batch_size, -1).sum(dim=1))
            if not viols:
                break
            total_viol = torch.stack(viols).sum(dim=0) if len(viols) > 1 else viols[0]

            # Track per-sample convergence (feasible or NaN)
            finite_mask = torch.isfinite(total_viol)
            unsettled = sample_iters == 0
            newly_settled = unsettled & (~finite_mask | (total_viol < self.tolerance))
            sample_iters[newly_settled] = num_iters

            # Check convergence (ignore NaN samples)
            if not finite_mask.any():
                break
            if total_viol[finite_mask].max().item() < self.tolerance:
                break

            # Backprop only through finite samples to avoid NaN contamination
            grads = torch.autograd.grad(
                total_viol[finite_mask].sum(), list(xs.values()),
                allow_unused=True,
            )
            xs = {
                k: (xs[k] - d * self.step_size * g).detach().requires_grad_(True)
                if g is not None else xs[k]
                for k, g in zip(self.target_keys, grads)
            }
            d = self.decay * d

        # Samples that never settled get the total iteration count
        sample_iters[sample_iters == 0] = num_iters
        self.num_iters = num_iters
        data["_proj_iters"] = sample_iters

        # Final update — revert NaN samples to pre-projection originals
        for k in self.target_keys:
            projected = xs[k].detach()
            nan_mask = torch.isnan(projected).any(dim=-1)
            if nan_mask.any():
                projected[nan_mask] = xs_orig[k][nan_mask]
            data[k] = projected

        # Final round
        for comp in self.rounding_components:
            data.update(comp(data))

        return data
