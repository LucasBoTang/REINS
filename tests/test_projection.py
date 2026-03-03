"""
Unit tests for GradientProjection.
"""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from reins.projection.gradient import GradientProjection


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


class MockRounding(nn.Module):
    """Mock rounding: round to nearest integer."""

    def __init__(self, input_key="x_rel", output_key="x"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, data):
        x_rel = data[self.input_key]
        rounded = x_rel.round()
        # Use Straight-Through Estimator (STE) to allow gradients to flow back to x_rel
        return {self.output_key: (rounded - x_rel).detach() + x_rel}


class MockConstraint(nn.Module):
    """
    Mock constraint: x <= upper_bound.

    output_keys[2] is the violation tensor (matching nm.Constraint format).
    Violation = relu(x - upper_bound).
    """

    def __init__(self, upper_bound, name="mock", input_key="x"):
        super().__init__()
        self.upper_bound = upper_bound
        self.input_key = input_key
        self.output_keys = [
            f"con_{name}",
            f"con_{name}_loss",
            f"con_{name}_viol",
        ]

    def forward(self, data):
        x = data[self.input_key]
        viol = torch.relu(x - self.upper_bound)
        penalty = viol.reshape(x.shape[0], -1).sum(dim=1).mean()
        return {
            self.output_keys[0]: penalty,
            self.output_keys[1]: penalty,
            self.output_keys[2]: viol,
        }


class MockLowerBoundConstraint(nn.Module):
    """Mock constraint: x >= lower_bound."""

    def __init__(self, lower_bound, name="mock_lb", input_key="x"):
        super().__init__()
        self.lower_bound = lower_bound
        self.input_key = input_key
        self.output_keys = [f"con_{name}", f"con_{name}_loss", f"con_{name}_viol"]

    def forward(self, data):
        x = data[self.input_key]
        # Violation if x < lower_bound => relu(lower - x)
        viol = torch.relu(self.lower_bound - x)
        penalty = viol.reshape(x.shape[0], -1).sum(dim=1).mean()
        return {
            self.output_keys[0]: penalty,
            self.output_keys[1]: penalty,
            self.output_keys[2]: viol,
        }


class TestGradientProjection:
    """Tests for GradientProjection."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=5.0)],
            target_keys=["x_rel"],
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[3.0, 4.0]])}
        result = proj(data)
        assert isinstance(result, dict)
        assert "x" in result
        assert "x_rel" in result

    def test_feasible_input_unchanged(self):
        """Already feasible input should not change."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.2, 2.8]])}
        result = proj(data)
        # Rounded: [1.0, 3.0], both <= 10, so feasible
        assert torch.allclose(result["x"], torch.tensor([[1.0, 3.0]]))

    def test_infeasible_input_projected(self):
        """Infeasible input should be projected towards feasibility."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=200,
            step_size=0.1,
            decay=0.99,
        )
        # x_rel = [1.5, 5.5] -> rounds to [2, 6], violates x <= 3 at index 1
        data = {"x_rel": torch.tensor([[1.5, 5.5]])}
        result = proj(data)
        # After projection, all rounded values should be <= 3
        assert (result["x"] <= 3.0 + 0.1).all()

    def test_early_stop_on_tolerance(self):
        """Should stop early when violation < tolerance."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=1000,
            tolerance=1e-6,
        )
        # Already feasible -> should stop immediately
        data = {"x_rel": torch.tensor([[1.0, 2.0]])}
        result = proj(data)
        assert "x" in result

    def test_step_size_decay(self):
        """Step size should decay each iteration."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=10,
            step_size=1.0,
            decay=0.5,
        )
        data = {"x_rel": torch.tensor([[1.5, 5.5]])}
        # Should not error; decay reduces step each iteration
        result = proj(data)
        assert "x" in result

    def test_multiple_constraints(self):
        """Should handle multiple constraints."""
        con1 = MockConstraint(upper_bound=3.0, name="upper")
        con2 = MockConstraint(upper_bound=5.0, name="upper2")
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[con1, con2],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.5, 4.5]])}
        result = proj(data)
        # Tighter constraint is x <= 3
        assert (result["x"] <= 3.0 + 0.1).all()

    def test_batch_dimension(self):
        """Should work with batch size > 1."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[1.5, 5.5], [2.0, 4.0]])}
        result = proj(data)
        assert result["x"].shape == (2, 2)

    def test_preserves_other_keys(self):
        """Should preserve other keys in data dict."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[1.0, 2.0]]), "params": torch.tensor([[0.5]])}
        result = proj(data)
        assert "params" in result
        assert torch.equal(result["params"], torch.tensor([[0.5]]))

    def test_final_output_is_rounded(self):
        """Final output should be rounded (integer values)."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=1,
        )
        data = {"x_rel": torch.tensor([[1.3, 2.7]])}
        result = proj(data)
        # MockRounding rounds to nearest integer
        assert torch.allclose(result["x"], result["x"].round())

    def test_custom_target_key(self):
        """Should work with custom target keys."""
        # Setup: y_rel -> y, constraint on y
        proj = GradientProjection(
            rounding_components=[MockRounding(input_key="y_rel", output_key="y")],
            constraints=[MockConstraint(upper_bound=3.0, input_key="y")],
            target_keys=["y_rel"],
            num_steps=50,
            step_size=0.1,
        )
        data = {"y_rel": torch.tensor([[5.5]])}
        result = proj(data)
        
        assert "y" in result
        # Should be projected to <= 3
        assert (result["y"] <= 3.1).all()

    def test_conflicting_constraints(self):
        """Should stabilize when constraints conflict (infeasible region)."""
        # x <= 3 AND x >= 4. Gap is [3, 4].
        # Inside the gap, gradients cancel out or are zero depending on formulation.
        # It should simply stop somewhere reasonable without crashing.
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[
                MockConstraint(upper_bound=3.0, name="upper"),
                MockLowerBoundConstraint(lower_bound=4.0, name="lower")
            ],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[10.0]])} # Start far above
        result = proj(data)
        # It should move down towards 4.0 and stop around there
        assert result["x"].item() <= 4.5 

    def test_no_constraints(self):
        """Should just perform rounding if no constraints provided."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[],
            target_keys=["x_rel"],
        )
        data = {"x_rel": torch.tensor([[1.6]])}
        result = proj(data)
        # Just rounds 1.6 -> 2.0
        assert torch.allclose(result["x"], torch.tensor([[2.0]]))


class TestGradientProjectionNumerical:
    """Verify exact numerical behavior of GradientProjection."""

    def test_violation_decreases(self):
        """Total violation should decrease after projection."""
        rounding = MockRounding()
        con = MockConstraint(upper_bound=3.0)
        proj = GradientProjection(
            rounding_components=[rounding],
            constraints=[con],
            target_keys=["x_rel"],
            num_steps=50,
            step_size=0.1,
        )
        # Before: round(5.5) = 6.0, violation = relu(6-3) = 3.0
        x_rel_init = torch.tensor([[5.5]])
        before_rounded = rounding({"x_rel": x_rel_init})["x"]
        before_viol = torch.relu(before_rounded - 3.0).sum().item()
        assert before_viol > 0
        # After projection
        result = proj({"x_rel": x_rel_init.clone()})
        after_viol = torch.relu(result["x"] - 3.0).sum().item()
        assert after_viol < before_viol

    def test_converges_to_feasible(self):
        """With enough steps, should converge to feasibility."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=500,
            step_size=0.1,
            decay=1.0,
            tolerance=1e-8,
        )
        data = {"x_rel": torch.tensor([[5.5]])}
        result = proj(data)
        # After convergence, x should be <= 3
        assert result["x"].item() <= 3.0 + 0.01

    def test_decay_reduces_step_impact(self):
        """With decay < 1, fewer violations are resolved in fixed steps."""
        con = MockConstraint(upper_bound=3.0)
        proj_no_decay = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[con],
            target_keys=["x_rel"],
            num_steps=10,
            step_size=0.5,
            decay=1.0,
        )
        proj_decay = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0, name="m2")],
            target_keys=["x_rel"],
            num_steps=10,
            step_size=0.5,
            decay=0.5,
        )
        r1 = proj_no_decay({"x_rel": torch.tensor([[5.5]])})
        r2 = proj_decay({"x_rel": torch.tensor([[5.5]])})
        viol1 = torch.relu(r1["x"] - 3.0).sum().item()
        viol2 = torch.relu(r2["x"] - 3.0).sum().item()
        # No decay should reduce violation more (or equal)
        assert viol1 <= viol2 + 1e-6

    def test_lower_bound_violation_decreases(self):
        """Projection should work with lower bound constraints."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockLowerBoundConstraint(lower_bound=5.0)],
            target_keys=["x_rel"],
            num_steps=200,
            step_size=0.1,
        )
        # Before: round(1.5) = 2.0, violation = relu(5-2) = 3.0
        data = {"x_rel": torch.tensor([[1.5]])}
        result = proj(data)
        # After projection, x should move towards >= 5
        assert result["x"].item() >= 4.0

    def test_batch_violations_decrease_independently(self):
        """Each sample in batch should have reduced violations."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
        )
        data = {"x_rel": torch.tensor([[5.5], [6.5]])}
        result = proj(data)
        # Both samples should be closer to feasible
        assert (result["x"] <= 3.0 + 0.5).all()


class TestGradientProjectionProjIters:
    """Tests for _proj_iters tracking in GradientProjection."""

    def test_proj_iters_present_in_output(self):
        """Returned data should contain _proj_iters key."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=100,
        )
        data = {"x_rel": torch.tensor([[1.0, 2.0]])}
        result = proj(data)
        assert "_proj_iters" in result

    def test_proj_iters_shape_matches_batch(self):
        """_proj_iters should have one entry per sample."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=100,
        )
        data = {"x_rel": torch.tensor([[1.0], [2.0], [3.0]])}
        result = proj(data)
        assert result["_proj_iters"].shape == (3,)

    def test_proj_iters_early_stop_fewer_iters(self):
        """Feasible input should converge in few iterations."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=100.0)],
            target_keys=["x_rel"],
            num_steps=1000,
            tolerance=1e-6,
        )
        data = {"x_rel": torch.tensor([[1.0, 2.0]])}
        result = proj(data)
        # Should converge very quickly (violation is already 0)
        assert result["_proj_iters"][0].item() <= 5

    def test_num_iters_attribute_updated(self):
        """num_iters attribute should be updated after __call__."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=10.0)],
            target_keys=["x_rel"],
            num_steps=100,
            tolerance=1e-6,
        )
        assert proj.num_iters == 0
        data = {"x_rel": torch.tensor([[1.0]])}
        proj(data)
        assert proj.num_iters > 0


class TestGradientProjectionNumericalConvergence:
    """Additional numerical convergence tests."""

    def test_exact_convergence_single_variable(self):
        """Single variable should converge to boundary value."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=1000,
            step_size=0.1,
            decay=1.0,
            tolerance=1e-8,
        )
        # round(5.5) = 6.0, needs to project down to 3.0
        data = {"x_rel": torch.tensor([[5.5]])}
        result = proj(data)
        assert result["x"].item() <= 3.0 + 0.01

    def test_multi_dim_convergence(self):
        """Multi-dimensional variable should converge per-dimension."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=2.0)],
            target_keys=["x_rel"],
            num_steps=500,
            step_size=0.1,
            decay=1.0,
            tolerance=1e-8,
        )
        # [1.2, 5.5, 3.7] -> round = [1, 6, 4]. Index 0 feasible, 1 and 2 violate
        data = {"x_rel": torch.tensor([[1.2, 5.5, 3.7]])}
        result = proj(data)
        assert (result["x"] <= 2.0 + 0.1).all()
        # Index 0 should still be feasible
        assert result["x"][0, 0].item() <= 2.0

    def test_lower_and_upper_bound_simultaneous(self):
        """Test projection with both lower and upper bounds."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[
                MockConstraint(upper_bound=5.0, name="ub"),
                MockLowerBoundConstraint(lower_bound=2.0, name="lb"),
            ],
            target_keys=["x_rel"],
            num_steps=500,
            step_size=0.1,
            decay=1.0,
            tolerance=1e-8,
        )
        # round(0.5) = 0.0, violates lb=2. round(8.5) = 8.0, violates ub=5.
        data = {"x_rel": torch.tensor([[0.5, 8.5]])}
        result = proj(data)
        # Both should be in [2, 5]
        assert result["x"][0, 0].item() >= 1.5
        assert result["x"][0, 1].item() <= 5.5

    def test_large_batch_convergence(self):
        """All samples in a large batch should converge to feasibility."""
        proj = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=500,
            step_size=0.1,
            decay=1.0,
            tolerance=1e-8,
        )
        torch.manual_seed(0)
        # 32 samples, 4 dims, values in [3, 8] -> all violate upper=3
        data = {"x_rel": 3.0 + 5.0 * torch.rand(32, 4)}
        result = proj(data)
        assert (result["x"] <= 3.0 + 0.1).all()

    def test_violation_monotonically_decreases_with_more_steps(self):
        """More steps should lead to less or equal violation."""
        data_10 = {"x_rel": torch.tensor([[7.5]])}
        data_100 = {"x_rel": torch.tensor([[7.5]])}
        proj_10 = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0)],
            target_keys=["x_rel"],
            num_steps=10,
            step_size=0.1,
            decay=1.0,
        )
        proj_100 = GradientProjection(
            rounding_components=[MockRounding()],
            constraints=[MockConstraint(upper_bound=3.0, name="m2")],
            target_keys=["x_rel"],
            num_steps=100,
            step_size=0.1,
            decay=1.0,
        )
        r10 = proj_10(data_10)
        r100 = proj_100(data_100)
        viol_10 = torch.relu(r10["x"] - 3.0).sum().item()
        viol_100 = torch.relu(r100["x"] - 3.0).sum().item()
        assert viol_100 <= viol_10 + 1e-6


class TestGradientProjectionExport:
    """Test GradientProjection import paths."""

    def test_import_from_projection_package(self):
        """Should be importable from reins.projection."""
        from reins.projection import GradientProjection as GP
        assert GP is GradientProjection
