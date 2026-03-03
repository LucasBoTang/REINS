"""
Unit tests for LearnableSolver.
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import neuromancer as nm
from neuromancer.system import Node
from neuromancer.dataset import DictDataset
from reins.loss import PenaltyLoss

from reins.solver import LearnableSolver
from reins.variable import TypeVariable
from reins.projection.gradient import GradientProjection
from reins.node.rounding.ste import STERounding


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


# ---- Helpers ----

def _make_var(key, num_vars, integer_indices=None, binary_indices=None):
    """Create a TypeVariable for testing."""
    return TypeVariable(
        key, num_vars=num_vars,
        integer_indices=integer_indices,
        binary_indices=binary_indices,
    )


def _make_relaxation(input_key, output_key, insize, outsize):
    """Create a minimal relaxation Node."""
    net = nn.Linear(insize, outsize)
    return Node(net, [input_key], [output_key], name="relaxation")


def _make_loss(var_key, constraints=None):
    """Create a minimal PenaltyLoss: minimize x^2."""
    x = nm.variable(var_key)
    f = torch.sum(x ** 2, dim=1)
    obj = f.minimize(weight=1.0, name="obj")
    return PenaltyLoss(objectives=[obj], constraints=constraints or [])


def _make_constraint(var_key, upper_bound=5.0):
    """Create an nm.Constraint: x <= upper_bound."""
    x = nm.variable(var_key)
    return 100.0 * (x <= upper_bound)


# ---- Fixtures ----

@pytest.fixture
def int_var():
    return _make_var("x", 3, integer_indices=[0, 1, 2])


@pytest.fixture
def rel(int_var):
    return _make_relaxation("b", "x_rel", insize=4, outsize=3)


@pytest.fixture
def rounding(int_var):
    return STERounding(int_var)


@pytest.fixture
def constraint():
    return _make_constraint("x", upper_bound=5.0)


@pytest.fixture
def loss(constraint):
    """Loss with constraint included (for default projection)."""
    return _make_loss("x", constraints=[constraint])


@pytest.fixture
def loss_no_constraints():
    """Loss without constraints."""
    return _make_loss("x")


# ---- TestLearnableSolverConstruction ----

class TestLearnableSolverConstruction:
    """Tests for LearnableSolver construction and validation."""

    def test_construction_basic(self, rel,rounding, loss):
        """Basic construction should succeed (constraints from loss)."""
        solver = LearnableSolver(rel, rounding, loss)
        assert solver.relaxation_node is rel
        assert solver.rounding_node is rounding

    def test_construction_no_constraints(self, rel,rounding, loss_no_constraints):
        """Construction without constraints should succeed (no projection)."""
        solver = LearnableSolver(rel, rounding, loss_no_constraints)
        assert solver.projection is None

    def test_key_mismatch_raises(self, rounding, loss):
        """Should raise ValueError when relaxation output doesn't match rounding input."""
        bad_rel = _make_relaxation("b", "y_rel", insize=4, outsize=3)
        with pytest.raises(ValueError, match="Key mismatch"):
            LearnableSolver(bad_rel, rounding, loss)

    def test_dimension_mismatch_raises(self, loss):
        """Should raise ValueError when relaxation output dim != total rounding variable dim."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        rel = _make_relaxation("b", "x_rel", insize=4, outsize=2)
        rnd = STERounding(var)
        with pytest.raises(ValueError, match="Relaxation output dim"):
            LearnableSolver(rel, rnd, loss)

    def test_dimension_check_skipped_without_attribute(self, loss):
        """Should skip dimension check if callable has no out_features."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        # Use a lambda wrapper that has no out_features attribute
        net = lambda x: torch.zeros(x.shape[0], 3)  # noqa: E731
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        # Should not raise
        solver = LearnableSolver(rel, rnd, loss)
        assert solver is not None

    def test_dimension_check_with_outsize_attribute(self, loss):
        """Should use outsize attribute as fallback when out_features is absent."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])

        class NetWithOutsize(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)
                self.outsize = 3  # fallback attribute
            def forward(self, x):
                return self.linear(x)

        net = NetWithOutsize()
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        # Should succeed (outsize=3 == num_vars=3)
        solver = LearnableSolver(rel, rnd, loss)
        assert solver is not None

    def test_dimension_check_outsize_mismatch_raises(self, loss):
        """Should raise when outsize doesn't match total rounding dim."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])

        class NetWithBadOutsize(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 5)
                self.outsize = 5
            def forward(self, x):
                return self.linear(x)

        net = NetWithBadOutsize()
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        with pytest.raises(ValueError, match="Relaxation output dim"):
            LearnableSolver(rel, rnd, loss)

    def test_projection_default_enabled(self, rel,rounding, loss):
        """Projection should be enabled by default (constraints from loss)."""
        solver = LearnableSolver(rel, rounding, loss)
        assert solver.projection_steps == 10000
        assert isinstance(solver.projection, GradientProjection)

    def test_projection_disabled_explicit(self, rel,rounding, loss):
        """Projection can be disabled with projection_steps=0."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        assert solver.projection is None

    def test_projection_skipped_without_constraints(self, rel,rounding, loss_no_constraints):
        """Projection is skipped when loss has no constraints."""
        solver = LearnableSolver(rel, rounding, loss_no_constraints)
        assert solver.projection_steps == 10000
        assert solver.projection is None

    def test_projection_custom_steps(self, rel,rounding, loss):
        """Should create GradientProjection with custom steps."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=10,
        )
        assert solver.projection_steps == 10
        assert isinstance(solver.projection, GradientProjection)

    def test_problem_contains_nodes(self, rel,rounding, loss):
        """Problem should contain relaxation and rounding nodes."""
        solver = LearnableSolver(rel, rounding, loss)
        assert rel in solver.problem.nodes
        assert rounding in solver.problem.nodes

    def test_stores_references(self, rel,rounding, loss):
        """Should store references to relaxation and rounding."""
        solver = LearnableSolver(rel, rounding, loss)
        assert solver.relaxation_node is rel
        assert solver.rounding_node is rounding


# ---- TestLearnableSolverPredict ----

class TestLearnableSolverPredict:
    """Tests for LearnableSolver.predict() without projection."""

    def test_predict_returns_dict(self, rel,rounding, loss):
        """predict() should return a dict with the output key."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert isinstance(result, dict)
        assert "x" in result

    def test_predict_output_shape(self, rel,rounding, loss):
        """Output should have correct shape (batch, num_vars)."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert result["x"].shape == (2, 3)

    def test_predict_preserves_input_keys(self, rel,rounding, loss):
        """Input key 'b' should still be in the result."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert "b" in result

    def test_predict_adds_relaxed_key(self, rel,rounding, loss):
        """Relaxed key 'x_rel' should be added by relaxation."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert "x_rel" in result

    def test_predict_eval_mode(self, rel,rounding, loss):
        """predict() should set problem to eval mode."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        solver.problem.train()
        data = {"b": torch.randn(2, 4)}
        solver.predict(data)
        assert not solver.problem.training

    def test_predict_integer_output(self, rel,rounding, loss):
        """Rounded integer variables should be integral."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(5, 4)}
        result = solver.predict(data)
        x = result["x"]
        assert torch.equal(x, x.round()), "Integer variables should be integral after rounding"

    def test_predict_proj_iters_without_projection(self, rel,rounding, loss):
        """predict() without projection should set _proj_iters to zeros."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        data = {"b": torch.randn(3, 4)}
        result = solver.predict(data)
        assert "_proj_iters" in result
        assert result["_proj_iters"].shape == (3,)
        assert (result["_proj_iters"] == 0).all()


# ---- TestLearnableSolverPredictWithProjection ----

class TestLearnableSolverPredictWithProjection:
    """Tests for LearnableSolver.predict() with projection."""

    def test_predict_with_projection(self, rel,rounding, loss):
        """Prediction with projection should return dict with output key."""
        solver = LearnableSolver(
            rel, rounding, loss,
            projection_steps=10,
            projection_step_size=0.1,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert isinstance(result, dict)
        assert "x" in result
        assert result["x"].shape == (2, 3)

    def test_predict_with_projection_feasibility(self):
        """After projection, constraints should be approximately satisfied."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        # Use large weights to produce large x_rel values
        net = nn.Linear(4, 3)
        nn.init.constant_(net.weight, 5.0)
        nn.init.constant_(net.bias, 10.0)
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        con = _make_constraint("x", upper_bound=3.0)
        loss = _make_loss("x", constraints=[con])
        solver = LearnableSolver(
            rel, rnd, loss,
            projection_steps=300,
            projection_step_size=0.1,
        )
        data = {"b": torch.ones(1, 4)}
        result = solver.predict(data)
        # After projection, values should be close to feasible
        assert (result["x"] <= 3.0 + 1.0).all()

    def test_predict_with_projection_integer_output(self):
        """After projection, integer variables should still be integral."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        net = nn.Linear(4, 3)
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        con = _make_constraint("x", upper_bound=5.0)
        loss = _make_loss("x", constraints=[con])
        solver = LearnableSolver(
            rel, rnd, loss,
            projection_steps=50,
            projection_step_size=0.1,
        )
        data = {"b": torch.randn(5, 4)}
        result = solver.predict(data)
        x = result["x"]
        assert torch.equal(x, x.round()), "Integer variables should be integral after projection"

    def test_predict_with_projection_has_proj_iters(self):
        """predict() with projection should set _proj_iters > 0."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        net = nn.Linear(4, 3)
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        con = _make_constraint("x", upper_bound=5.0)
        loss = _make_loss("x", constraints=[con])
        solver = LearnableSolver(
            rel, rnd, loss,
            projection_steps=50,
            projection_step_size=0.1,
        )
        data = {"b": torch.randn(2, 4)}
        result = solver.predict(data)
        assert "_proj_iters" in result
        assert result["_proj_iters"].shape == (2,)
        assert (result["_proj_iters"] > 0).all()


# ---- TestLearnableSolverVariableTypes ----

class TestLearnableSolverVariableTypes:
    """Tests for mixed variable types and multi-variable scenarios."""

    def test_mixed_continuous_integer_binary(self):
        """Continuous cols stay continuous, integer cols integral, binary cols {0,1}."""
        # dim 0: integer, dim 1: binary, dim 2-3: continuous
        var = _make_var("x", 4, integer_indices=[0], binary_indices=[1])
        rel = _make_relaxation("b", "x_rel", insize=4, outsize=4)
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        data = {"b": torch.randn(10, 4)}
        result = solver.predict(data)
        x = result["x"]

        # integer column should be integral
        assert torch.equal(x[:, 0], x[:, 0].round()), "Integer column should be integral"
        # binary column should be 0 or 1
        assert ((x[:, 1] == 0) | (x[:, 1] == 1)).all(), "Binary column should be 0 or 1"
        # continuous columns should generally NOT be integral (not rounded)
        # (with random weights, it's extremely unlikely all values are exact integers)
        cont = x[:, 2:]
        assert not torch.equal(cont, cont.round()), "Continuous columns should not be rounded"

    def test_binary_only_output(self):
        """All-binary variable should produce only {0, 1} values."""
        var = _make_var("x", 3, binary_indices=[0, 1, 2])
        rel = _make_relaxation("b", "x_rel", insize=4, outsize=3)
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        data = {"b": torch.randn(10, 4)}
        result = solver.predict(data)
        x = result["x"]
        assert ((x == 0) | (x == 1)).all(), "Binary variables should be 0 or 1"

    def test_multi_variable(self):
        """Solver with two variables (x: integer, y: binary) via a shared relaxation node."""
        var_x = _make_var("x", 2, integer_indices=[0, 1])
        var_y = _make_var("y", 2, binary_indices=[0, 1])
        rnd = STERounding([var_x, var_y])

        # Custom relaxation node that outputs both x_rel and y_rel
        class DualHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(4, 4)
                self.out_features = 4
            def forward(self, b):
                return self.net(b)[:, :2], self.net(b)[:, 2:]

        dual = DualHead()
        rel = Node(dual, ["b"], ["x_rel", "y_rel"], name="relaxation")

        loss_x = nm.variable("x")
        loss_y = nm.variable("y")
        f = torch.sum(loss_x ** 2 + loss_y ** 2, dim=1)
        obj = f.minimize(weight=1.0, name="obj")
        loss = PenaltyLoss(objectives=[obj], constraints=[])

        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        data = {"b": torch.randn(5, 4)}
        result = solver.predict(data)

        # Both keys present
        assert "x" in result and "y" in result
        # x should be integral
        assert torch.equal(result["x"], result["x"].round()), "x should be integral"
        # y should be binary
        assert ((result["y"] == 0) | (result["y"] == 1)).all(), "y should be 0 or 1"


# ---- TestLearnableSolverTrain ----

class TestLearnableSolverTrain:
    """Integration tests for LearnableSolver.train()."""

    def test_train_runs_without_error(self, rel,rounding, loss):
        """Training should complete without errors."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=1e-3)

        b_train = torch.randn(100, 4)
        b_val = torch.randn(20, 4)
        ds_train = DictDataset({"b": b_train}, name="train")
        ds_val = DictDataset({"b": b_val}, name="dev")
        loader_train = DataLoader(
            ds_train, batch_size=32, collate_fn=ds_train.collate_fn
        )
        loader_val = DataLoader(
            ds_val, batch_size=32, collate_fn=ds_val.collate_fn
        )

        solver.train(
            loader_train, loader_val, optimizer,
            epochs=2, patience=5, warmup=0,
            device="cpu",
        )

    def test_train_loss_decreases(self):
        """Training loss should decrease over epochs."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        net = nn.Linear(4, 3)
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        b_data = torch.randn(128, 4)
        ds = DictDataset({"b": b_data}, name="train")
        loader = DataLoader(ds, batch_size=128, collate_fn=ds.collate_fn)

        # Compute loss before training
        solver.problem.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            output_before = solver.problem(batch)
            loss_before = output_before["train_loss"].item()

        # Train
        optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=1e-2)
        ds_val = DictDataset({"b": torch.randn(16, 4)}, name="dev")
        loader_val = DataLoader(ds_val, batch_size=16, collate_fn=ds_val.collate_fn)
        solver.train(
            loader, loader_val, optimizer,
            epochs=50, patience=50, warmup=0,
            device="cpu",
        )

        # Compute loss after training
        solver.problem.eval()
        with torch.no_grad():
            output_after = solver.problem(batch)
            loss_after = output_after["train_loss"].item()

        assert loss_after < loss_before, (
            f"Loss should decrease: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_predict_after_train(self, rel,rounding, loss):
        """Prediction should work after training."""
        solver = LearnableSolver(
            rel, rounding, loss, projection_steps=0,
        )
        optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=1e-3)

        b_train = torch.randn(64, 4)
        b_val = torch.randn(16, 4)
        ds_train = DictDataset({"b": b_train}, name="train")
        ds_val = DictDataset({"b": b_val}, name="dev")
        loader_train = DataLoader(
            ds_train, batch_size=32, collate_fn=ds_train.collate_fn
        )
        loader_val = DataLoader(
            ds_val, batch_size=32, collate_fn=ds_val.collate_fn
        )

        solver.train(
            loader_train, loader_val, optimizer,
            epochs=2, patience=5, warmup=0,
            device="cpu",
        )

        data = {"b": torch.randn(1, 4)}
        result = solver.predict(data)
        assert "x" in result
        assert result["x"].shape == (1, 3)


# ---- TestLearnableSolverExport ----

class TestLearnableSolverExport:
    """Tests for LearnableSolver exports."""

    def test_import_from_package(self):
        """Should be importable from reins."""
        from reins import LearnableSolver as LS
        assert LS is LearnableSolver

    def test_in_all(self):
        """Should be listed in __all__."""
        import reins
        assert "LearnableSolver" in reins.__all__


# ---- TestLearnableSolverNumerical ----

class TestLearnableSolverNumerical:
    """Numerical tests for LearnableSolver predict behavior."""

    def test_predict_deterministic_in_eval(self):
        """Two identical predict calls should return identical results."""
        var = _make_var("x", 3, integer_indices=[0, 1, 2])
        net = nn.Linear(4, 3)
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        b = torch.randn(5, 4)
        r1 = solver.predict({"b": b.clone()})
        r2 = solver.predict({"b": b.clone()})
        assert torch.equal(r1["x"], r2["x"])

    def test_predict_known_weights(self):
        """With known weights, predict should return expected integer values."""
        var = _make_var("x", 2, integer_indices=[0, 1])
        net = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            net.weight.copy_(torch.eye(2))
            net.bias.zero_()
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        # b = [1.3, 2.7] -> x_rel = [1.3, 2.7]
        # STE: floor + binarize(frac - 0.5) => floor(1.3)+0=1, floor(2.7)+1=3
        data = {"b": torch.tensor([[1.3, 2.7]])}
        result = solver.predict(data)
        assert torch.equal(result["x"], torch.tensor([[1.0, 3.0]]))

    def test_predict_negative_values(self):
        """STE rounding should handle negative values correctly."""
        var = _make_var("x", 2, integer_indices=[0, 1])
        net = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            net.weight.copy_(torch.eye(2))
            net.bias.zero_()
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        # b = [-1.7, -0.3] -> x_rel = [-1.7, -0.3]
        # STE: floor(-1.7)=-2, frac=0.3, binarize(-0.2)=0 => -2
        #      floor(-0.3)=-1, frac=0.7, binarize(0.2)=1  => 0
        data = {"b": torch.tensor([[-1.7, -0.3]])}
        result = solver.predict(data)
        assert torch.equal(result["x"], torch.tensor([[-2.0, 0.0]]))

    def test_predict_batch_independence(self):
        """Each sample in batch should be independent."""
        var = _make_var("x", 2, integer_indices=[0, 1])
        net = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            net.weight.copy_(torch.eye(2))
            net.bias.zero_()
        rel = Node(net, ["b"], ["x_rel"], name="relaxation")
        rnd = STERounding(var)
        loss = _make_loss("x")
        solver = LearnableSolver(rel, rnd, loss, projection_steps=0)

        b1 = torch.tensor([[1.3, 2.7]])
        b2 = torch.tensor([[3.1, 4.9]])
        # Single predict
        r1 = solver.predict({"b": b1.clone()})
        r2 = solver.predict({"b": b2.clone()})
        # Batch predict
        r_batch = solver.predict({"b": torch.cat([b1, b2], dim=0)})
        assert torch.equal(r_batch["x"][0], r1["x"][0])
        assert torch.equal(r_batch["x"][1], r2["x"][0])
