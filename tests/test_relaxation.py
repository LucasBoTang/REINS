"""
Unit tests for RelaxationNode.
"""

import pytest
import torch
import torch.nn as nn

from neuromancer.system import Node
from reins.variable import Variable
from reins.node.relaxation import RelaxationNode


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


def _var(key):
    """Create a reins Variable with the given key."""
    return Variable(key)


# ── TestRelaxationNodeInit ──────────────────────────────────────────────────

class TestRelaxationNodeInit:
    """Test RelaxationNode initialization and key handling."""

    def test_single_output_key(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")])
        assert rel.output_keys == ["x_rel"]
        assert rel.input_keys == ["b"]

    def test_single_var_normalized(self):
        """Single Variable (not in list) should be normalized to list."""
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, _var("b"), _var("x"))
        assert rel.output_keys == ["x_rel"]

    def test_multi_output_keys(self):
        net = nn.Linear(4, 6)
        rel = RelaxationNode(net, [_var("b")], [_var("x"), _var("y")], sizes=[3, 3])
        assert rel.output_keys == ["x_rel", "y_rel"]

    def test_is_node(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")])
        assert isinstance(rel, Node)
        assert isinstance(rel, nn.Module)

    def test_name(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")], name="my_relaxation")
        assert rel.name == "my_relaxation"

    def test_sizes_stored(self):
        net = nn.Linear(4, 6)
        rel = RelaxationNode(net, [_var("b")], [_var("x"), _var("y")], sizes=[3, 3])
        assert rel.sizes == [3, 3]

    def test_multi_input_keys(self):
        net = nn.Linear(8, 3)
        rel = RelaxationNode(net, [_var("b"), _var("d")], [_var("x")])
        assert rel.input_keys == ["b", "d"]

    def test_multi_output_without_sizes_raises(self):
        net = nn.Linear(4, 6)
        with pytest.raises(ValueError, match="sizes is required"):
            RelaxationNode(net, [_var("b")], [_var("x"), _var("y")])


# ── TestRelaxationNodeForward ───────────────────────────────────────────────

class TestRelaxationNodeForward:
    """Test RelaxationNode forward pass."""

    def test_single_var_output(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")])
        data = {"b": torch.randn(8, 4)}
        result = rel(data)
        assert "x_rel" in result
        assert result["x_rel"].shape == (8, 3)

    def test_multi_var_split(self):
        net = nn.Linear(4, 6)
        rel = RelaxationNode(net, [_var("b")], [_var("x"), _var("y")], sizes=[3, 3])
        data = {"b": torch.randn(8, 4)}
        result = rel(data)
        assert "x_rel" in result
        assert "y_rel" in result
        assert result["x_rel"].shape == (8, 3)
        assert result["y_rel"].shape == (8, 3)

    def test_uneven_split(self):
        net = nn.Linear(4, 7)
        rel = RelaxationNode(net, [_var("b")], [_var("x"), _var("y")], sizes=[5, 2])
        data = {"b": torch.randn(8, 4)}
        result = rel(data)
        assert result["x_rel"].shape == (8, 5)
        assert result["y_rel"].shape == (8, 2)

    def test_split_correctness(self):
        """Split outputs should match full output slices."""
        net = nn.Linear(4, 6)
        rel = RelaxationNode(net, [_var("b")], [_var("x"), _var("y")], sizes=[4, 2])
        b = torch.randn(8, 4)
        full_output = net(b)
        data = {"b": b}
        result = rel(data)
        assert torch.allclose(result["x_rel"], full_output[:, :4])
        assert torch.allclose(result["y_rel"], full_output[:, 4:])

    def test_single_batch(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")])
        data = {"b": torch.randn(1, 4)}
        result = rel(data)
        assert result["x_rel"].shape == (1, 3)

    def test_does_not_mutate_data(self):
        net = nn.Linear(4, 3)
        rel = RelaxationNode(net, [_var("b")], [_var("x")])
        data = {"b": torch.randn(8, 4)}
        original_keys = set(data.keys())
        rel(data)
        assert set(data.keys()) == original_keys

    def test_multi_input(self):
        """Network receiving concatenated inputs."""
        net = nn.Linear(6, 3)
        rel = RelaxationNode(net, [_var("b"), _var("d")], [_var("x")])
        data = {"b": torch.randn(8, 4), "d": torch.randn(8, 2)}
        result = rel(data)
        assert result["x_rel"].shape == (8, 3)


# ── TestRelaxationNodeExport ────────────────────────────────────────────────

class TestRelaxationNodeExport:
    """Test that RelaxationNode is exported correctly."""

    def test_import_from_node(self):
        from reins.node import RelaxationNode as R
        assert R is RelaxationNode
