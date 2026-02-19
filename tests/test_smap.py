"""
Unit tests for SmapNode.
"""

import pytest
import torch
import torch.nn as nn

from neuromancer.system import Node
from reins.node.smap import SmapNode


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


# ── TestSmapNodeInit ──────────────────────────────────────────────────

class TestSmapNodeInit:
    """Test SmapNode initialization and key handling."""

    def test_single_output_key(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"], name="smap")
        assert smap.output_keys == ["x_rel"]
        assert smap.input_keys == ["b"]

    def test_string_output_key(self):
        """Single string should be normalized to list."""
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], "x", name="smap")
        assert smap.output_keys == ["x_rel"]

    def test_multi_output_keys(self):
        net = nn.Linear(4, 6)
        smap = SmapNode(net, ["b"], ["x", "y"], sizes=[3, 3], name="smap")
        assert smap.output_keys == ["x_rel", "y_rel"]

    def test_is_node(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"], name="smap")
        assert isinstance(smap, Node)
        assert isinstance(smap, nn.Module)

    def test_name(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"], name="my_smap")
        assert smap.name == "my_smap"

    def test_sizes_stored(self):
        net = nn.Linear(4, 6)
        smap = SmapNode(net, ["b"], ["x", "y"], sizes=[3, 3])
        assert smap.sizes == [3, 3]

    def test_multi_input_keys(self):
        net = nn.Linear(8, 3)
        smap = SmapNode(net, ["b", "d"], ["x"])
        assert smap.input_keys == ["b", "d"]

    def test_multi_output_without_sizes_raises(self):
        net = nn.Linear(4, 6)
        with pytest.raises(ValueError, match="sizes is required"):
            SmapNode(net, ["b"], ["x", "y"])


# ── TestSmapNodeForward ───────────────────────────────────────────────

class TestSmapNodeForward:
    """Test SmapNode forward pass."""

    def test_single_var_output(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"])
        data = {"b": torch.randn(8, 4)}
        result = smap(data)
        assert "x_rel" in result
        assert result["x_rel"].shape == (8, 3)

    def test_multi_var_split(self):
        net = nn.Linear(4, 6)
        smap = SmapNode(net, ["b"], ["x", "y"], sizes=[3, 3])
        data = {"b": torch.randn(8, 4)}
        result = smap(data)
        assert "x_rel" in result
        assert "y_rel" in result
        assert result["x_rel"].shape == (8, 3)
        assert result["y_rel"].shape == (8, 3)

    def test_uneven_split(self):
        net = nn.Linear(4, 7)
        smap = SmapNode(net, ["b"], ["x", "y"], sizes=[5, 2])
        data = {"b": torch.randn(8, 4)}
        result = smap(data)
        assert result["x_rel"].shape == (8, 5)
        assert result["y_rel"].shape == (8, 2)

    def test_split_correctness(self):
        """Split outputs should match full output slices."""
        net = nn.Linear(4, 6)
        smap = SmapNode(net, ["b"], ["x", "y"], sizes=[4, 2])
        b = torch.randn(8, 4)
        full_output = net(b)
        data = {"b": b}
        result = smap(data)
        assert torch.allclose(result["x_rel"], full_output[:, :4])
        assert torch.allclose(result["y_rel"], full_output[:, 4:])

    def test_single_batch(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"])
        data = {"b": torch.randn(1, 4)}
        result = smap(data)
        assert result["x_rel"].shape == (1, 3)

    def test_does_not_mutate_data(self):
        net = nn.Linear(4, 3)
        smap = SmapNode(net, ["b"], ["x"])
        data = {"b": torch.randn(8, 4)}
        original_keys = set(data.keys())
        smap(data)
        assert set(data.keys()) == original_keys

    def test_multi_input(self):
        """Network receiving concatenated inputs."""
        net = nn.Linear(6, 3)
        smap = SmapNode(net, ["b", "d"], ["x"])
        data = {"b": torch.randn(8, 4), "d": torch.randn(8, 2)}
        result = smap(data)
        assert result["x_rel"].shape == (8, 3)


# ── TestSmapNodeExport ────────────────────────────────────────────────

class TestSmapNodeExport:
    """Test that SmapNode is exported correctly."""

    def test_import_from_node(self):
        from reins.node import SmapNode as S
        assert S is SmapNode
