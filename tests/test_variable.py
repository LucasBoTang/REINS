"""
Unit tests for VarType enum and TypeVariable class.
"""

import pytest
import torch
import neuromancer as nm
from neuromancer.constraint import Variable

from reins.variable import VarType, TypeVariable


class TestVarType:
    """Test VarType enumeration."""

    def test_values(self):
        assert VarType.CONTINUOUS.value == "continuous"
        assert VarType.INTEGER.value == "integer"
        assert VarType.BINARY.value == "binary"

    def test_repr(self):
        assert repr(VarType.CONTINUOUS) == "VarType.CONTINUOUS"
        assert repr(VarType.INTEGER) == "VarType.INTEGER"
        assert repr(VarType.BINARY) == "VarType.BINARY"

    def test_all_members(self):
        members = list(VarType)
        assert len(members) == 3


class TestTypeVariableIndexBased:
    """Test TypeVariable with index-based type specification."""

    def test_all_continuous(self):
        x = TypeVariable("x", num_vars=5)
        assert x.num_vars == 5
        assert x.var_types == [VarType.CONTINUOUS] * 5
        assert x.integer_indices == []
        assert x.binary_indices == []
        assert x.continuous_indices == [0, 1, 2, 3, 4]

    def test_integer_indices(self):
        x = TypeVariable("x", num_vars=5, integer_indices=[0, 2, 4])
        assert x.integer_indices == [0, 2, 4]
        assert x.continuous_indices == [1, 3]
        assert x.binary_indices == []
        assert x.var_types[0] == VarType.INTEGER
        assert x.var_types[1] == VarType.CONTINUOUS
        assert x.var_types[2] == VarType.INTEGER

    def test_binary_indices(self):
        x = TypeVariable("x", num_vars=4, binary_indices=[1, 3])
        assert x.binary_indices == [1, 3]
        assert x.continuous_indices == [0, 2]
        assert x.integer_indices == []

    def test_mixed_indices(self):
        x = TypeVariable("x", num_vars=6,
                         integer_indices=[0, 1],
                         binary_indices=[4, 5])
        assert x.integer_indices == [0, 1]
        assert x.binary_indices == [4, 5]
        assert x.continuous_indices == [2, 3]
        assert x.num_vars == 6

    def test_all_integer(self):
        x = TypeVariable("x", num_vars=3, integer_indices=[0, 1, 2])
        assert x.integer_indices == [0, 1, 2]
        assert x.continuous_indices == []
        assert x.binary_indices == []


class TestTypeVariableExplicitTypes:
    """Test TypeVariable with explicit var_types list."""

    def test_explicit_types(self):
        types = [VarType.INTEGER, VarType.CONTINUOUS, VarType.BINARY]
        x = TypeVariable("x", var_types=types)
        assert x.var_types == types
        assert x.num_vars == 3
        assert x.integer_indices == [0]
        assert x.continuous_indices == [1]
        assert x.binary_indices == [2]

    def test_all_continuous_explicit(self):
        types = [VarType.CONTINUOUS] * 4
        x = TypeVariable("x", var_types=types)
        assert x.continuous_indices == [0, 1, 2, 3]
        assert x.integer_indices == []
        assert x.binary_indices == []

    def test_single_vartype_broadcast(self):
        x = TypeVariable("x", num_vars=5, var_types=VarType.INTEGER)
        assert x.num_vars == 5
        assert x.var_types == [VarType.INTEGER] * 5
        assert x.integer_indices == [0, 1, 2, 3, 4]

    def test_single_vartype_default_num_vars(self):
        x = TypeVariable("x", var_types=VarType.BINARY)
        assert x.num_vars == 1
        assert x.var_types == [VarType.BINARY]
        assert x.binary_indices == [0]


class TestTypeVariableProperties:
    """Test TypeVariable property access."""

    def test_inherits_from_variable(self):
        x = TypeVariable("x", num_vars=3)
        assert isinstance(x, Variable)

    def test_key(self):
        x = TypeVariable("x", num_vars=3)
        assert x.key == "x"

    def test_relaxed_is_variable(self):
        x = TypeVariable("x", num_vars=3, integer_indices=[0])
        assert isinstance(x.relaxed, Variable)
        assert x.relaxed.key == "x_rel"

    def test_self_and_relaxed_are_different(self):
        x = TypeVariable("x", num_vars=3)
        assert x is not x.relaxed

    def test_repr(self):
        x = TypeVariable("x", num_vars=2, var_types=VarType.INTEGER)
        r = repr(x)
        assert "TypeVariable" in r
        assert "'x'" in r
        assert "num_vars=2" in r


class TestTypeVariableRelaxed:
    """Test auto-creation of relaxed variable."""

    def test_relaxed_with_discrete(self):
        x = TypeVariable("x", num_vars=3, integer_indices=[0])
        assert x.relaxed.key == "x_rel"

    def test_relaxed_with_binary(self):
        x = TypeVariable("x", num_vars=3, binary_indices=[0])
        assert x.relaxed.key == "x_rel"

    def test_relaxed_pure_continuous(self):
        x = TypeVariable("x", num_vars=3)
        assert x.relaxed.key == "x_rel"

    def test_relaxed_key_preserved(self):
        x = TypeVariable("x", num_vars=5, integer_indices=[0, 1, 2])
        assert x.key == "x"
        assert x.relaxed.key == "x_rel"


class TestTypeVariableErrors:
    """Test error handling in TypeVariable."""

    def test_indices_without_num_vars(self):
        with pytest.raises(ValueError, match="num_vars is required"):
            TypeVariable("x", integer_indices=[0])

    def test_binary_indices_without_num_vars(self):
        with pytest.raises(ValueError, match="num_vars is required"):
            TypeVariable("x", binary_indices=[0])

    def test_var_types_with_indices_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            TypeVariable("x", var_types=[VarType.CONTINUOUS],
                         integer_indices=[0])

    def test_integer_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            TypeVariable("x", num_vars=3, integer_indices=[5])

    def test_binary_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            TypeVariable("x", num_vars=3, binary_indices=[3])

    def test_negative_index(self):
        with pytest.raises(ValueError, match="out of range"):
            TypeVariable("x", num_vars=3, integer_indices=[-1])

    def test_overlap_integer_binary(self):
        with pytest.raises(ValueError, match="appear in both"):
            TypeVariable("x", num_vars=5,
                         integer_indices=[0, 1],
                         binary_indices=[1, 2])

    def test_key_ending_with_rel(self):
        with pytest.raises(ValueError, match="_rel"):
            TypeVariable("x_rel", num_vars=3)


class TestTypeVariableComputationGraph:
    """Test that TypeVariable works directly in computation graphs."""

    def test_arithmetic(self):
        x = TypeVariable("x", num_vars=3, integer_indices=[0, 1, 2])
        y = nm.variable("y")
        z = x + y
        data = {"x": torch.tensor([1.0, 2.0, 3.0]),
                "y": torch.tensor([10.0, 20.0, 30.0])}
        result = z(data)
        assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))

    def test_relaxed_in_graph(self):
        x = TypeVariable("x", num_vars=2, integer_indices=[0, 1])
        y = x.relaxed + 1.0
        data = {"x_rel": torch.tensor([0.5, 0.7])}
        result = y(data)
        assert torch.allclose(result, torch.tensor([1.5, 1.7]))

    def test_matmul(self):
        x = TypeVariable("x", num_vars=3, var_types=VarType.INTEGER)
        W = torch.randn(3, 2)
        z = x @ W
        data = {"x": torch.ones(1, 3)}
        result = z(data)
        assert result.shape == (1, 2)

    def test_constraint_creation(self):
        from neuromancer.constraint import Constraint
        x = TypeVariable("x", num_vars=3, var_types=VarType.INTEGER)
        con = x <= 5.0
        assert isinstance(con, Constraint)

    def test_objective_creation(self):
        from neuromancer.constraint import Objective
        x = TypeVariable("x", num_vars=3, var_types=VarType.INTEGER)
        obj = (x ** 2).minimize()
        assert isinstance(obj, Objective)


class TestTypeVariableNodeIntegration:
    """Test TypeVariable integration with neuromancer Node."""

    def test_node_with_relaxed_key(self):
        from neuromancer.system import Node
        import torch.nn as nn

        net = nn.Linear(3, 2)
        x = TypeVariable("x", num_vars=2, integer_indices=[0, 1])
        node = Node(net, [x.relaxed.key], [x.relaxed.key], name="relaxation")
        data = {"x_rel": torch.randn(4, 3)}
        out = node(data)
        assert x.relaxed.key in out
        assert out["x_rel"].shape == (4, 2)

    def test_node_chain_with_variables(self):
        from neuromancer.system import Node
        import torch.nn as nn

        x = TypeVariable("x", num_vars=3, integer_indices=[0, 1, 2])
        net = nn.Linear(2, 3)
        rel = Node(net, ["b"], [x.relaxed.key], name="relaxation")
        rnd = Node(lambda d: d, [x.relaxed.key], [x.key], name="round")

        data = {"b": torch.randn(4, 2)}
        data = rel(data)
        assert x.relaxed.key in data
        data = rnd(data)
        assert x.key in data


class TestTypeVariableExport:
    """Test that VarType and TypeVariable are exported from reins."""

    def test_vartype_export(self):
        import reins
        assert hasattr(reins, "VarType")
        assert "VarType" in reins.__all__
        assert reins.VarType is VarType

    def test_type_variable_export(self):
        import reins
        assert hasattr(reins, "TypeVariable")
        assert "TypeVariable" in reins.__all__
        assert reins.TypeVariable is TypeVariable
