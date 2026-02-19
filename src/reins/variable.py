"""
Variable type management and unified variable creation for REINS.
"""

from enum import Enum

import torch
from neuromancer.constraint import (
    Constraint as _NMConstraint,
    Variable as _NMVariable,
    variable as _nm_variable,
    Eq, LT, GT
)


class Constraint(_NMConstraint):
    """Constraint that recognises Variable subclasses (isinstance check)."""

    def __init__(self, left, right, comparator, weight=1.0, name=None):
        # Skip type checks in neuromancer Constraint
        super(_NMConstraint, self).__init__()
        
        # Convert left to _NMVariable
        if not isinstance(left, _NMVariable):
            if isinstance(left, (int, float, complex, bool)):
                display_name = str(left)
            else:
                display_name = str(id(left))
            if not isinstance(left, torch.Tensor):
                left = torch.tensor(left)
            left = _nm_variable(left, display_name=display_name)

        # Convert right to _NMVariable
        if not isinstance(right, _NMVariable):
            if isinstance(right, (int, float, complex, bool)):
                display_name = str(right)
            else:
                display_name = str(id(right))
            if not isinstance(right, torch.Tensor):
                right = torch.tensor(right)
            right = _nm_variable(right, display_name=display_name)

        # Set name and keys
        if name is None:
            name = f'{left.display_name} {comparator} {right.display_name}'
        self.key = f'{left.key}_{comparator}_{right.key}'
        input_keys = left.keys + right.keys
        output_keys = [self.key, f'{self.key}_value', f'{self.key}_violation']
        self.input_keys, self.output_keys, self.name = input_keys, output_keys, name
        self.left = left
        self.right = right
        self.comparator = comparator
        self.weight = weight
        
    def __xor__(self, norm):
        comparator = type(self.comparator)(norm=norm)
        return Constraint(self.left, self.right, comparator, weight=self.weight, name=self.name)

    def __mul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=self.weight*weight, name=self.name)

    def __rmul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=self.weight*weight, name=self.name)


class Variable(_NMVariable):
    """
    Variable for REINS computation graphs.

    Wraps neuromancer Variable to ensure ``key`` is set correctly
    (neuromancer's positional arg is ``input_variables``, not ``key``).

    Args:
        key: String key for data dict lookup.
    """

    def __init__(self, key):
        super().__init__(key=key)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return Constraint(self, other, Eq())

    def __lt__(self, other):
        return Constraint(self, other, LT())

    def __le__(self, other):
        return Constraint(self, other, LT())

    def __gt__(self, other):
        return Constraint(self, other, GT())

    def __ge__(self, other):
        return Constraint(self, other, GT())


class VarType(Enum):
    """
    Variable type enumeration.
    """
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"

    def __repr__(self):
        return f"VarType.{self.name}"


class TypeVariable(Variable):
    """
    Typed decision variable for REINS mixed-integer optimization.

    Behaves exactly like a neuromancer Variable (can be used directly
    in computation graph expressions such as ``x @ Q``, ``x <= b``),
    with additional type metadata and a ``.relaxed`` accessor.

    Args:
        key: Variable name (must not end with '_rel').
        num_vars: Total number of variables.
        integer_indices: Indices of integer variables.
        binary_indices: Indices of binary variables.
        var_types: Single VarType (broadcast to all vars) or list of VarType.
            Mutually exclusive with indices-based parameters.
    """

    def __init__(self, key, num_vars=None,
                 integer_indices=None,
                 binary_indices=None,
                 var_types=None):
        # Validate key
        if key.endswith("_rel"):
            raise ValueError(
                f"Variable key '{key}' cannot end with '_rel' "
                f"(reserved for relaxed variables)."
            )

        # Initialize neuromancer Variable
        super().__init__(key=key)

        # Resolve type metadata
        types = _resolve_var_types(num_vars, integer_indices,
                                   binary_indices, var_types)

        # Store metadata
        self._var_types = types
        self._num_vars = len(types)
        self._integer_indices = [i for i, vt in enumerate(types)
                                 if vt == VarType.INTEGER]
        self._binary_indices = [i for i, vt in enumerate(types)
                                if vt == VarType.BINARY]
        self._continuous_indices = [i for i, vt in enumerate(types)
                                    if vt == VarType.CONTINUOUS]

        # Relaxed variable (key + '_rel') for referencing relaxed solutions
        self._relaxed = Variable(key + "_rel")

    @property
    def var_types(self):
        """List of VarType for each variable dimension."""
        return self._var_types

    @property
    def num_vars(self):
        """Total number of variable dimensions."""
        return self._num_vars

    @property
    def integer_indices(self):
        """Indices of integer-typed dimensions."""
        return self._integer_indices

    @property
    def binary_indices(self):
        """Indices of binary-typed dimensions."""
        return self._binary_indices

    @property
    def continuous_indices(self):
        """Indices of continuous-typed dimensions."""
        return self._continuous_indices

    @property
    def relaxed(self):
        """Relaxed Variable (key + '_rel')."""
        return self._relaxed

    def __repr__(self):
        return (f"TypeVariable(key='{self.key}', num_vars={self._num_vars}, "
                f"var_types={self._var_types})")


def _build_var_types(num_vars, integer_indices=None, binary_indices=None):
    """Build a VarType list from num_vars and index lists."""
    # Check for overlap between integer and binary indices
    overlap = set(integer_indices or []) & set(binary_indices or [])
    if overlap:
        raise ValueError(
            f"Indices {overlap} appear in both integer_indices and "
            f"binary_indices. Each index must have exactly one type."
        )

    # Initialize all as continuous as default
    types = [VarType.CONTINUOUS] * num_vars

    # Set integer types based on indices
    for i in (integer_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Integer index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.INTEGER

    # Set binary types based on indices
    for i in (binary_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Binary index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.BINARY

    return types


def _resolve_var_types(num_vars, integer_indices, binary_indices, var_types):
    """Dispatch to explicit list or index-based construction."""
    if var_types is not None:
        if integer_indices is not None or binary_indices is not None:
            raise ValueError(
                "Cannot specify both var_types and indices-based parameters. "
                "Choose one approach: either pass var_types OR use indices."
            )
        # Single VarType -> broadcast to num_vars
        if isinstance(var_types, VarType):
            if num_vars is None:
                num_vars = 1
            return [var_types] * num_vars
        # List of VarTypes -> use directly
        return list(var_types)

    # Build from indices
    if num_vars is None:
        raise ValueError(
            "num_vars is required when using integer_indices or binary_indices."
        )
    return _build_var_types(num_vars, integer_indices, binary_indices)