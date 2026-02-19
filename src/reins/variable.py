"""
Variable type management and unified variable creation for NeuroPMINLP.
"""

from enum import Enum

import neuromancer as nm


class VarType(Enum):
    """
    Variable type enumeration.
    """
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"

    def __repr__(self):
        return f"VarType.{self.name}"


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
        # Single VarType → broadcast to num_vars
        if isinstance(var_types, VarType):
            if num_vars is None:
                num_vars = 1
            return [var_types] * num_vars
        # List of VarTypes → use directly
        return list(var_types)

    # Build from indices
    if num_vars is None:
        raise ValueError(
            "num_vars is required when using integer_indices or binary_indices."
        )
    return _build_var_types(num_vars, integer_indices, binary_indices)


def _attach_metadata(var, types):
    """Attach type metadata (var_types, indices) to var."""
    # Attach variable types
    var.var_types = types
    # Attach number of variables
    var.num_vars = len(types)
    # Attach indices for each type
    var.integer_indices = [i for i, vt in enumerate(types)
                           if vt == VarType.INTEGER]
    var.binary_indices = [i for i, vt in enumerate(types)
                          if vt == VarType.BINARY]
    var.continuous_indices = [i for i, vt in enumerate(types)
                              if vt == VarType.CONTINUOUS]


def _attach_relaxed(var):
    """Attach relaxed variable (key + "_rel") for all typed variables."""
    var.relaxed = nm.variable(var.key + "_rel")


def variable(key, num_vars=None,
             integer_indices=None,
             binary_indices=None,
             var_types=None):
    """
    Create a neuromancer variable, optionally with type metadata.

    Without type params: equivalent to ``nm.variable(key)``.
    With type params: attaches ``var_types``, ``num_vars``, indices,
    and ``relaxed`` (continuous relaxation, key = key + "_rel").

    Args:
        key: Variable name.
        num_vars: Total number of variables.
        integer_indices: Indices of integer variables.
        binary_indices: Indices of binary variables.
        var_types: Single VarType (broadcast to all vars) or list of VarType.
            Mutually exclusive with indices-based parameters.

    Returns:
        Neuromancer variable, with type attrs attached if type params given.
    """
    # Validate key
    if key.endswith("_rel"):
        raise ValueError(
            f"Variable key '{key}' cannot end with '_rel' "
            f"(reserved for relaxed variables)."
        )

    # Create base variable
    var = nm.variable(key)

    # No type information -> plain neuromancer variable
    if all(v is None for v in (num_vars, integer_indices,
                                binary_indices, var_types)):
        return var

    # Resolve and attach type metadata
    types = _resolve_var_types(num_vars, integer_indices,
                               binary_indices, var_types)
    _attach_metadata(var, types)

    # Attach relaxed variable
    _attach_relaxed(var)

    return var
