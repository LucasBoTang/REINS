"""
Base classes for rounding nodes.
"""

from abc import ABC, abstractmethod

import torch
from neuromancer.system import Node

from reins.node.rounding.functions import DiffFloor
from reins.variable import TypeVariable


class RoundingNode(Node, ABC):
    """
    Base class for differentiable rounding nodes.

    Args:
        vars: TypeVariable or list of TypeVariables.
        name: Module name.
    """

    def __init__(self, vars, name="rounding"):
        # Normalize single variable to list
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        # Validate variable types
        for v in vars:
            if not isinstance(v, TypeVariable):
                raise TypeError(
                    f"Expected TypeVariable, got {type(v).__name__} for key '{getattr(v, 'key', '?')}'."
                )

        # Initialize Node with input/output keys
        input_keys = [v.relaxed.key for v in vars]
        output_keys = [v.key for v in vars]
        super().__init__(
            callable=None,
            input_keys=input_keys,
            output_keys=output_keys,
            name=name,
        )

        # Variable metadata for rounding logic
        self.vars = vars
        # Total variable count
        self.num_vars = sum(v.num_vars for v in vars)

    @abstractmethod
    def forward(self, data):
        """
        Round continuous variables in data dict.

        Args:
            data: Dictionary containing variable tensors.

        Returns:
            Dictionary with rounded output variables.
        """
        pass


class LearnableRoundingLayer(RoundingNode, ABC):
    """
    Base class for network-based rounding layers.

    Args:
        callable: Network mapping [params, vars] to per-variable outputs.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables.
        continuous_update: Whether to update continuous variables (default: False).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, name="learnable_rounding"):
        super().__init__(vars, name)

        # Normalize params to list
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.param_keys = [p.key for p in params]
        self.continuous_update = continuous_update

        # Extend input keys to include parameter keys
        self.input_keys = list(self.param_keys) + self.input_keys

        # Network and differentiable floor
        self.net = callable
        self.floor = DiffFloor()

    def forward(self, data):
        # Network input: [params, relaxed vars]
        features = torch.cat(
            [data[k] for k in self.param_keys]
            + [data[v.relaxed.key] for v in self.vars],
            dim=-1,
        )
        hidden = self.net(features)

        # Round per variable using offset tracking
        output = {}
        offset = 0
        for var in self.vars:
            n = var.num_vars
            x = data[var.relaxed.key].clone()
            h_var = hidden[:, offset:offset + n]

            # Optionally update continuous variables via network adjustment
            if self.continuous_update and var.continuous_indices:
                x[:, var.continuous_indices] += h_var[:, var.continuous_indices]

            # Round integer variables: floor(x) + binary
            if var.integer_indices:
                x_int = x[:, var.integer_indices]
                x_floor = self.floor(x_int)
                binary = self._round_integer(x_int, x_floor, h_var[:, var.integer_indices])
                x[:, var.integer_indices] = x_floor + binary
            
            # Round binary variables
            if var.binary_indices:
                binary = self._round_binary(x[:, var.binary_indices], h_var[:, var.binary_indices])
                x[:, var.binary_indices] = binary

            output[var.key] = x
            offset += n
        return output

    @abstractmethod
    def _round_integer(self, x_int, x_floor, h_int):
        """Return binary rounding decision (0 or 1) for integer variables."""
        pass

    @abstractmethod
    def _round_binary(self, x_bin, h_bin):
        """Return rounded binary values (0 or 1)."""
        pass
