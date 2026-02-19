"""
Base classes for rounding nodes.
"""

from abc import ABC, abstractmethod

import torch
from neuromancer.system import Node

from reins.node.rounding.functions import DiffFloor


class RoundingNode(Node, ABC):
    """
    Base class for differentiable rounding nodes.

    Inherits from neuromancer Node for compatibility with Problem.step().
    forward() takes a data dict and returns only the output dict
    {output_key: value}, which Problem merges via dict unpacking.

    Accepts single variable or list of variables. Type metadata
    is auto-extracted from each variable object.

    Args:
        vars: Variable or list of variables (created by reins.variable).
        name: Module name.
    """

    def __init__(self, vars, name="rounding"):
        # Normalize single variable to list
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

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
        # Total variable count across all variables (for MLP output sizing)
        self.num_vars = sum(v.num_vars for v in vars)

    @abstractmethod
    def forward(self, data):
        """
        Round continuous variables in data dict.

        Args:
            data: Dictionary containing variable tensors.

        Returns:
            Dictionary with only rounded output variables
            (e.g., {"x": rounded_x, "y": rounded_y}).
        """
        pass


class LearnableRoundingLayer(RoundingNode, ABC):
    """
    Base class for network-based rounding layers.

    Handles common logic: param_keys, net, continuous_update,
    feature concatenation, offset-based variable loop.
    Subclasses implement _round_integer() and _round_binary().

    Args:
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable outputs.
        continuous_update: Whether to update continuous variables (default: False).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, name="learnable_rounding"):
        super().__init__(vars, name)
        self.param_keys = param_keys
        self.continuous_update = continuous_update

        # Extend input keys to include parameter keys
        self.input_keys = list(param_keys) + self.input_keys

        # Network and differentiable floor
        self.net = net
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
        """Compute binary rounding decision for integer variables.

        Args:
            x_int: Relaxed integer variable values.
            x_floor: Differentiable floor of x_int.
            h_int: Network output slice for integer indices.

        Returns:
            Binary tensor (0 or 1) to add to x_floor.
        """
        pass

    @abstractmethod
    def _round_binary(self, x_bin, h_bin):
        """Round binary variables.

        Args:
            x_bin: Relaxed binary variable values.
            h_bin: Network output slice for binary indices.

        Returns:
            Rounded binary tensor (0 or 1).
        """
        pass
