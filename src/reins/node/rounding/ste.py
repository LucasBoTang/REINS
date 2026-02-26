"""
Straight-Through Estimator based rounding layer.
"""

from reins.node.rounding.base import RoundingNode
from reins.node.rounding.functions import DiffFloor, DiffBinarize, DiffGumbelBinarize


class STERounding(RoundingNode):
    """
    STE-based rounding without learnable parameters.

    Args:
        vars: TypeVariable or list of TypeVariables.
        stochastic: Use Gumbel-Softmax noise during training (default: False).
        temperature: Gumbel-Softmax temperature when stochastic (default: 1.0).
        name: Module name.
    """

    def __init__(self, vars, stochastic=False, temperature=1.0, name=None):
        if name is None:
            name = "stochastic_ste_rounding" if stochastic else "ste_rounding"
        super().__init__(vars, name)
        # Differentiable floor via STE
        self.floor = DiffFloor()
        # Gumbel-Softmax binarization for stochastic, deterministic STE otherwise
        if stochastic:
            self.binarize = DiffGumbelBinarize(temperature=temperature)
        else:
            self.binarize = DiffBinarize()

    def forward(self, data):
        output = {}
        # Round each variable independently (no cross-variable dependency)
        for var in self.vars:
            x = data[var.relaxed.key].clone()

            # Round integer variables: floor(x) + binarize(frac - 0.5)
            if var.integer_indices:
                x_int = x[:, var.integer_indices]
                # Differentiable floor
                x_floor = self.floor(x_int)
                # Binarize fractional part (detach floor to avoid double gradient)
                binary = self.binarize(x_int - x_floor.detach() - 0.5)
                x[:, var.integer_indices] = x_floor + binary

            # Round binary variables: binarize(x - 0.5)
            if var.binary_indices:
                x[:, var.binary_indices] = self.binarize(
                    x[:, var.binary_indices] - 0.5
                )

            # Store rounded result
            output[var.key] = x
        return output
