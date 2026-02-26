"""
Adaptive selection rounding layer.
"""

import torch

from reins.node.rounding.base import LearnableRoundingLayer
from reins.node.rounding.functions import DiffBinarize, DiffGumbelBinarize


class AdaptiveSelectionRounding(LearnableRoundingLayer):
    """
    Adaptive selection rounding with learned rounding directions.

    Args:
        callable: Network mapping [params, vars] to per-variable selection.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables.
        continuous_update: Whether to update continuous variables (default: False).
        stochastic: Use Gumbel-Softmax noise during training (default: False).
        tolerance: Tolerance for near-integer masking (default: 1e-3).
        temperature: Gumbel-Softmax temperature when stochastic (default: 1.0).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, stochastic=False,
                 tolerance=1e-3, temperature=1.0, name=None):
        if name is None:
            name = ("stochastic_adaptive_selection_rounding"
                    if stochastic else "adaptive_selection_rounding")
        super().__init__(callable, params, vars, continuous_update, name)
        # Gumbel-Softmax binarization for stochastic, deterministic STE otherwise
        if stochastic:
            self.binarize = DiffGumbelBinarize(temperature=temperature)
        else:
            self.binarize = DiffBinarize()
        self.tolerance = tolerance

    def _round_integer(self, x_int, x_floor, h_int):
        binary = self.binarize(h_int)
        return self._int_mask(binary, x_int, x_floor)

    def _round_binary(self, x_bin, h_bin):
        return self.binarize(h_bin)

    def _int_mask(self, binary, x, x_floor):
        """Mask rounding for values already close to an integer."""
        frac = (x - x_floor).detach()
        binary = torch.where(frac < self.tolerance,
                             torch.zeros_like(binary), binary)
        binary = torch.where(frac > 1.0 - self.tolerance,
                             torch.ones_like(binary), binary)
        return binary
