"""
Adaptive selection rounding layers (deterministic and stochastic).
"""

import torch

from reins.node.rounding.base import LearnableRoundingLayer
from reins.node.rounding.functions import DiffBinarize, DiffGumbelBinarize


class AdaptiveSelectionRounding(LearnableRoundingLayer):
    """
    Adaptive selection rounding with network-adjusted variables.

    Network selects rounding direction for integer/binary variables.
    Uses deterministic STE binarization.

    Args:
        callable: Network mapping [params, vars] to per-variable selection.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables with type metadata.
        continuous_update: Whether to update continuous variables (default: False).
        tolerance: Tolerance for near-integer masking (default: 1e-3).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, tolerance=1e-3,
                 name="adaptive_selection_rounding"):
        super().__init__(callable, params, vars, continuous_update, name)
        # Deterministic STE binarization
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


class StochasticAdaptiveSelectionRounding(AdaptiveSelectionRounding):
    """
    Stochastic adaptive selection rounding with Gumbel-Softmax noise.

    Same as AdaptiveSelectionRounding but uses DiffGumbelBinarize
    for stochastic training and deterministic evaluation.

    Args:
        callable: Network mapping [params, vars] to per-variable selection.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables with type metadata.
        continuous_update: Whether to update continuous variables (default: False).
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, temperature=1.0,
                 name="stochastic_adaptive_selection_rounding"):
        super().__init__(callable, params, vars,
                         continuous_update=continuous_update,
                         name=name)
        # Replace deterministic STE binarization with Gumbel-Softmax version
        self.binarize = DiffGumbelBinarize(temperature=temperature)
