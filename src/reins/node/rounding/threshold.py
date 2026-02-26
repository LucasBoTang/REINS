"""
Dynamic threshold rounding layer.
"""

import torch

from reins.node.rounding.base import LearnableRoundingLayer
from reins.node.rounding.functions import ThresholdBinarize, GumbelThresholdBinarize


class DynamicThresholdRounding(LearnableRoundingLayer):
    """
    Dynamic threshold rounding with learned thresholds.

    Args:
        callable: Network mapping [params, vars] to per-variable outputs.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables.
        continuous_update: Whether to update continuous variables (default: False).
        stochastic: Use Gumbel-Softmax noise during training (default: False).
        slope: Slope for sigmoid-smoothed binarization when deterministic (default: 10).
        temperature: Gumbel-Softmax temperature when stochastic (default: 1.0).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, stochastic=False,
                 slope=10, temperature=1.0, name=None):
        if name is None:
            name = ("stochastic_dynamic_threshold_rounding"
                    if stochastic else "dynamic_threshold_rounding")
        super().__init__(callable, params, vars, continuous_update, name)
        # Gumbel-Softmax threshold binarization for stochastic,
        # sigmoid-smoothed threshold binarization otherwise
        if stochastic:
            self.threshold_binarize = GumbelThresholdBinarize(
                temperature=temperature
            )
        else:
            self.threshold_binarize = ThresholdBinarize(slope=slope)

    def _round_integer(self, x_int, x_floor, h_int):
        thresh = torch.sigmoid(h_int)
        x_frac = (x_int - x_floor).detach()
        return self.threshold_binarize(x_frac, thresh)

    def _round_binary(self, x_bin, h_bin):
        thresh = torch.sigmoid(h_bin)
        return self.threshold_binarize(x_bin, thresh)
