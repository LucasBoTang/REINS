"""
Dynamic threshold rounding layers (deterministic and stochastic).
"""

import torch

from reins.node.rounding.base import LearnableRoundingLayer
from reins.node.rounding.functions import ThresholdBinarize, GumbelThresholdBinarize


class DynamicThresholdRounding(LearnableRoundingLayer):
    """
    Dynamic threshold rounding with MLP-predicted thresholds.

    MLP predicts per-variable thresholds from concatenated
    problem parameters and relaxed solutions.

    Args:
        callable: Network mapping [params, vars] to per-variable outputs.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables with type metadata.
        continuous_update: Whether to update continuous variables (default: False).
        slope: Slope for sigmoid-smoothed binarization (default: 10).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, slope=10,
                 name="dynamic_threshold_rounding"):
        super().__init__(callable, params, vars, continuous_update, name)
        # Sigmoid-smoothed threshold binarization
        self.threshold_binarize = ThresholdBinarize(slope=slope)

    def _round_integer(self, x_int, x_floor, h_int):
        thresh = torch.sigmoid(h_int)
        x_frac = (x_int - x_floor).detach()
        return self.threshold_binarize(x_frac, thresh)

    def _round_binary(self, x_bin, h_bin):
        thresh = torch.sigmoid(h_bin)
        return self.threshold_binarize(x_bin, thresh)


class StochasticDynamicThresholdRounding(DynamicThresholdRounding):
    """
    Stochastic dynamic threshold rounding with Gumbel-Softmax noise.

    Same as DynamicThresholdRounding but uses GumbelThresholdBinarize
    for stochastic training and deterministic evaluation.

    Args:
        callable: Network mapping [params, vars] to per-variable outputs.
        params: Parameter Variable or list of parameter Variables.
        vars: TypeVariable or list of TypeVariables with type metadata.
        continuous_update: Whether to update continuous variables (default: False).
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, callable, params, vars,
                 continuous_update=False, temperature=1.0,
                 name="stochastic_dynamic_threshold_rounding"):
        super().__init__(callable, params, vars,
                         continuous_update=continuous_update,
                         name=name)
        # Replace sigmoid-smoothed binarization with Gumbel-Softmax version
        self.threshold_binarize = GumbelThresholdBinarize(
            temperature=temperature
        )
