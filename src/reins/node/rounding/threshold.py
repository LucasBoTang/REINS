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
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable outputs.
        continuous_update: Whether to update continuous variables (default: False).
        slope: Slope for sigmoid-smoothed binarization (default: 10).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, slope=10,
                 name="dynamic_threshold_rounding"):
        super().__init__(vars, param_keys, net, continuous_update, name)
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
        vars: Variable or list of variables with type metadata.
        param_keys: List of parameter keys to read from data dict.
        net: Network mapping [params, vars] to per-variable outputs.
        continuous_update: Whether to update continuous variables (default: False).
        temperature: Gumbel-Softmax temperature (default: 1.0).
        name: Module name.
    """

    def __init__(self, vars, param_keys, net,
                 continuous_update=False, temperature=1.0,
                 name="stochastic_dynamic_threshold_rounding"):
        super().__init__(vars, param_keys, net,
                         continuous_update=continuous_update,
                         name=name)
        # Replace sigmoid-smoothed binarization with Gumbel-Softmax version
        self.threshold_binarize = GumbelThresholdBinarize(
            temperature=temperature
        )
