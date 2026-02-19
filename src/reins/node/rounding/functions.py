"""
Straight-through estimators for nondifferentiable operators.
"""

import torch
from torch import nn
from torch.autograd import Function


class DiffFloor(nn.Module):
    """
    Differentiable floor using straight-through estimator.

    Forward: floor(x). Backward: identity (gradient passes through).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # STE: Attach gradient from x
        return x + (torch.floor(x) - x).detach()


class DiffBinarize(nn.Module):
    """
    Differentiable binarize using straight-through estimator.

    Forward: 1 if x >= 0 else 0 (clamped to [-1, 1]).
    Backward: identity (gradient passes through).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _binarizeFuncSTE.apply(x)


class _binarizeFuncSTE(Function):
    """Custom autograd binarize function with STE."""

    @staticmethod
    def forward(ctx, input):
        # Clamp to [-1, 1] then binarize at 0
        logit = torch.clamp(input, min=-1, max=1)
        # Binarize to 0 or 1
        return (logit >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Identity STE: pass gradient through unchanged
        return grad_output.clone()


class DiffGumbelBinarize(nn.Module):
    """
    Differentiable binarize using Gumbel-Softmax trick.

    Train: Gumbel noise + sigmoid soft sample with STE hard sample.
    Eval: temperature-scaled sigmoid thresholded at 0.5.

    Args:
        temperature: Gumbel-Softmax temperature (default: 1.0).
        eps: Epsilon for numerical stability (default: 1e-9).
    """
    def __init__(self, temperature=1.0, eps=1e-9):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x):
        if self.training:
            # Sample Logistic(0, 1) noise directly using torch.logit
            # This is equivalent to Gumbel(0,1) - Gumbel(0,1) but faster and more stable
            u = torch.rand_like(x)
            logistic_noise = torch.logit(u, eps=self.eps)
            noisy_diff = x + logistic_noise
            # Soft relaxation via sigmoid
            soft_sample = torch.sigmoid(noisy_diff / self.temperature)
            # Hard binarization
            hard_sample = (soft_sample > 0.5).float()
            # STE: Attach gradient from soft sample to hard output
            return hard_sample + (soft_sample - soft_sample.detach())
        else:
            # Deterministic threshold in eval
            return (torch.sigmoid(x / self.temperature) > 0.5).float()


class GumbelThresholdBinarize(nn.Module):
    """
    Learned threshold binarization with Gumbel-Softmax stochasticity.

    Train: Gumbel noise + sigmoid on (x - threshold), with STE hard output.
    Eval: deterministic x >= threshold.

    Args:
        temperature: Gumbel-Softmax temperature (default: 1.0).
        eps: Epsilon for numerical stability (default: 1e-9).
    """
    def __init__(self, temperature=1.0, eps=1e-9):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x, threshold):
        # Clamp threshold to valid probability range
        threshold = torch.clamp(threshold, 0, 1)
        diff = x - threshold
        if self.training:
            # Sample Logistic(0, 1) noise directly using torch.logit
            u = torch.rand_like(x)
            logistic_noise = torch.logit(u, eps=self.eps)
            noisy_diff = diff + logistic_noise
            # Soft relaxation via sigmoid
            soft_sample = torch.sigmoid(noisy_diff / self.temperature)
            # Hard binarization
            hard_sample = (soft_sample > 0.5).float()
            # STE: Attach gradient from soft sample to hard output
            return hard_sample + (soft_sample - soft_sample.detach())
        else:
            # Deterministic threshold in eval
            return (diff >= 0).float()


class ThresholdBinarize(nn.Module):
    """
    Learned threshold binarization with sigmoid smoothing.

    Forward: 1 if x >= threshold else 0, with sigmoid STE for gradients.

    Args:
        slope: Sigmoid slope for soft approximation (default: 10).
    """
    def __init__(self, slope=10):
        super().__init__()
        self.slope = slope

    def forward(self, x, threshold):
        # Clamp threshold to valid probability range
        threshold = torch.clamp(threshold, 0, 1)
        # Hard binarization
        hard_round = (x >= threshold).float()
        # Soft approximation via scaled sigmoid
        diff = x - threshold
        smoothed_round = torch.sigmoid(self.slope * diff)
        # STE: Attach gradient from sigmoid to hard output
        return hard_round + (smoothed_round - smoothed_round.detach())
