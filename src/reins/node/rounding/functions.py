"""
Straight-through estimators for nondifferentiable operators.
"""

import torch
from torch import nn


class DiffFloor(nn.Module):
    """Differentiable floor using straight-through estimator."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # STE: Attach gradient from x
        return x + (torch.floor(x) - x).detach()


class DiffBinarize(nn.Module):
    """Differentiable binarize using straight-through estimator."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Clamp to [-1, 1] then binarize at 0
        x = torch.clamp(x, min=-1, max=1)
        # STE: Attach gradient from x
        return x + ((x >= 0).float() - x).detach()


class DiffGumbelBinarize(nn.Module):
    """
    Stochastic differentiable binarize using Gumbel-Softmax trick.

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
            # Deterministic threshold in eval with STE for gradient flow
            soft = torch.sigmoid(x / self.temperature)
            hard = (x > 0).float()
            return soft + (hard - soft).detach()


class GumbelThresholdBinarize(nn.Module):
    """
    Stochastic threshold binarization with Gumbel-Softmax noise.

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
            # Deterministic threshold in eval with STE for gradient flow
            soft = torch.sigmoid(diff / self.temperature)
            hard = (diff >= 0).float()
            return soft + (hard - soft).detach()


class ThresholdBinarize(nn.Module):
    """
    Deterministic threshold binarization with sigmoid STE.

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
