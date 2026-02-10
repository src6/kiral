"""VP-SDE schedule utilities."""

from __future__ import annotations

import torch


def linear_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


def integrated_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.exp(-integrated_beta(t, beta_min, beta_max))
