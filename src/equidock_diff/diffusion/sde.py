"""Diffusion solver interfaces (core logic intentionally left to user)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.dim() < target.dim():
        value = value.unsqueeze(-1)
    return value


@dataclass(frozen=True)
class SDEStep:
    t: torch.Tensor
    dt: torch.Tensor


def forward_step(
    positions: torch.Tensor,
    step: SDEStep,
    beta_t: torch.Tensor,
) -> torch.Tensor:
    dt = step.dt.abs()
    beta_dt = _expand_like(beta_t * dt, positions)
    mean_scale = torch.exp(-0.5 * beta_dt)
    std = torch.sqrt(torch.clamp_min(1.0 - torch.exp(-beta_dt), 1e-8))
    noise = torch.randn_like(positions)
    return mean_scale * positions + std * noise


def reverse_step(
    positions: torch.Tensor,
    step: SDEStep,
    score: torch.Tensor,
    beta_t: torch.Tensor,
) -> torch.Tensor:
    dt = step.dt.abs()
    beta = _expand_like(beta_t, positions)
    drift = 0.5 * beta * positions + beta * score
    noise_scale = torch.sqrt(torch.clamp_min(beta * _expand_like(dt, positions), 1e-8))
    noise = torch.randn_like(positions)
    return positions + drift * _expand_like(dt, positions) + noise_scale * noise
