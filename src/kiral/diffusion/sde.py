"""Diffusion solver interfaces (core logic intentionally left to user)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.dim() < target.dim():
        value = value.unsqueeze(-1)
    return value


def clip_score_norm(score: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    if max_norm is None or max_norm <= 0.0:
        return score

    score_norm = torch.linalg.norm(score, dim=-1, keepdim=True)
    safe_norm = torch.clamp_min(score_norm, 1e-8)
    scale = torch.clamp(max_norm / safe_norm, max=1.0)
    return score * scale


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
    max_score_norm: float | None = None,
) -> torch.Tensor:
    dt = step.dt.abs()
    score = clip_score_norm(score, max_score_norm)
    beta = _expand_like(beta_t, positions)
    drift = 0.5 * beta * positions + beta * score
    noise_scale = torch.sqrt(torch.clamp_min(beta * _expand_like(dt, positions), 1e-8))
    noise = torch.randn_like(positions)
    return positions + drift * _expand_like(dt, positions) + noise_scale * noise
