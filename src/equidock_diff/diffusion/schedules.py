"""VP-SDE schedule utilities."""

from __future__ import annotations

import torch

DEFAULT_COSINE_OFFSET = 0.008
DEFAULT_COSINE_NU = 1.5


def linear_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


def integrated_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min * t + 0.5 * (beta_max - beta_min) * t**2


def alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.exp(-integrated_beta(t, beta_min, beta_max))


def cosine_signal_amplitude(
    t: torch.Tensor,
    offset: float = DEFAULT_COSINE_OFFSET,
    nu: float = DEFAULT_COSINE_NU,
) -> torch.Tensor:
    """Return the cosine alpha_bar schedule used by the VP-SDE forward process."""

    if nu <= 0.0:
        raise ValueError("cosine nu must be positive.")
    if offset < 0.0:
        raise ValueError("cosine offset must be non-negative.")

    base = torch.clamp(t + offset, min=0.0, max=1.0 + offset)
    phase = 0.5 * torch.pi * torch.pow(base, nu) / (1.0 + offset)
    alpha = torch.cos(phase) ** 2
    return alpha.clamp(min=0.0, max=1.0)


def cosine_beta(
    t: torch.Tensor,
    offset: float = DEFAULT_COSINE_OFFSET,
    nu: float = DEFAULT_COSINE_NU,
) -> torch.Tensor:
    if nu <= 0.0:
        raise ValueError("cosine nu must be positive.")
    if offset < 0.0:
        raise ValueError("cosine offset must be non-negative.")

    base = torch.clamp(t + offset, min=1e-6, max=1.0 + offset - 1e-5)
    phase = 0.5 * torch.pi * torch.pow(base, nu) / (1.0 + offset)
    phase_derivative = 0.5 * torch.pi * nu * torch.pow(base, nu - 1.0) / (1.0 + offset)
    beta = 4.0 * torch.tan(phase) * phase_derivative
    return torch.clamp(beta, min=1e-6)


def beta_schedule_value(
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
    *,
    noise_schedule: str,
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
) -> torch.Tensor:
    if noise_schedule == "linear":
        return linear_beta(t, beta_min, beta_max)
    if noise_schedule == "cosine":
        beta = cosine_beta(t, offset=cosine_offset, nu=cosine_nu)
        return beta.clamp(min=beta_min, max=beta_max)
    raise ValueError(f"Unknown noise schedule: {noise_schedule!r}")
