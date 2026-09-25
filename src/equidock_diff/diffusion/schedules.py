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
    *,
    exact: bool = False,
) -> torch.Tensor:
    if nu <= 0.0:
        raise ValueError("cosine nu must be positive.")
    if offset < 0.0:
        raise ValueError("cosine offset must be non-negative.")

    base = torch.clamp(t + offset, min=1e-6, max=1.0 + offset - 1e-5)
    phase = 0.5 * torch.pi * torch.pow(base, nu) / (1.0 + offset)
    phase_derivative = 0.5 * torch.pi * nu * torch.pow(base, nu - 1.0) / (1.0 + offset)
    multiplier = 2.0 if exact else 4.0
    beta = multiplier * torch.tan(phase) * phase_derivative
    return torch.clamp(beta, min=1e-6)

def beta_schedule_value(
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
    *,
    noise_schedule: str,
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
    exact: bool = False,
) -> torch.Tensor:
    if noise_schedule == "linear":
        return linear_beta(t, beta_min, beta_max)
    if noise_schedule == "cosine":
        beta = cosine_beta(t, offset=cosine_offset, nu=cosine_nu, exact=exact)
        return beta.clamp(min=beta_min, max=beta_max)
    raise ValueError(f"Unknown noise schedule: {noise_schedule!r}")


def alpha_bar_for_schedule(
    t: torch.Tensor,
    beta_min: float,
    beta_max: float,
    *,
    noise_schedule: str,
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
) -> torch.Tensor:
    """Signal variance retention ᾱ(t) as the report's schedule definitions intend.

    Linear: ᾱ = exp(-∫₀ᵗ β) with the exact integral (not the β(t)·t shortcut).
    Cosine: ᾱ = cos²(π/2 · (t+s)^ν / (1+s)), i.e. the report's α_t is the VARIANCE
    retention (Nichol–Dhariwal convention), so the x0 std coefficient is √ᾱ.
    """
    if noise_schedule == "linear":
        return torch.exp(-integrated_beta(t, beta_min, beta_max))
    if noise_schedule == "cosine":
        return cosine_signal_amplitude(t, offset=cosine_offset, nu=cosine_nu)
    raise ValueError(f"Unknown noise schedule: {noise_schedule!r}")


def step_beta_from_alpha(
    t: torch.Tensor,
    dt: torch.Tensor,
    beta_min: float,
    beta_max: float,
    *,
    noise_schedule: str,
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
    alpha_floor: float = 1e-6,
) -> torch.Tensor:
    """Effective β for one reverse step, defined by ∫_{t-dt}^{t} β = log ᾱ(t−dt) − log ᾱ(t).

    Makes the reverse solver's accumulated SNR path match the forward process used
    in training exactly, with no [beta_min, beta_max] clamp distorting the path.
    """
    t_prev = torch.clamp(t - dt, min=0.0)
    alpha_t = alpha_bar_for_schedule(
        t,
        beta_min,
        beta_max,
        noise_schedule=noise_schedule,
        cosine_offset=cosine_offset,
        cosine_nu=cosine_nu,
    ).clamp(min=alpha_floor)
    alpha_prev = alpha_bar_for_schedule(
        t_prev,
        beta_min,
        beta_max,
        noise_schedule=noise_schedule,
        cosine_offset=cosine_offset,
        cosine_nu=cosine_nu,
    ).clamp(min=alpha_floor)
    beta_dt = torch.clamp_min(torch.log(alpha_prev) - torch.log(alpha_t), 1e-6)
    return beta_dt / dt
