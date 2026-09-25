from __future__ import annotations

import pytest
import torch

from equidock_diff.diffusion.schedules import (
    alpha_bar_for_schedule,
    beta_schedule_value,
    cosine_beta,
    cosine_signal_amplitude,
    integrated_beta,
    linear_beta,
    step_beta_from_alpha,
)


def test_cosine_signal_amplitude_starts_at_one_and_decays() -> None:
    t = torch.tensor([0.0, 0.25, 0.5, 0.95], dtype=torch.float32)

    alpha_bar = cosine_signal_amplitude(t, offset=0.008, nu=1.5)

    assert alpha_bar[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert torch.all(alpha_bar[:-1] >= alpha_bar[1:])
    assert torch.all(alpha_bar >= 0.0)


def test_cosine_signal_amplitude_returns_alpha_bar_quantity() -> None:
    t = torch.tensor([0.5], dtype=torch.float32)

    alpha_bar = cosine_signal_amplitude(t, offset=0.008, nu=1.5)
    signal_scale = torch.sqrt(alpha_bar)

    base = torch.clamp(t + 0.008, min=0.0, max=1.0 + 0.008)
    phase = 0.5 * torch.pi * torch.pow(base, 1.5) / (1.0 + 0.008)
    expected_alpha_bar = torch.cos(phase) ** 2

    assert alpha_bar.item() == pytest.approx(expected_alpha_bar.item())
    assert alpha_bar.item() < signal_scale.item() < 1.0


def test_cosine_beta_is_positive_and_finite() -> None:
    t = torch.linspace(0.05, 0.95, steps=8, dtype=torch.float32)

    beta = cosine_beta(t, offset=0.008, nu=1.5)

    assert torch.isfinite(beta).all()
    assert torch.all(beta > 0.0)


def test_beta_schedule_value_dispatches_cosine() -> None:
    t = torch.tensor([0.5], dtype=torch.float32)

    linear = beta_schedule_value(t, 0.1, 2.0, noise_schedule="linear")
    cosine = beta_schedule_value(
        t,
        0.1,
        2.0,
        noise_schedule="cosine",
        cosine_offset=0.008,
        cosine_nu=1.5,
    )

    assert torch.isfinite(linear).all()
    assert torch.isfinite(cosine).all()
    assert linear.item() != pytest.approx(cosine.item())
    assert 0.1 <= cosine.item() <= 2.0


def test_linear_integrated_beta_matches_numerical_integral() -> None:
    beta_min = 0.1
    beta_max = 2.0
    t_eval = torch.tensor([0.2, 0.5, 0.8, 1.0], dtype=torch.float64)

    # Check against numerical trapezoid integration
    for t in t_eval:
        steps = 10000
        s = torch.linspace(0.0, t.item(), steps=steps, dtype=torch.float64)
        betas = linear_beta(s, beta_min, beta_max)
        numerical_integral = torch.trapezoid(betas, s).item()
        analytical_integral = integrated_beta(t, beta_min, beta_max).item()
        assert analytical_integral == pytest.approx(numerical_integral, rel=1e-5)

        # Explicitly verify the identity: alpha_bar = exp(-integrated_beta)
        alpha_bar = alpha_bar_for_schedule(
            t.float(), beta_min, beta_max, noise_schedule="linear"
        )
        expected_alpha = torch.exp(-integrated_beta(t.float(), beta_min, beta_max))
        assert alpha_bar.item() == pytest.approx(expected_alpha.item(), rel=1e-6)


def test_cosine_exact_beta_matches_autograd_negative_log_derivative() -> None:
    offset = 0.008
    nu = 1.5
    t = torch.linspace(0.1, 0.9, steps=9, dtype=torch.float64, requires_grad=True)

    base = torch.clamp(t + offset, min=0.0, max=1.0 + offset)
    phase = 0.5 * torch.pi * torch.pow(base, nu) / (1.0 + offset)
    alpha = torch.cos(phase) ** 2
    log_alpha = torch.log(alpha)

    # d/dt log(alpha) = -beta(t)  ==> beta(t) = -d/dt log(alpha)
    grad_outputs = torch.ones_like(log_alpha)
    (d_log_alpha,) = torch.autograd.grad(log_alpha, t, grad_outputs=grad_outputs)
    expected_exact_beta = -d_log_alpha

    computed_exact_beta = cosine_beta(t.detach().float(), offset=offset, nu=nu, exact=True)
    assert computed_exact_beta.numpy() == pytest.approx(
        expected_exact_beta.detach().numpy(), rel=1e-4, abs=1e-4
    )

    # Legacy factor-of-2 behavior verification: legacy beta is 2x exact beta
    legacy_beta = cosine_beta(t.detach().float(), offset=offset, nu=nu, exact=False)
    assert legacy_beta.numpy() == pytest.approx(
        (2.0 * computed_exact_beta).numpy(), rel=1e-5
    )


def test_step_beta_from_alpha_accumulates_exact_snr() -> None:
    beta_min = 0.1
    beta_max = 2.0
    t = torch.tensor([0.6], dtype=torch.float32)
    dt = torch.tensor([0.04], dtype=torch.float32)

    step_b = step_beta_from_alpha(
        t,
        dt,
        beta_min,
        beta_max,
        noise_schedule="linear",
    )

    alpha_t = alpha_bar_for_schedule(
        t, beta_min, beta_max, noise_schedule="linear"
    )
    alpha_prev = alpha_bar_for_schedule(
        t - dt, beta_min, beta_max, noise_schedule="linear"
    )

    # Identity: beta_step * dt = log(alpha_prev) - log(alpha_t)
    assert (step_b * dt).item() == pytest.approx(
        (torch.log(alpha_prev) - torch.log(alpha_t)).item(), rel=1e-5
    )
