from __future__ import annotations

import pytest
import torch

from equidock_diff.diffusion.schedules import (
    beta_schedule_value,
    cosine_beta,
    cosine_signal_amplitude,
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
