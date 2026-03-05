from __future__ import annotations

import torch

from equidock_diff.diffusion.sde import SDEStep, clip_score_norm, forward_step, reverse_step


def test_forward_step_preserves_shape_and_finiteness() -> None:
    positions = torch.randn(10, 3)
    step = SDEStep(t=torch.tensor(0.4), dt=torch.tensor(0.1))
    out = forward_step(positions, step, beta_t=torch.tensor(0.5))

    assert out.shape == positions.shape
    assert torch.isfinite(out).all()


def test_reverse_step_preserves_shape_and_finiteness() -> None:
    positions = torch.randn(10, 3)
    score = torch.randn(10, 3)
    step = SDEStep(t=torch.tensor(0.4), dt=torch.tensor(0.1))
    out = reverse_step(positions, step, score, beta_t=torch.tensor(0.5))

    assert out.shape == positions.shape
    assert torch.isfinite(out).all()


def test_clip_score_norm_limits_vector_magnitude() -> None:
    score = torch.tensor([[3.0, 4.0, 0.0], [0.1, 0.2, 0.2]], dtype=torch.float32)

    clipped = clip_score_norm(score, max_norm=1.0)

    norms = torch.linalg.norm(clipped, dim=-1)
    assert torch.all(norms <= 1.00001)
    assert torch.allclose(clipped[1], score[1])
