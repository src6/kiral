from __future__ import annotations

import pytest
import torch

from equidock_diff.diffusion.dpm_solver import (
    dpm_solver_second_order_step,
    sample_positions_dpm,
)


class DummyScoreModel(torch.nn.Module):
    def __init__(self, target_positions: torch.Tensor):
        super().__init__()
        self.target_positions = target_positions

    def forward(self, h, pos, edge_idx, t):
        # Predict displacement toward target
        return self.target_positions - pos


def test_dpm_solver_second_order_step_runs() -> None:
    num_nodes = 6
    node_features = torch.zeros(num_nodes, 8, dtype=torch.float32)
    positions = torch.randn(num_nodes, 3, dtype=torch.float32)
    reference = torch.zeros(num_nodes, 3, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    model = DummyScoreModel(reference)

    t = torch.tensor(0.5, dtype=torch.float32)
    dt = torch.tensor(0.05, dtype=torch.float32)

    next_pos = dpm_solver_second_order_step(
        model=model,
        node_features=node_features,
        positions=positions,
        edge_index=edge_index,
        t=t,
        dt=dt,
        beta_min=0.1,
        beta_max=2.0,
        noise_schedule="cosine",
    )

    assert next_pos.shape == positions.shape
    assert torch.isfinite(next_pos).all()


def test_sample_positions_dpm_trajectory_and_anchoring() -> None:
    num_nodes = 8
    node_features = torch.zeros(num_nodes, 11, dtype=torch.float32)
    node_features[:4, -1] = 0.0  # protein
    node_features[4:, -1] = 1.0  # ligand

    reference_positions = torch.randn(num_nodes, 3, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    model = DummyScoreModel(reference_positions)

    steps = 10
    sampled, trajectory = sample_positions_dpm(
        model=model,
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        device=torch.device("cpu"),
        sample_steps=steps,
        noise_schedule="cosine",
        reference_positions=reference_positions,
        anchor_protein=True,
    )

    assert torch.isfinite(sampled).all()
    assert len(trajectory) == steps + 1
    # Protein atoms must remain strictly anchored to crystal reference coordinates
    assert torch.allclose(sampled[:4], reference_positions[:4], atol=1e-5)
    # Ligand atoms should be pulled toward target
    assert torch.allclose(sampled[4:], reference_positions[4:], atol=0.2)
