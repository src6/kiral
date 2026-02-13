from __future__ import annotations

import torch

from equidock_diff.train import build_synthetic_graph, make_model, training_step


class _Args:
    hidden_dim = 32
    num_layers = 2


def test_training_step_is_finite() -> None:
    device = torch.device("cpu")
    model = make_model(_Args(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0
