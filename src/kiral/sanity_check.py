"""SE(3) sanity check script for Equidock-Diff."""

from __future__ import annotations

import argparse

import torch

from kiral.models.egnn import EGNNConfig
from kiral.models.score_net import ScoreNet, ScoreNetConfig
from kiral.train import build_synthetic_graph, resolve_device
from kiral.utils.geometry import apply_rigid_transform, random_rotation_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Equidock-Diff SE(3) sanity check")
    parser.add_argument("--device", default="mps", help="Device: mps or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-nodes", type=int, default=12, help="Nodes per graph")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of EGNN layers")
    parser.add_argument("--trials", type=int, default=8, help="Number of random rotations")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    model = ScoreNet(
        ScoreNetConfig(
            egnn=EGNNConfig(
                node_dim=4,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
            )
        )
    ).to(device)
    model.eval()

    node_features, positions, edge_index = build_synthetic_graph(args.num_nodes, 1, device)
    time = torch.tensor(0.5, device=device, dtype=torch.float32)

    trial_errors = []
    with torch.no_grad():
        base_score = model(node_features, positions, edge_index, time)
        for _ in range(args.trials):
            rotation = random_rotation_matrix(1, device=device, dtype=torch.float32)[0]
            translation = torch.randn(3, device=device, dtype=torch.float32)
            transformed_positions = apply_rigid_transform(positions, rotation, translation)
            transformed_score = model(node_features, transformed_positions, edge_index, time)
            expected_score = apply_rigid_transform(base_score, rotation, torch.zeros_like(translation))
            error = (transformed_score - expected_score).norm(dim=-1)
            trial_errors.append(error)

    errors = torch.cat(trial_errors)
    print(f"device={device}")
    print(f"trials={args.trials}")
    print(f"mean_equivariance_error={errors.mean().item():.6e}")
    print(f"max_equivariance_error={errors.max().item():.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
