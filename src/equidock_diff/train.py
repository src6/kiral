"""Training entry point for the Equidock-Diff MVP."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from equidock_diff.data.pipeline import load_protein_ligand_graph
from equidock_diff.diffusion.schedules import linear_beta
from equidock_diff.diffusion.sde import SDEStep, forward_step, reverse_step
from equidock_diff.models.score_net import ScoreNet, ScoreNetConfig
from equidock_diff.models.egnn import EGNNConfig
from equidock_diff.utils.artifacts import (
    write_experiment_log,
    write_ligand_artifacts,
    write_loss_csv,
    write_pdb,
    write_trajectory_pdb,
)
from equidock_diff.utils.geometry import random_rotation_matrix
from equidock_diff.utils.plotting import maybe_write_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Equidock-Diff training entry point")
    parser.add_argument("--device", default="mps", help="Device: mps or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup only")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--num-nodes", type=int, default=12, help="Nodes per synthetic graph")
    parser.add_argument(
        "--protein-path",
        type=Path,
        default=None,
        help="Path to one protein PDB file for the real-pair path",
    )
    parser.add_argument(
        "--ligand-path",
        type=Path,
        default=None,
        help="Path to one ligand SDF file for the real-pair path",
    )
    parser.add_argument(
        "--crop-cutoff",
        type=float,
        default=10.0,
        help="Protein crop cutoff in Angstrom for the real-pair path",
    )
    parser.add_argument(
        "--edge-cutoff",
        type=float,
        default=4.5,
        help="Edge radius in Angstrom for the real-pair path",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of EGNN layers")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--beta-min", type=float, default=0.1, help="VP-SDE beta minimum")
    parser.add_argument("--beta-max", type=float, default=2.0, help="VP-SDE beta maximum")
    parser.add_argument("--sample-steps", type=int, default=25, help="Reverse diffusion steps")
    parser.add_argument(
        "--sample-score-clip",
        type=float,
        default=10.0,
        help="Clamp score predictions during reverse diffusion",
    )
    parser.add_argument(
        "--sample-position-clip",
        type=float,
        default=50.0,
        help="Clamp coordinates during reverse diffusion",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/training/synthetic_sample.pdb"),
        help="Path for the synthetic sample artifact",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("docs/training/synthetic_trajectory.pdb"),
        help="Path for the reverse diffusion trajectory artifact",
    )
    parser.add_argument(
        "--ligand-output",
        type=Path,
        default=None,
        help="Optional path for a ligand-only sample artifact",
    )
    parser.add_argument(
        "--ligand-trajectory-output",
        type=Path,
        default=None,
        help="Optional path for a ligand-only reverse diffusion trajectory artifact",
    )
    parser.add_argument(
        "--loss-csv",
        type=Path,
        default=Path("docs/training/loss_trace.csv"),
        help="Path for the per-step loss log",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("docs/training/trajectory_plot.png"),
        help="Path for the optional trajectory plot image",
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=None,
        help="Optional markdown log capturing command, settings, and key metrics",
    )
    return parser


def resolve_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_synthetic_graph(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.zeros(batch_size * num_nodes, 4, device=device, dtype=torch.float32)
    positions = []
    edges = []

    for batch_idx in range(batch_size):
        offset = batch_idx * num_nodes
        graph_pos = torch.randn(num_nodes, 3, device=device, dtype=torch.float32)
        graph_pos = graph_pos - graph_pos.mean(dim=0, keepdim=True)
        graph_pos[: num_nodes // 2] *= 0.5
        positions.append(graph_pos)

        ligand_mask = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        ligand_mask[: num_nodes // 2] = 1.0
        features[offset : offset + num_nodes, 0] = ligand_mask
        features[offset : offset + num_nodes, 1] = 1.0 - ligand_mask
        features[offset : offset + num_nodes, 2] = torch.linspace(
            0.0, 1.0, num_nodes, device=device
        )
        features[offset : offset + num_nodes, 3] = batch_idx

        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                edges.append((offset + src, offset + dst))

    edge_index = torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()
    return features, torch.cat(positions, dim=0), edge_index


def make_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    node_dim: int = 4,
) -> ScoreNet:
    return make_model_for_node_dim(args, device, node_dim=node_dim)


def make_model_for_node_dim(
    args: argparse.Namespace,
    device: torch.device,
    *,
    node_dim: int,
) -> ScoreNet:
    model = ScoreNet(
        ScoreNetConfig(
            egnn=EGNNConfig(
                node_dim=node_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
            )
        )
    )
    return model.to(device)


def load_graph_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    has_protein = args.protein_path is not None
    has_ligand = args.ligand_path is not None
    if has_protein != has_ligand:
        raise ValueError("Pass both --protein-path and --ligand-path to use the real-pair path.")

    if has_protein and has_ligand:
        batch = load_protein_ligand_graph(
            args.protein_path,
            args.ligand_path,
            cutoff=args.crop_cutoff,
            edge_cutoff=getattr(args, "edge_cutoff", 4.5),
        )
        return (
            batch.node_features.to(device),
            batch.positions.to(device),
            batch.edge_index.to(device),
        )

    return build_synthetic_graph(args.num_nodes, args.batch_size, device)


def training_step(
    model: nn.Module,
    node_features: torch.Tensor,
    clean_positions: torch.Tensor,
    edge_index: torch.Tensor,
    beta_min: float,
    beta_max: float,
) -> tuple[torch.Tensor, float]:
    t = torch.rand(1, device=clean_positions.device, dtype=clean_positions.dtype).clamp_(
        0.05, 0.95
    )
    beta_t = linear_beta(t, beta_min, beta_max)
    noised_positions = forward_step(clean_positions, SDEStep(t=t, dt=t), beta_t)
    target_score = clean_positions - noised_positions
    predicted_score = model(node_features, noised_positions, edge_index, t)
    loss = torch.mean((predicted_score - target_score) ** 2)
    return loss, float(beta_t.item())


def sample_positions(
    model: nn.Module,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    sample_steps: int,
    beta_min: float,
    beta_max: float,
    score_clip: float,
    position_clip: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    positions = torch.randn(num_nodes, 3, device=device, dtype=torch.float32)
    trajectory = [positions.detach().cpu().clone()]
    dt = torch.tensor(1.0 / sample_steps, device=device, dtype=torch.float32)
    for step_idx in reversed(range(sample_steps)):
        t = torch.tensor(
            (step_idx + 1) / sample_steps, device=device, dtype=torch.float32
        )
        beta_t = linear_beta(t, beta_min, beta_max)
        score = model(node_features, positions, edge_index, t)
        score = torch.nan_to_num(score, nan=0.0, posinf=score_clip, neginf=-score_clip)
        score = score.clamp(-score_clip, score_clip)
        positions = reverse_step(positions, SDEStep(t=t, dt=dt), score, beta_t)
        positions = torch.nan_to_num(
            positions,
            nan=0.0,
            posinf=position_clip,
            neginf=-position_clip,
        )
        positions = positions - positions.mean(dim=0, keepdim=True)
        positions = positions.clamp(-position_clip, position_clip)
        trajectory.append(positions.detach().cpu().clone())
    return positions, trajectory


def main() -> int:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    node_features, positions, edge_index = load_graph_inputs(args, device)
    graph_source = "real_pair" if args.protein_path is not None else "synthetic"

    if args.dry_run:
        rot = random_rotation_matrix(1, device=device, dtype=torch.float32)
        print(f"Device: {device}")
        print(f"Rotation sample shape: {rot.shape}")
        model = make_model_for_node_dim(args, device, node_dim=node_features.size(-1))
        with torch.no_grad():
            score = model(node_features, positions, edge_index, torch.tensor(0.5, device=device))
        print("Graph source: " + graph_source)
        print(f"Score sample shape: {score.shape}")
        return 0

    model = make_model_for_node_dim(args, device, node_dim=node_features.size(-1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = perf_counter()
    loss_rows: list[tuple[int, float, float]] = []
    for step_idx in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, beta_t = training_step(
            model,
            node_features,
            positions,
            edge_index,
            args.beta_min,
            args.beta_max,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_rows.append((step_idx, float(loss.item()), beta_t))

        if step_idx == 1 or step_idx == args.steps or step_idx % max(args.steps // 5, 1) == 0:
            print(f"step={step_idx} loss={loss.item():.6f} beta_t={beta_t:.4f}")

    elapsed = perf_counter() - start
    print(f"training_seconds={elapsed:.3f}")
    write_loss_csv(args.loss_csv, loss_rows)
    print(f"loss_csv={args.loss_csv}")

    with torch.no_grad():
        sampled_positions, trajectory = sample_positions(
            model,
            node_features,
            edge_index,
            positions.size(0),
            device,
            args.sample_steps,
            args.beta_min,
            args.beta_max,
            args.sample_score_clip,
            args.sample_position_clip,
        )
    write_pdb(args.output, sampled_positions)
    write_trajectory_pdb(args.trajectory_output, trajectory)
    print(f"sample_path={args.output}")
    print(f"trajectory_path={args.trajectory_output}")
    ligand_sample_path, ligand_trajectory_path = write_ligand_artifacts(
        node_features=node_features,
        sampled_positions=sampled_positions,
        trajectory=trajectory,
        ligand_output=args.ligand_output,
        ligand_trajectory_output=args.ligand_trajectory_output,
    )
    if ligand_sample_path is not None:
        print(f"ligand_sample_path={ligand_sample_path}")
    if ligand_trajectory_path is not None:
        print(f"ligand_trajectory_path={ligand_trajectory_path}")
    plot_written = maybe_write_plot(args.plot_output, trajectory, loss_rows)
    if plot_written:
        print(f"plot_path={args.plot_output}")
    else:
        print("plot_path=not_written (matplotlib not available)")
    if args.experiment_log is not None:
        write_experiment_log(
            args.experiment_log,
            command=f"uv run python -m equidock_diff.train {shlex.join(sys.argv[1:])}",
            device=device,
            graph_source=graph_source,
            args=args,
            loss_rows=loss_rows,
            training_seconds=elapsed,
            node_count=positions.size(0),
            edge_count=edge_index.size(1),
            sample_path=args.output,
            trajectory_path=args.trajectory_output,
            loss_csv_path=args.loss_csv,
            plot_path=args.plot_output if plot_written else None,
        )
        print(f"experiment_log={args.experiment_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
