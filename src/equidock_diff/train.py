"""Training entry point for the Equidock-Diff pipeline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from argparse import SUPPRESS
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from equidock_diff.data.io import ProteinLigandPaths, load_paths, load_pdbbind_split_paths
from equidock_diff.data.pipeline import load_protein_ligand_graph, load_protein_ligand_graph_cached
from equidock_diff.diffusion.schedules import (
    DEFAULT_COSINE_NU,
    DEFAULT_COSINE_OFFSET,
    beta_schedule_value,
    cosine_signal_amplitude,
)
from equidock_diff.diffusion.sde import SDEStep, forward_step, reverse_step
from equidock_diff.models.egnn import EGNNConfig, infer_ligand_mask
from equidock_diff.models.score_net import ScoreNet, ScoreNetConfig
from equidock_diff.utils.artifacts import (
    write_experiment_log,
    write_ligand_artifacts,
    write_loss_csv,
    write_pdb,
    write_trajectory_pdb,
)
from equidock_diff.utils.geometry import aligned_rmsd, random_rotation_matrix
from equidock_diff.utils.plotting import maybe_write_plot


@dataclass(frozen=True)
class CheckpointState:
    completed_steps: int
    training_seconds: float
    loss_rows: list[tuple[int, float, float]]
    saved_args: dict[str, object]
    graph_source: str
    node_feature_dim: int
    source_ids: list[str] | None = None


@dataclass(frozen=True)
class SamplerStepDiagnostics:
    step_index: int
    t: float
    dt: float
    mean_score_norm_before_clip: float
    max_score_norm_before_clip: float
    mean_score_norm_after_clip: float
    max_score_norm_after_clip: float
    clipped_coordinate_fraction: float
    ligand_center_displacement: float
    ligand_radius: float


@dataclass(frozen=True)
class SamplerDiagnostics:
    complex_id: str | None
    seed: int
    sample_steps: int
    sample_time_power: float
    sample_score_clip: float
    sample_position_clip: float
    anchor_protein: bool
    experiment_log: str | None
    step_metrics: list[SamplerStepDiagnostics]


RESUME_COMPAT_KEYS = (
    "hidden_dim",
    "num_layers",
    "hetero_edges",
    "ligand_global_node",
    "complete_frame",
    "frame_hetero_backbone",
    "learning_rate",
    "ligand_bond_weight",
    "ligand_shape_weight",
    "beta_min",
    "beta_max",
    "noise_schedule",
    "cosine_offset",
    "cosine_nu",
    "protein_path",
    "ligand_path",
    "crop_cutoff",
    "edge_cutoff",
    "batch_size",
    "num_nodes",
    "dataset_root",
    "dataset_split",
    "dataset_limit",
)


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
    parser.add_argument(
        "--ligand-global-node",
        action="store_true",
        help="Add a ligand-wide context update inside each EGNN layer",
    )
    parser.add_argument(
        "--complete-frame",
        action="store_true",
        help="Use a three-vector coordinate basis in EGNN updates",
    )
    parser.add_argument(
        "--hetero-edges",
        action="store_true",
        help="Use typed ligand/protein message transforms in the EGNN backbone",
    )
    parser.add_argument(
        "--frame-hetero-backbone",
        action="store_true",
        help="Use the heterogeneous frame-based backbone with local orientation features",
    )
    parser.add_argument(
        "--hetgnn-backbone",
        dest="frame_hetero_backbone",
        action="store_true",
        help=SUPPRESS,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument(
        "--ligand-bond-weight",
        type=float,
        default=0.0,
        help="Optional weight for ligand bond-length regularization",
    )
    parser.add_argument(
        "--ligand-shape-weight",
        type=float,
        default=0.0,
        help="Optional weight for ligand local shape preservation",
    )
    parser.add_argument("--beta-min", type=float, default=0.1, help="VP-SDE beta minimum")
    parser.add_argument("--beta-max", type=float, default=2.0, help="VP-SDE beta maximum")
    parser.add_argument(
        "--noise-schedule",
        choices=("linear", "cosine"),
        default="linear",
        help="Noise schedule for training and sampling",
    )
    parser.add_argument(
        "--cosine-offset",
        type=float,
        default=DEFAULT_COSINE_OFFSET,
        help="Offset s for cosine scheduling",
    )
    parser.add_argument(
        "--cosine-nu",
        type=float,
        default=DEFAULT_COSINE_NU,
        help="Exponent nu for the modified cosine schedule",
    )
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
        "--sample-time-power",
        type=float,
        default=1.0,
        help="Power-respace reverse-diffusion timesteps; 1.0 preserves uniform spacing",
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
    parser.add_argument(
        "--sampler-diagnostics-json",
        type=Path,
        default=None,
        help="Optional JSON output for per-step sampler diagnostics",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path to save checkpoints during and after training",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Checkpoint interval in steps; 0 disables periodic saves but still saves the final state",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from before continuing training",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional PDBbind root for dataset training mode",
    )
    parser.add_argument(
        "--dataset-split",
        type=Path,
        default=None,
        help="Optional split file containing one complex id per line for dataset mode",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=0,
        help="Optional limit on the number of dataset complexes to use; 0 keeps all",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=Path("data/.cache/equidock_diff_graphs"),
        help="Directory for cached dataset graphs; used only in dataset mode",
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
                use_hetero_edges=args.hetero_edges,
                use_ligand_global_node=args.ligand_global_node,
                use_complete_frame=args.complete_frame,
                use_frame_hetero_backbone=args.frame_hetero_backbone,
            )
        )
    )
    return model.to(device)


def load_graph_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if dataset_mode_enabled(args):
        raise ValueError("load_graph_inputs does not support dataset mode.")

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
            None if batch.ligand_bond_index is None else batch.ligand_bond_index.to(device),
        )

    node_features, positions, edge_index = build_synthetic_graph(args.num_nodes, args.batch_size, device)
    return node_features, positions, edge_index, None


def _checkpoint_arg_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


def checkpoint_args_dict(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for key in dir(args):
        if key.startswith("_"):
            continue
        value = getattr(args, key)
        if callable(value):
            continue
        values[key] = _checkpoint_arg_value(value)
    return values


PATH_ARG_NAMES = {
    "protein_path",
    "ligand_path",
    "output",
    "trajectory_output",
    "ligand_output",
    "ligand_trajectory_output",
    "loss_csv",
    "plot_output",
    "experiment_log",
    "checkpoint_path",
    "resume_from",
    "dataset_root",
    "dataset_split",
    "dataset_cache_dir",
    "sampler_diagnostics_json",
}


def saved_args_to_namespace(saved_args: dict[str, object]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in saved_args.items():
        if not hasattr(args, key):
            continue
        if key in PATH_ARG_NAMES and value is not None:
            setattr(args, key, Path(str(value)))
            continue
        setattr(args, key, value)
    return args


def dataset_mode_enabled(args: argparse.Namespace) -> bool:
    return args.dataset_root is not None or args.dataset_split is not None


def build_dataset_examples(args: argparse.Namespace) -> list[ProteinLigandPaths]:
    dataset_root = args.dataset_root or Path("data/pdbbind_v2020")
    if args.dataset_split is not None:
        examples = load_pdbbind_split_paths(dataset_root, args.dataset_split)
    else:
        examples = load_paths(dataset_root)

    if args.dataset_limit > 0:
        examples = examples[: args.dataset_limit]
    if not examples:
        raise ValueError(f"No dataset protein-ligand pairs found under {dataset_root}.")
    return examples


def load_dataset_example(
    example: ProteinLigandPaths,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch = load_protein_ligand_graph_cached(
        example.protein_path,
        example.ligand_path,
        cutoff=args.crop_cutoff,
        edge_cutoff=args.edge_cutoff,
        cache_dir=args.dataset_cache_dir,
    )
    return (
        batch.node_features.to(device),
        batch.positions.to(device),
        batch.edge_index.to(device),
        None if batch.ligand_bond_index is None else batch.ligand_bond_index.to(device),
    )


def dataset_example_for_step(
    examples: list[ProteinLigandPaths],
    step_number: int,
) -> ProteinLigandPaths:
    if not examples:
        raise ValueError("dataset examples cannot be empty.")
    if step_number <= 0:
        return examples[0]
    return examples[(step_number - 1) % len(examples)]


def _expand_schedule_value(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while value.dim() < target.dim():
        value = value.unsqueeze(-1)
    return value


def noised_positions_for_schedule(
    clean_positions: torch.Tensor,
    t: torch.Tensor,
    *,
    beta_min: float,
    beta_max: float,
    noise_schedule: str,
    cosine_offset: float,
    cosine_nu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta_t = beta_schedule_value(
        t,
        beta_min,
        beta_max,
        noise_schedule=noise_schedule,
        cosine_offset=cosine_offset,
        cosine_nu=cosine_nu,
    )
    if noise_schedule == "cosine":
        alpha_bar_t = cosine_signal_amplitude(t, offset=cosine_offset, nu=cosine_nu)
        signal_scale = torch.sqrt(torch.clamp(alpha_bar_t, min=0.0, max=1.0))
        sigma_t = torch.sqrt(torch.clamp_min(1.0 - alpha_bar_t, 1e-8))
        noised_positions = (
            _expand_schedule_value(signal_scale, clean_positions) * clean_positions
            + _expand_schedule_value(sigma_t, clean_positions) * torch.randn_like(clean_positions)
        )
        return noised_positions, beta_t

    noised_positions = forward_step(clean_positions, SDEStep(t=t, dt=t), beta_t)
    return noised_positions, beta_t


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    completed_steps: int,
    training_seconds: float,
    loss_rows: list[tuple[int, float, float]],
    graph_source: str,
    node_feature_dim: int,
    source_ids: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "completed_steps": completed_steps,
        "training_seconds": training_seconds,
        "loss_rows": loss_rows,
        "graph_source": graph_source,
        "node_feature_dim": node_feature_dim,
        "source_ids": source_ids,
        "saved_args": checkpoint_args_dict(args),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> CheckpointState:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("torch_rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu())

    return CheckpointState(
        completed_steps=int(checkpoint["completed_steps"]),
        training_seconds=float(checkpoint.get("training_seconds", 0.0)),
        loss_rows=[tuple(row) for row in checkpoint.get("loss_rows", [])],
        saved_args=dict(checkpoint.get("saved_args", {})),
        graph_source=str(checkpoint.get("graph_source", "synthetic")),
        node_feature_dim=int(checkpoint.get("node_feature_dim", -1)),
        source_ids=checkpoint.get("source_ids"),
    )


def validate_resume_compatibility(
    args: argparse.Namespace,
    checkpoint_state: CheckpointState,
    *,
    graph_source: str,
    node_feature_dim: int,
    source_ids: list[str] | None = None,
) -> None:
    if args.steps < checkpoint_state.completed_steps:
        raise ValueError(
            "Requested --steps is smaller than the checkpoint step count; increase --steps or use a different checkpoint."
        )

    mismatches: list[str] = []
    for key in RESUME_COMPAT_KEYS:
        current_value = _checkpoint_arg_value(getattr(args, key, None))
        saved_value = checkpoint_state.saved_args.get(key)
        if current_value != saved_value:
            mismatches.append(f"{key}: current={current_value!r}, checkpoint={saved_value!r}")

    if checkpoint_state.graph_source != graph_source:
        mismatches.append(
            f"graph_source: current={graph_source!r}, checkpoint={checkpoint_state.graph_source!r}"
        )
    if checkpoint_state.node_feature_dim != node_feature_dim:
        mismatches.append(
            f"node_feature_dim: current={node_feature_dim!r}, checkpoint={checkpoint_state.node_feature_dim!r}"
        )
    if checkpoint_state.source_ids != source_ids:
        mismatches.append("dataset/source ordering changed between checkpoint and current run")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Checkpoint is not compatible with the current run: {mismatch_text}")


def resolve_checkpoint_output(args: argparse.Namespace) -> Path | None:
    if args.checkpoint_path is not None:
        return args.checkpoint_path
    return args.resume_from


def training_step(
    model: nn.Module,
    node_features: torch.Tensor,
    clean_positions: torch.Tensor,
    edge_index: torch.Tensor,
    beta_min: float,
    beta_max: float,
    ligand_bond_index: torch.Tensor | None = None,
    *,
    noise_schedule: str = "linear",
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
) -> tuple[torch.Tensor, float]:
    frame_hetero_backbone = False
    model_config = getattr(model, "config", None)
    egnn_config = getattr(model_config, "egnn", None)
    if egnn_config is not None:
        frame_hetero_backbone = bool(getattr(egnn_config, "use_frame_hetero_backbone", False))
    loss, beta_t, _, _, _ = training_step_with_breakdown(
        model,
        node_features,
        clean_positions,
        edge_index,
        ligand_bond_index,
        beta_min,
        beta_max,
        ligand_bond_weight=0.0,
        ligand_shape_weight=0.0,
        frame_hetero_backbone=frame_hetero_backbone,
        noise_schedule=noise_schedule,
        cosine_offset=cosine_offset,
        cosine_nu=cosine_nu,
    )
    return loss, beta_t


def ligand_bond_length_loss(
    predicted_positions: torch.Tensor,
    clean_positions: torch.Tensor,
    ligand_bond_index: torch.Tensor | None,
) -> torch.Tensor:
    if ligand_bond_index is None or ligand_bond_index.numel() == 0:
        return predicted_positions.new_zeros(())

    src, dst = ligand_bond_index
    keep = src < dst
    if not keep.any():
        return predicted_positions.new_zeros(())
    src = src[keep]
    dst = dst[keep]

    predicted_lengths = torch.norm(predicted_positions[src] - predicted_positions[dst], dim=-1)
    clean_lengths = torch.norm(clean_positions[src] - clean_positions[dst], dim=-1)
    return torch.mean((predicted_lengths - clean_lengths) ** 2)


def ligand_shape_loss(
    predicted_positions: torch.Tensor,
    clean_positions: torch.Tensor,
    ligand_mask: torch.Tensor | None,
    *,
    cutoff: float = 6.0,
) -> torch.Tensor:
    if ligand_mask is None:
        ligand_predicted = predicted_positions
        ligand_clean = clean_positions
    else:
        if not bool(ligand_mask.any()):
            return predicted_positions.new_zeros(())
        ligand_predicted = predicted_positions[ligand_mask]
        ligand_clean = clean_positions[ligand_mask]

    if ligand_clean.size(0) < 2:
        return predicted_positions.new_zeros(())

    clean_dist = torch.cdist(ligand_clean, ligand_clean)
    predicted_dist = torch.cdist(ligand_predicted, ligand_predicted)
    keep = torch.triu(torch.ones_like(clean_dist, dtype=torch.bool), diagonal=1) & (clean_dist <= cutoff)
    if not bool(keep.any()):
        return predicted_positions.new_zeros(())
    return torch.mean((predicted_dist[keep] - clean_dist[keep]) ** 2)


def training_step_with_breakdown(
    model: nn.Module,
    node_features: torch.Tensor,
    clean_positions: torch.Tensor,
    edge_index: torch.Tensor,
    ligand_bond_index: torch.Tensor | None,
    beta_min: float,
    beta_max: float,
    *,
    ligand_bond_weight: float,
    ligand_shape_weight: float,
    frame_hetero_backbone: bool,
    noise_schedule: str = "linear",
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(1, device=clean_positions.device, dtype=clean_positions.dtype).clamp_(
        0.05, 0.95
    )
    if frame_hetero_backbone:
        ligand_mask = infer_ligand_mask(node_features)
        noised_positions = clean_positions.clone()
        noised_positions[ligand_mask], beta_t = noised_positions_for_schedule(
            clean_positions[ligand_mask],
            t,
            beta_min=beta_min,
            beta_max=beta_max,
            noise_schedule=noise_schedule,
            cosine_offset=cosine_offset,
            cosine_nu=cosine_nu,
        )
        target_score = torch.zeros_like(clean_positions)
        target_score[ligand_mask] = clean_positions[ligand_mask] - noised_positions[ligand_mask]
    else:
        ligand_mask = None
        noised_positions, beta_t = noised_positions_for_schedule(
            clean_positions,
            t,
            beta_min=beta_min,
            beta_max=beta_max,
            noise_schedule=noise_schedule,
            cosine_offset=cosine_offset,
            cosine_nu=cosine_nu,
        )
        target_score = clean_positions - noised_positions
    predicted_score = model(node_features, noised_positions, edge_index, t)
    if frame_hetero_backbone and ligand_mask is not None and bool(ligand_mask.any()):
        predicted_score = predicted_score.clone()
        predicted_score[~ligand_mask] = 0.0
        score_loss = torch.mean((predicted_score[ligand_mask] - target_score[ligand_mask]) ** 2)
    else:
        score_loss = torch.mean((predicted_score - target_score) ** 2)
    bond_loss = clean_positions.new_zeros(())
    shape_loss = clean_positions.new_zeros(())
    predicted_positions = None
    if ligand_bond_weight > 0.0 or ligand_shape_weight > 0.0:
        predicted_positions = noised_positions + predicted_score
    if ligand_bond_weight > 0.0:
        assert predicted_positions is not None
        bond_loss = ligand_bond_length_loss(
            predicted_positions,
            clean_positions,
            ligand_bond_index,
        )
    if ligand_shape_weight > 0.0:
        assert predicted_positions is not None
        shape_loss = ligand_shape_loss(
            predicted_positions,
            clean_positions,
            ligand_mask,
        )
    total_loss = score_loss + ligand_bond_weight * bond_loss + ligand_shape_weight * shape_loss
    return total_loss, float(beta_t.item()), score_loss, bond_loss, shape_loss


def build_sample_schedule(
    sample_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    time_power: float = 1.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive.")
    if time_power <= 0.0:
        raise ValueError("sample_time_power must be positive.")
    boundaries = torch.linspace(0.0, 1.0, sample_steps + 1, device=device, dtype=dtype)
    boundaries = boundaries.pow(time_power)
    schedule: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step_idx in reversed(range(sample_steps)):
        t = boundaries[step_idx + 1]
        dt = boundaries[step_idx + 1] - boundaries[step_idx]
        schedule.append((t, dt))
    return schedule


def _positions_for_diagnostics(
    positions: torch.Tensor,
    ligand_mask: torch.Tensor | None,
) -> torch.Tensor:
    if ligand_mask is None:
        return positions
    if not bool(ligand_mask.any()):
        return positions
    return positions[ligand_mask]


def _sampler_step_diagnostics(
    *,
    step_index: int,
    t: torch.Tensor,
    dt: torch.Tensor,
    score_before_clip: torch.Tensor,
    score_after_clip: torch.Tensor,
    positions_before_clamp: torch.Tensor,
    positions_after_clamp: torch.Tensor,
    previous_positions: torch.Tensor,
    ligand_mask: torch.Tensor | None,
    position_clip: float,
) -> SamplerStepDiagnostics:
    relevant_score_before = _positions_for_diagnostics(score_before_clip, ligand_mask)
    relevant_score_after = _positions_for_diagnostics(score_after_clip, ligand_mask)
    score_before_norm = torch.linalg.norm(relevant_score_before, dim=-1)
    score_after_norm = torch.linalg.norm(relevant_score_after, dim=-1)

    relevant_before_clamp = _positions_for_diagnostics(positions_before_clamp, ligand_mask)
    relevant_after_clamp = _positions_for_diagnostics(positions_after_clamp, ligand_mask)
    relevant_previous = _positions_for_diagnostics(previous_positions, ligand_mask)
    clipped_fraction = (
        float((relevant_before_clamp.abs() > position_clip).float().mean().item())
        if relevant_before_clamp.numel() > 0
        else 0.0
    )
    current_center = relevant_after_clamp.mean(dim=0)
    previous_center = relevant_previous.mean(dim=0)
    center_displacement = float(torch.linalg.norm(current_center - previous_center).item())
    radius = float(
        torch.linalg.norm(relevant_after_clamp - current_center.unsqueeze(0), dim=-1).max().item()
    )
    return SamplerStepDiagnostics(
        step_index=step_index,
        t=float(t.item()),
        dt=float(dt.item()),
        mean_score_norm_before_clip=float(score_before_norm.mean().item()),
        max_score_norm_before_clip=float(score_before_norm.max().item()),
        mean_score_norm_after_clip=float(score_after_norm.mean().item()),
        max_score_norm_after_clip=float(score_after_norm.max().item()),
        clipped_coordinate_fraction=clipped_fraction,
        ligand_center_displacement=center_displacement,
        ligand_radius=radius,
    )


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
    *,
    sample_time_power: float = 1.0,
    noise_schedule: str = "linear",
    cosine_offset: float = DEFAULT_COSINE_OFFSET,
    cosine_nu: float = DEFAULT_COSINE_NU,
    reference_positions: torch.Tensor | None = None,
    anchor_protein: bool = False,
    sampler_diagnostics: SamplerDiagnostics | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], SamplerDiagnostics | None]:
    ligand_mask = infer_ligand_mask(node_features) if anchor_protein else None
    if anchor_protein:
        if reference_positions is None:
            raise ValueError("reference_positions are required when anchor_protein is enabled.")
        positions = reference_positions.clone()
        assert ligand_mask is not None
        positions[ligand_mask] = torch.randn_like(positions[ligand_mask])
    else:
        positions = torch.randn(num_nodes, 3, device=device, dtype=torch.float32)
    trajectory = [positions.detach().cpu().clone()]
    schedule = build_sample_schedule(
        sample_steps,
        device=device,
        dtype=torch.float32,
        time_power=sample_time_power,
    )
    step_diagnostics: list[SamplerStepDiagnostics] = []
    for loop_idx, (t, dt) in enumerate(schedule):
        beta_t = beta_schedule_value(
            t,
            beta_min,
            beta_max,
            noise_schedule=noise_schedule,
            cosine_offset=cosine_offset,
            cosine_nu=cosine_nu,
        )
        score = model(node_features, positions, edge_index, t)
        if anchor_protein and ligand_mask is not None:
            score = score.clone()
            score[~ligand_mask] = 0.0
        score_before_clip = torch.nan_to_num(
            score,
            nan=0.0,
            posinf=score_clip,
            neginf=-score_clip,
        )
        score = score_before_clip.clamp(-score_clip, score_clip)
        previous_positions = positions.detach().clone()
        positions = reverse_step(
            positions,
            SDEStep(t=t, dt=dt),
            score,
            beta_t,
            max_score_norm=score_clip,
        )
        positions = torch.nan_to_num(
            positions,
            nan=0.0,
            posinf=position_clip,
            neginf=-position_clip,
        )
        positions_before_clamp = positions.detach().clone()
        if anchor_protein:
            assert reference_positions is not None
            assert ligand_mask is not None
            positions = positions.clone()
            positions[~ligand_mask] = reference_positions[~ligand_mask]
        else:
            positions = positions - positions.mean(dim=0, keepdim=True)
        positions = positions.clamp(-position_clip, position_clip)
        if sampler_diagnostics is not None:
            step_diagnostics.append(
                _sampler_step_diagnostics(
                    step_index=loop_idx,
                    t=t,
                    dt=dt,
                    score_before_clip=score_before_clip,
                    score_after_clip=score,
                    positions_before_clamp=positions_before_clamp,
                    positions_after_clamp=positions,
                    previous_positions=previous_positions,
                    ligand_mask=ligand_mask,
                    position_clip=position_clip,
                )
            )
        trajectory.append(positions.detach().cpu().clone())
    diagnostics = None
    if sampler_diagnostics is not None:
        diagnostics = SamplerDiagnostics(
            complex_id=sampler_diagnostics.complex_id,
            seed=sampler_diagnostics.seed,
            sample_steps=sample_steps,
            sample_time_power=sample_time_power,
            sample_score_clip=score_clip,
            sample_position_clip=position_clip,
            anchor_protein=anchor_protein,
            experiment_log=sampler_diagnostics.experiment_log,
            step_metrics=step_diagnostics,
        )
    return positions, trajectory, diagnostics


def write_sampler_diagnostics(path: Path, diagnostics: SamplerDiagnostics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "complex_id": diagnostics.complex_id,
        "seed": diagnostics.seed,
        "sample_steps": diagnostics.sample_steps,
        "sample_time_power": diagnostics.sample_time_power,
        "sample_score_clip": diagnostics.sample_score_clip,
        "sample_position_clip": diagnostics.sample_position_clip,
        "anchor_protein": diagnostics.anchor_protein,
        "experiment_log": diagnostics.experiment_log,
        "steps": [asdict(step) for step in diagnostics.step_metrics],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")
    if args.dataset_limit < 0:
        raise ValueError("--dataset-limit must be non-negative.")
    if args.sample_time_power <= 0.0:
        raise ValueError("--sample-time-power must be positive.")
    if args.ligand_shape_weight < 0.0:
        raise ValueError("--ligand-shape-weight must be non-negative.")
    if dataset_mode_enabled(args) and (args.protein_path is not None or args.ligand_path is not None):
        raise ValueError("Dataset mode cannot be combined with --protein-path/--ligand-path.")
    if dataset_mode_enabled(args) and args.batch_size != 1:
        raise ValueError("Dataset mode currently supports only batch size 1.")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    dataset_examples = build_dataset_examples(args) if dataset_mode_enabled(args) else None
    source_ids = None if dataset_examples is None else [entry.complex_id for entry in dataset_examples]
    if dataset_examples is not None:
        node_features, positions, edge_index, ligand_bond_index = load_dataset_example(
            dataset_examples[0],
            args,
            device,
        )
        graph_source = "dataset"
    else:
        node_features, positions, edge_index, ligand_bond_index = load_graph_inputs(args, device)
        graph_source = "real_pair" if args.protein_path is not None else "synthetic"

    if args.dry_run:
        rot = random_rotation_matrix(1, device=device, dtype=torch.float32)
        print(f"Device: {device}")
        print(f"Rotation sample shape: {rot.shape}")
        model = make_model_for_node_dim(args, device, node_dim=node_features.size(-1))
        with torch.no_grad():
            score = model(node_features, positions, edge_index, torch.tensor(0.5, device=device))
        print("Graph source: " + graph_source)
        print(f"Noise schedule: {args.noise_schedule}")
        if dataset_examples is not None:
            print(f"Dataset size: {len(dataset_examples)}")
            print(f"First complex: {dataset_examples[0].complex_id}")
            print(f"Dataset cache dir: {args.dataset_cache_dir}")
        print(f"Score sample shape: {score.shape}")
        return 0

    model = make_model_for_node_dim(args, device, node_dim=node_features.size(-1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_output = resolve_checkpoint_output(args)
    resumed_state: CheckpointState | None = None
    if args.resume_from is not None:
        resumed_state = load_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        validate_resume_compatibility(
            args,
            resumed_state,
            graph_source=graph_source,
            node_feature_dim=node_features.size(-1),
            source_ids=source_ids,
        )
        print(f"resumed_from={args.resume_from}")
        print(f"resume_step={resumed_state.completed_steps}")
        if dataset_examples is not None:
            print(f"dataset_size={len(dataset_examples)}")
            print(f"dataset_cache_dir={args.dataset_cache_dir}")

    completed_steps = 0 if resumed_state is None else resumed_state.completed_steps
    prior_training_seconds = 0.0 if resumed_state is None else resumed_state.training_seconds
    loss_rows: list[tuple[int, float, float]] = [] if resumed_state is None else list(
        resumed_state.loss_rows
    )

    start = perf_counter()
    for step_idx in range(completed_steps + 1, args.steps + 1):
        current_complex_id = None
        if dataset_examples is not None:
            current_example = dataset_example_for_step(dataset_examples, step_idx)
            node_features, positions, edge_index, ligand_bond_index = load_dataset_example(
                current_example,
                args,
                device,
            )
            current_complex_id = current_example.complex_id

        optimizer.zero_grad(set_to_none=True)
        loss, beta_t, score_loss, bond_loss, shape_loss = training_step_with_breakdown(
            model,
            node_features,
            positions,
            edge_index,
            ligand_bond_index,
            args.beta_min,
            args.beta_max,
            ligand_bond_weight=args.ligand_bond_weight,
            ligand_shape_weight=args.ligand_shape_weight,
            frame_hetero_backbone=args.frame_hetero_backbone,
            noise_schedule=args.noise_schedule,
            cosine_offset=args.cosine_offset,
            cosine_nu=args.cosine_nu,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_rows.append((step_idx, float(loss.item()), beta_t))

        if step_idx == 1 or step_idx == args.steps or step_idx % max(args.steps // 5, 1) == 0:
            line = f"step={step_idx} loss={loss.item():.6f} beta_t={beta_t:.4f}"
            if current_complex_id is not None:
                line += f" complex={current_complex_id}"
            if args.ligand_bond_weight > 0.0:
                line += (
                    f" score_loss={score_loss.item():.6f}"
                    f" bond_loss={bond_loss.item():.6f}"
                )
            if args.ligand_shape_weight > 0.0:
                line += f" shape_loss={shape_loss.item():.6f}"
            print(line)

        if checkpoint_output is not None and args.checkpoint_every > 0:
            if step_idx % args.checkpoint_every == 0:
                elapsed_for_checkpoint = prior_training_seconds + (perf_counter() - start)
                save_checkpoint(
                    checkpoint_output,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    completed_steps=step_idx,
                    training_seconds=elapsed_for_checkpoint,
                    loss_rows=loss_rows,
                    graph_source=graph_source,
                    node_feature_dim=node_features.size(-1),
                    source_ids=source_ids,
                )
                print(f"checkpoint_path={checkpoint_output}")

    elapsed = prior_training_seconds + (perf_counter() - start)
    if completed_steps == args.steps:
        print("resume_checkpoint_already_complete=true")
    print(f"training_seconds={elapsed:.3f}")
    write_loss_csv(args.loss_csv, loss_rows)
    print(f"loss_csv={args.loss_csv}")
    if checkpoint_output is not None:
        save_checkpoint(
            checkpoint_output,
            model=model,
            optimizer=optimizer,
            args=args,
            completed_steps=args.steps,
            training_seconds=elapsed,
            loss_rows=loss_rows,
            graph_source=graph_source,
            node_feature_dim=node_features.size(-1),
            source_ids=source_ids,
        )
        print(f"checkpoint_path={checkpoint_output}")

    if dataset_examples is not None:
        sample_example = dataset_example_for_step(dataset_examples, args.steps)
        node_features, positions, edge_index, ligand_bond_index = load_dataset_example(
            sample_example,
            args,
            device,
        )
        print(f"sample_complex={sample_example.complex_id}")

    with torch.no_grad():
        sampler_context = None
        if args.sampler_diagnostics_json is not None:
            complex_id = None
            if dataset_examples is not None:
                complex_id = sample_example.complex_id
            elif args.protein_path is not None:
                complex_id = args.protein_path.name.replace("_protein.pdb", "")
            sampler_context = SamplerDiagnostics(
                complex_id=complex_id,
                seed=args.seed,
                sample_steps=args.sample_steps,
                sample_time_power=args.sample_time_power,
                sample_score_clip=args.sample_score_clip,
                sample_position_clip=args.sample_position_clip,
                anchor_protein=args.frame_hetero_backbone,
                experiment_log=str(args.experiment_log) if args.experiment_log is not None else None,
                step_metrics=[],
            )
        sampled_positions, trajectory, sampler_diagnostics = sample_positions(
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
            sample_time_power=args.sample_time_power,
            noise_schedule=args.noise_schedule,
            cosine_offset=args.cosine_offset,
            cosine_nu=args.cosine_nu,
            reference_positions=positions,
            anchor_protein=args.frame_hetero_backbone,
            sampler_diagnostics=sampler_context,
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
    extra_metrics: dict[str, float] = {}
    ligand_mask = node_features[:, -1] > 0.5
    if graph_source in {"real_pair", "dataset"} and bool(ligand_mask.any()):
        reference_ligand = positions[ligand_mask]
        sampled_ligand = sampled_positions[ligand_mask]
        raw_ligand_rmse = torch.sqrt(
            torch.mean((sampled_ligand - reference_ligand) ** 2)
        ).item()
        aligned_ligand = aligned_rmsd(sampled_ligand, reference_ligand).item()
        extra_metrics["raw_ligand_rmse"] = raw_ligand_rmse
        extra_metrics["aligned_ligand_rmsd"] = aligned_ligand
        print(f"raw_ligand_rmse={raw_ligand_rmse:.6f}")
        print(f"aligned_ligand_rmsd={aligned_ligand:.6f}")
    plot_written = maybe_write_plot(args.plot_output, trajectory, loss_rows)
    if plot_written:
        print(f"plot_path={args.plot_output}")
    else:
        print("plot_path=not_written (matplotlib not available)")
    if args.sampler_diagnostics_json is not None and sampler_diagnostics is not None:
        write_sampler_diagnostics(args.sampler_diagnostics_json, sampler_diagnostics)
        print(f"sampler_diagnostics_json={args.sampler_diagnostics_json}")
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
            extra_metrics=extra_metrics or None,
        )
        print(f"experiment_log={args.experiment_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
