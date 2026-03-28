from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def write_pdb(path: Path, positions: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, coord in enumerate(positions.detach().cpu(), start=1):
        x, y, z = coord.tolist()
        lines.append(
            f"ATOM  {idx:5d}  CA  GLY A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trajectory_pdb(path: Path, trajectory: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame_idx, positions in enumerate(trajectory, start=1):
        lines.append(f"MODEL     {frame_idx:4d}")
        for atom_idx, coord in enumerate(positions, start=1):
            x, y, z = coord.tolist()
            lines.append(
                f"ATOM  {atom_idx:5d}  CA  GLY A{atom_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ligand_mask_from_features(node_features: torch.Tensor) -> torch.Tensor:
    if node_features.size(-1) < 1:
        raise ValueError("node_features must include the ligand indicator column.")
    return node_features[:, -1] > 0.5


def write_ligand_artifacts(
    *,
    node_features: torch.Tensor,
    sampled_positions: torch.Tensor,
    trajectory: list[torch.Tensor],
    ligand_output: Path | None,
    ligand_trajectory_output: Path | None,
) -> tuple[Path | None, Path | None]:
    ligand_mask = ligand_mask_from_features(node_features).detach().cpu()
    ligand_positions = sampled_positions.detach().cpu()[ligand_mask]
    ligand_trajectory = [frame[ligand_mask] for frame in trajectory]

    written_sample = None
    if ligand_output is not None:
        write_pdb(ligand_output, ligand_positions)
        written_sample = ligand_output

    written_trajectory = None
    if ligand_trajectory_output is not None:
        write_trajectory_pdb(ligand_trajectory_output, ligand_trajectory)
        written_trajectory = ligand_trajectory_output

    return written_sample, written_trajectory


def write_loss_csv(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "beta_t"])
        writer.writerows(rows)


def write_experiment_log(
    path: Path,
    *,
    command: str,
    device: torch.device,
    graph_source: str,
    args: argparse.Namespace,
    loss_rows: list[tuple[int, float, float]],
    training_seconds: float,
    node_count: int,
    edge_count: int,
    sample_path: Path | None,
    trajectory_path: Path | None,
    loss_csv_path: Path | None,
    plot_path: Path | None,
    extra_metrics: dict[str, float] | None = None,
    resolved_crop_cutoff: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_step, final_loss, final_beta = loss_rows[-1]
    best_loss = min(row[1] for row in loss_rows)
    lines = [
        "# Experiment Log",
        "",
        f"- Command: `{command}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{device}`",
        f"- Graph source: `{graph_source}`",
        f"- Training steps: `{args.steps}`",
        f"- Sample steps: `{args.sample_steps}`",
        f"- Node count: `{node_count}`",
        f"- Edge count: `{edge_count}`",
        f"- Crop cutoff: `{args.crop_cutoff}`",
        f"- Context policy: `{getattr(args, 'context_policy', 'fixed')}`",
        f"- Edge cutoff: `{args.edge_cutoff}`",
        f"- Final loss: `{final_loss:.6f}` at step `{final_step}`",
        f"- Best loss: `{best_loss:.6f}`",
        f"- Final beta_t: `{final_beta:.4f}`",
        f"- Training seconds: `{training_seconds:.3f}`",
    ]
    if loss_csv_path is not None:
        lines.append(f"- Loss CSV: `{loss_csv_path}`")
    if sample_path is not None:
        lines.append(f"- Sample artifact: `{sample_path}`")
    if trajectory_path is not None:
        lines.append(f"- Trajectory artifact: `{trajectory_path}`")
    if resolved_crop_cutoff is not None:
        lines.append(f"- Resolved crop cutoff: `{resolved_crop_cutoff:.6f}`")
    if extra_metrics is not None:
        for key, value in extra_metrics.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{value:.6f}`")
    if plot_path is not None:
        lines.append(f"- Plot artifact: `{plot_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
