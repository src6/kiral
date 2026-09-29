from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
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


def write_loss_terms_csv(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    """Write the per-term loss breakdown.

    A single averaged loss hides which component is failing: the score term outvotes the geometry
    terms, so a total that looks flat can sit on top of a clash or bond term that never improves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "score", "bond", "shape", "clash"])
        writer.writerows(rows)


LOSS_TERM_COLUMNS = ("loss", "score", "bond", "shape", "clash")


def mean_loss_terms(rows: Iterable[Sequence[float]]) -> tuple[float, ...]:
    """Average each loss term across the rows of one evaluation pass.

    Validation complexes are scored one at a time (dataset mode is batch size 1), so the number
    reported per term has to be an explicit mean over complexes. Reporting the last row instead
    would make the held-out trace depend on the order the split file happens to be written in.
    """
    materialized = [tuple(float(value) for value in row) for row in rows]
    if not materialized:
        raise ValueError("mean_loss_terms requires at least one row.")
    width = len(materialized[0])
    if any(len(row) != width for row in materialized):
        raise ValueError("all rows must carry the same number of loss terms.")
    count = len(materialized)
    return tuple(sum(row[column] for row in materialized) / count for column in range(width))


def write_validation_loss_csv(
    path: Path,
    rows: list[tuple[int, float, float, float, float, float]],
) -> None:
    """Write the held-out validation trace to its own file.

    Kept out of ``loss_terms.csv`` on purpose: that file has one row per training step on whichever
    complex the step sampled, so folding a per-evaluation mean over held-out complexes into it would
    leave a single file carrying two different granularities.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *LOSS_TERM_COLUMNS])
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
    retained_protein_nodes: int | None = None,
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
        f"- Use cross interface block: `{bool(getattr(args, 'use_cross_interface_block', False))}`",
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
    if getattr(args, "context_policy", "fixed") == "gated":
        lines.append(f"- Protein node budget: `{getattr(args, 'protein_node_budget', '')}`")
    if retained_protein_nodes is not None:
        lines.append(f"- Retained protein nodes: `{retained_protein_nodes}`")
    if extra_metrics is not None:
        for key, value in extra_metrics.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{value:.6f}`")
    if plot_path is not None:
        lines.append(f"- Plot artifact: `{plot_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
