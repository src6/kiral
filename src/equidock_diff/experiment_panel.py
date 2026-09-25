"""Build a deterministic dissertation evaluation panel from the local dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from equidock_diff.data.io import ProteinLigandPaths, filter_paths_by_complex_ids, load_paths
from equidock_diff.data.pipeline import load_protein_ligand_graph
from equidock_diff.data.validate_pdbbind import count_empty_crops
from equidock_diff.models.egnn import EGNNConfig
from equidock_diff.models.score_net import ScoreNet, ScoreNetConfig
from equidock_diff.train import resolve_device, training_step_with_breakdown
from equidock_diff.utils.chemistry import featurize_ligand


DEFAULT_REQUIRED_COMPLEX_IDS = ("10gs", "11gs", "1a30")


@dataclass(frozen=True)
class PanelSelectionConfig:
    root: Path
    output: Path
    target_size: int
    crop_cutoff: float
    edge_cutoff: float
    smoke_steps: int
    seed: int
    device: str
    hidden_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 3e-4
    ligand_bond_weight: float = 0.1
    beta_min: float = 0.1
    beta_max: float = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the fixed dissertation evaluation panel from the local PDBbind layout",
    )
    parser.add_argument("--root", type=Path, default=Path("data/pdbbind_v2020"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/evaluation/dissertation_panel20.txt"),
    )
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--crop-cutoff", type=float, default=8.0)
    parser.add_argument("--edge-cutoff", type=float, default=4.5)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def passes_crop_validation(entry: ProteinLigandPaths, *, cutoff: float) -> bool:
    counts = count_empty_crops([entry], cutoff=cutoff, limit=1)
    return counts["checked_pairs"] == 1 and counts["empty_crops"] == 0 and counts["coord_parse_failed"] == 0


def ligand_featurizes(entry: ProteinLigandPaths) -> bool:
    outcome = featurize_ligand(entry.ligand_path)
    return not outcome.skipped and outcome.graph is not None


def smoke_run_is_finite(
    entry: ProteinLigandPaths,
    config: PanelSelectionConfig,
) -> bool:
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    batch = load_protein_ligand_graph(
        entry.protein_path,
        entry.ligand_path,
        cutoff=config.crop_cutoff,
        edge_cutoff=config.edge_cutoff,
    )
    node_features = batch.node_features.to(device)
    positions = batch.positions.to(device)
    edge_index = batch.edge_index.to(device)
    ligand_bond_index = None
    if batch.ligand_bond_index is not None:
        ligand_bond_index = batch.ligand_bond_index.to(device)

    model = ScoreNet(
        ScoreNetConfig(
            egnn=EGNNConfig(
                node_dim=node_features.size(-1),
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for _ in range(config.smoke_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _, _, _ = training_step_with_breakdown(
            model,
            node_features,
            positions,
            edge_index,
            ligand_bond_index,
            config.beta_min,
            config.beta_max,
            ligand_bond_weight=config.ligand_bond_weight,
            ligand_shape_weight=0.0,
            ligand_protein_clash_weight=0.0,
            ligand_protein_contact_weight=0.0,
            frame_hetero_backbone=False,
        )
        if not torch.isfinite(loss):
            return False
        loss.backward()
        optimizer.step()
    return True


def select_panel(
    paths: list[ProteinLigandPaths],
    config: PanelSelectionConfig,
) -> list[ProteinLigandPaths]:
    required = filter_paths_by_complex_ids(paths, DEFAULT_REQUIRED_COMPLEX_IDS)
    by_complex_id = {entry.complex_id: entry for entry in paths}

    ordered_candidates: list[ProteinLigandPaths] = []
    seen: set[str] = set()
    for entry in list(required) + paths:
        if entry.complex_id in seen:
            continue
        seen.add(entry.complex_id)
        ordered_candidates.append(entry)

    selected: list[ProteinLigandPaths] = []
    for entry in ordered_candidates:
        if not passes_crop_validation(entry, cutoff=config.crop_cutoff):
            continue
        if not ligand_featurizes(entry):
            continue
        if not smoke_run_is_finite(entry, config):
            continue
        selected.append(entry)
        if len(selected) == config.target_size:
            return selected

    raise ValueError(
        f"Unable to find {config.target_size} valid complexes under {config.root}; found {len(selected)}."
    )


def write_manifest(path: Path, entries: list[ProteinLigandPaths]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fixed dissertation evaluation panel",
        f"# Complex count: {len(entries)}",
        "",
    ]
    lines.extend(entry.complex_id for entry in entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    config = PanelSelectionConfig(
        root=args.root,
        output=args.output,
        target_size=args.target_size,
        crop_cutoff=args.crop_cutoff,
        edge_cutoff=args.edge_cutoff,
        smoke_steps=args.smoke_steps,
        seed=args.seed,
        device=args.device,
    )
    paths = load_paths(config.root)
    if not paths:
        raise ValueError(f"No protein-ligand pairs found under {config.root}.")
    panel = select_panel(paths, config)
    write_manifest(config.output, panel)
    print(f"panel_size={len(panel)}")
    print(f"output_path={config.output}")
    for entry in panel:
        print(entry.complex_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
