"""Compute lightweight structural diagnostics for a set of complexes."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from equidock_diff.data.io import (
    ProteinLigandPaths,
    filter_paths_by_complex_ids,
    load_paths,
    load_split_complex_ids,
)
from equidock_diff.data.pipeline import load_protein_ligand_graph
from equidock_diff.utils.chemistry import featurize_ligand


DEFAULT_CROP_CUTOFF = 10.0
DEFAULT_EDGE_CUTOFF = 4.5
WIDE_PROXIMITY_CUTOFF = 8.0


@dataclass(frozen=True)
class ComplexDiagnosticsRow:
    complex_id: str
    ligand_heavy_atoms: int
    ligand_bond_count: int
    ligand_mean_degree: float
    ligand_max_degree: int
    ligand_cyclomatic_number: int
    ligand_radius: float
    ligand_max_pairwise_span: float
    cropped_protein_nodes: int
    ligand_protein_contacts_edge_cutoff: int
    ligand_protein_contacts_wide_cutoff: int
    protein_ligand_centroid_distance: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute structural diagnostics for a manifest of protein-ligand complexes"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="PDBbind dataset root")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest to describe")
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=None,
        help="Optional comparison manifest for cohort-level contrast",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV output path")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        required=True,
        help="Markdown summary output path",
    )
    return parser


def _ligand_unique_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    unique = {
        tuple(sorted((int(src), int(dst))))
        for src, dst in edge_index.t().tolist()
        if int(src) != int(dst)
    }
    return sorted(unique)


def _connected_components(num_nodes: int, unique_edges: list[tuple[int, int]]) -> int:
    if num_nodes == 0:
        return 0
    parent = list(range(num_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for src, dst in unique_edges:
        union(src, dst)
    return len({find(node) for node in range(num_nodes)})


def _max_pairwise_span(positions: torch.Tensor) -> float:
    if positions.size(0) < 2:
        return 0.0
    return float(torch.cdist(positions, positions).max().item())


def compute_row(paths: ProteinLigandPaths) -> ComplexDiagnosticsRow:
    ligand_outcome = featurize_ligand(paths.ligand_path)
    if ligand_outcome.skipped or ligand_outcome.graph is None:
        reason = ligand_outcome.skip_reason or "unknown"
        raise ValueError(f"Unable to featurize {paths.ligand_path}: {reason}")
    ligand_graph = ligand_outcome.graph
    batch = load_protein_ligand_graph(
        paths.protein_path,
        paths.ligand_path,
        cutoff=DEFAULT_CROP_CUTOFF,
        edge_cutoff=DEFAULT_EDGE_CUTOFF,
    )
    batch_ligand_mask = batch.node_features[:, -1] > 0.5
    ligand_positions = batch.positions[batch_ligand_mask]
    protein_positions = batch.positions[~batch_ligand_mask]
    distances = torch.cdist(ligand_positions, protein_positions) if protein_positions.numel() > 0 else None

    unique_edges = _ligand_unique_edges(ligand_graph.edge_index)
    degree = torch.bincount(ligand_graph.edge_index[0], minlength=ligand_graph.x.size(0))
    components = _connected_components(ligand_graph.x.size(0), unique_edges)
    ligand_center = ligand_graph.pos.mean(dim=0)
    protein_center = protein_positions.mean(dim=0) if protein_positions.numel() > 0 else ligand_center
    return ComplexDiagnosticsRow(
        complex_id=paths.complex_id,
        ligand_heavy_atoms=ligand_graph.x.size(0),
        ligand_bond_count=len(unique_edges),
        ligand_mean_degree=float(degree.float().mean().item()) if degree.numel() > 0 else 0.0,
        ligand_max_degree=int(degree.max().item()) if degree.numel() > 0 else 0,
        ligand_cyclomatic_number=len(unique_edges) - ligand_graph.x.size(0) + components,
        ligand_radius=float(torch.linalg.norm(ligand_graph.pos - ligand_center.unsqueeze(0), dim=-1).max().item()),
        ligand_max_pairwise_span=_max_pairwise_span(ligand_graph.pos),
        cropped_protein_nodes=int((~batch_ligand_mask).sum().item()),
        ligand_protein_contacts_edge_cutoff=int((distances <= DEFAULT_EDGE_CUTOFF).sum().item()) if distances is not None else 0,
        ligand_protein_contacts_wide_cutoff=int((distances <= WIDE_PROXIMITY_CUTOFF).sum().item()) if distances is not None else 0,
        protein_ligand_centroid_distance=float(torch.linalg.norm(protein_center - ligand_center).item()),
    )


def load_rows(dataset_root: Path, manifest: Path) -> list[ComplexDiagnosticsRow]:
    paths = filter_paths_by_complex_ids(load_paths(dataset_root), load_split_complex_ids(manifest))
    return [compute_row(path) for path in paths]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _diagnostic_note(
    focus_rows: list[ComplexDiagnosticsRow],
    comparison_rows: list[ComplexDiagnosticsRow],
) -> str:
    if not comparison_rows:
        return "No comparison cohort was provided."
    metrics = {
        "ligand_heavy_atoms": ("larger ligands", "smaller ligands"),
        "ligand_mean_degree": ("higher ligand graph degree", "lower ligand graph degree"),
        "ligand_cyclomatic_number": ("more cyclic ligand graphs", "less cyclic ligand graphs"),
        "ligand_radius": ("larger ligand spatial radius", "smaller ligand spatial radius"),
        "cropped_protein_nodes": (
            "denser cropped protein neighborhoods",
            "sparser cropped protein neighborhoods",
        ),
        "ligand_protein_contacts_edge_cutoff": (
            "more short-range protein-ligand contacts",
            "fewer short-range protein-ligand contacts",
        ),
        "protein_ligand_centroid_distance": (
            "larger protein-ligand centroid separation",
            "smaller protein-ligand centroid separation",
        ),
    }
    observations: list[str] = []
    for field, labels in metrics.items():
        focus_mean = _mean([float(getattr(row, field)) for row in focus_rows])
        comparison_mean = _mean([float(getattr(row, field)) for row in comparison_rows])
        if math.isclose(comparison_mean, 0.0, abs_tol=1e-8):
            continue
        ratio = focus_mean / comparison_mean
        if ratio >= 1.2:
            observations.append(f"hard cases trend toward {labels[0]}")
        elif ratio <= 0.8:
            observations.append(f"hard cases trend toward {labels[1]}")
    if not observations:
        return "The hard-case subset does not show a strong size, density, or graph-complexity split under these descriptors."
    return "; ".join(observations[:3]) + "."


def write_csv(
    path: Path,
    focus_rows: list[ComplexDiagnosticsRow],
    comparison_rows: list[ComplexDiagnosticsRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cohort",
                "complex_id",
                "ligand_heavy_atoms",
                "ligand_bond_count",
                "ligand_mean_degree",
                "ligand_max_degree",
                "ligand_cyclomatic_number",
                "ligand_radius",
                "ligand_max_pairwise_span",
                "cropped_protein_nodes",
                "ligand_protein_contacts_edge_cutoff",
                "ligand_protein_contacts_wide_cutoff",
                "protein_ligand_centroid_distance",
            ]
        )
        for cohort, rows in (("focus", focus_rows), ("comparison", comparison_rows)):
            for row in rows:
                writer.writerow(
                    [
                        cohort,
                        row.complex_id,
                        row.ligand_heavy_atoms,
                        row.ligand_bond_count,
                        f"{row.ligand_mean_degree:.6f}",
                        row.ligand_max_degree,
                        row.ligand_cyclomatic_number,
                        f"{row.ligand_radius:.6f}",
                        f"{row.ligand_max_pairwise_span:.6f}",
                        row.cropped_protein_nodes,
                        row.ligand_protein_contacts_edge_cutoff,
                        row.ligand_protein_contacts_wide_cutoff,
                        f"{row.protein_ligand_centroid_distance:.6f}",
                    ]
                )


def write_markdown(
    path: Path,
    focus_rows: list[ComplexDiagnosticsRow],
    comparison_rows: list[ComplexDiagnosticsRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Complex Diagnostics",
        "",
        "## Hard-Case Rows",
        "",
        "| Complex | Heavy Atoms | Bonds | Mean Degree | Max Degree | Cyclomatic | Ligand Radius | Max Span | Cropped Protein Nodes | Contacts <=4.5A | Contacts <=8.0A | Centroid Distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus_rows:
        lines.append(
            f"| `{row.complex_id}` | `{row.ligand_heavy_atoms}` | `{row.ligand_bond_count}` | "
            f"`{row.ligand_mean_degree:.3f}` | `{row.ligand_max_degree}` | `{row.ligand_cyclomatic_number}` | "
            f"`{row.ligand_radius:.3f}` | `{row.ligand_max_pairwise_span:.3f}` | `{row.cropped_protein_nodes}` | "
            f"`{row.ligand_protein_contacts_edge_cutoff}` | `{row.ligand_protein_contacts_wide_cutoff}` | "
            f"`{row.protein_ligand_centroid_distance:.3f}` |"
        )
    lines.extend(["", "## Cohort Contrast", ""])
    if comparison_rows:
        lines.extend(
            [
                "| Metric | Hard Cases Mean | Comparison Mean |",
                "| --- | ---: | ---: |",
                f"| Heavy atoms | `{_mean([row.ligand_heavy_atoms for row in focus_rows]):.3f}` | `{_mean([row.ligand_heavy_atoms for row in comparison_rows]):.3f}` |",
                f"| Mean degree | `{_mean([row.ligand_mean_degree for row in focus_rows]):.3f}` | `{_mean([row.ligand_mean_degree for row in comparison_rows]):.3f}` |",
                f"| Cyclomatic number | `{_mean([row.ligand_cyclomatic_number for row in focus_rows]):.3f}` | `{_mean([row.ligand_cyclomatic_number for row in comparison_rows]):.3f}` |",
                f"| Ligand radius | `{_mean([row.ligand_radius for row in focus_rows]):.3f}` | `{_mean([row.ligand_radius for row in comparison_rows]):.3f}` |",
                f"| Cropped protein nodes | `{_mean([row.cropped_protein_nodes for row in focus_rows]):.3f}` | `{_mean([row.cropped_protein_nodes for row in comparison_rows]):.3f}` |",
                f"| Contacts <=4.5A | `{_mean([row.ligand_protein_contacts_edge_cutoff for row in focus_rows]):.3f}` | `{_mean([row.ligand_protein_contacts_edge_cutoff for row in comparison_rows]):.3f}` |",
                f"| Centroid distance | `{_mean([row.protein_ligand_centroid_distance for row in focus_rows]):.3f}` | `{_mean([row.protein_ligand_centroid_distance for row in comparison_rows]):.3f}` |",
                "",
                "## Diagnostic Note",
                "",
                _diagnostic_note(focus_rows, comparison_rows),
            ]
        )
    else:
        lines.append("No comparison cohort was provided.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    focus_rows = load_rows(args.dataset_root, args.manifest)
    comparison_rows: list[ComplexDiagnosticsRow] = []
    if args.comparison_manifest is not None:
        comparison_rows = load_rows(args.dataset_root, args.comparison_manifest)
    write_csv(args.output_csv, focus_rows, comparison_rows)
    write_markdown(args.output_markdown, focus_rows, comparison_rows)
    print(f"csv_path={args.output_csv}")
    print(f"markdown_path={args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
