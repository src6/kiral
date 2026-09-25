from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LigandStructure:
    coords: np.ndarray
    bonds: list[tuple[int, int]]


@dataclass(frozen=True)
class ComplexFigureSpec:
    complex_id: str
    title: str
    pocket_pdb: Path
    reference_sdf: Path
    baseline_pdb: Path
    improved_pdb: Path


def parse_pdb_coords(path: Path) -> np.ndarray:
    coords: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
    if not coords:
        raise ValueError(f"No coordinates found in {path}")
    return np.asarray(coords, dtype=float)


def parse_sdf_structure(path: Path) -> LigandStructure:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError(f"SDF too short: {path}")
    counts = lines[3]
    atom_count = int(counts[:3])
    bond_count = int(counts[3:6])
    atom_lines = lines[4 : 4 + atom_count]
    bond_lines = lines[4 + atom_count : 4 + atom_count + bond_count]
    atom_coords = []
    keep_indices: list[int] = []
    old_to_new: dict[int, int] = {}
    for index, line in enumerate(atom_lines):
        element = line[31:34].strip()
        if element == "H":
            continue
        old_to_new[index] = len(keep_indices)
        keep_indices.append(index)
        atom_coords.append([float(line[:10]), float(line[10:20]), float(line[20:30])])
    bonds: list[tuple[int, int]] = []
    for line in bond_lines:
        left = int(line[:3]) - 1
        right = int(line[3:6]) - 1
        if left in old_to_new and right in old_to_new:
            bonds.append((old_to_new[left], old_to_new[right]))
    return LigandStructure(coords=np.asarray(atom_coords, dtype=float), bonds=bonds)


def project_points(*arrays: np.ndarray) -> list[np.ndarray]:
    stacked = np.vstack(arrays)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]]
    projected: list[np.ndarray] = []
    start = 0
    for array in arrays:
        end = start + len(array)
        projected.append(centered[start:end] @ basis)
        start = end
    return projected


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mobile_center = mobile.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center
    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    return mobile_centered @ rotation + target_center


def draw_structure(ax, coords_2d: np.ndarray, bonds: list[tuple[int, int]], *, color: str, alpha: float) -> None:
    for left, right in bonds:
        ax.plot(
            [coords_2d[left, 0], coords_2d[right, 0]],
            [coords_2d[left, 1], coords_2d[right, 1]],
            color=color,
            linewidth=2.2,
            alpha=alpha,
            zorder=2,
        )
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        s=38,
        color=color,
        edgecolors="white",
        linewidths=0.8,
        alpha=alpha,
        zorder=3,
    )


def draw_panel(ax, spec: ComplexFigureSpec) -> None:
    reference = parse_sdf_structure(spec.reference_sdf)
    pocket = parse_pdb_coords(spec.pocket_pdb)
    baseline = kabsch_align(parse_pdb_coords(spec.baseline_pdb), reference.coords)
    improved = kabsch_align(parse_pdb_coords(spec.improved_pdb), reference.coords)
    pocket_2d, reference_2d, baseline_2d, improved_2d = project_points(
        pocket, reference.coords, baseline, improved
    )

    ax.scatter(
        pocket_2d[:, 0],
        pocket_2d[:, 1],
        s=16,
        color="#cfd3d8",
        alpha=0.35,
        zorder=1,
    )
    draw_structure(ax, baseline_2d, reference.bonds, color="#d96c6c", alpha=0.88)
    draw_structure(ax, improved_2d, reference.bonds, color="#2aa198", alpha=0.9)
    draw_structure(ax, reference_2d, reference.bonds, color="#d8a106", alpha=0.95)
    ax.set_title(spec.title, fontsize=22, pad=14)
    ax.set_aspect("equal")
    ax.axis("off")


def build_specs(repo_root: Path) -> list[ComplexFigureSpec]:
    data_root = repo_root / "data" / "pdbbind_v2020" / "protein_ligand_general_minus_refined" / "1981-2000"
    return [
        ComplexFigureSpec(
            complex_id="13gs",
            title="13gs: 2.80 -> 1.09 A",
            pocket_pdb=data_root / "13gs" / "13gs_pocket.pdb",
            reference_sdf=data_root / "13gs" / "13gs_ligand.sdf",
            baseline_pdb=repo_root
            / "docs/training/panel20/schedule_cpu_rerun/13gs_baseline_cosine_seed42_cpu_rerun_ligand_sample.pdb",
            improved_pdb=repo_root
            / "docs/training/panel20/schedule_cpu_rerun/13gs_frame_backbone_cosine_seed42_cpu_rerun_ligand_sample.pdb",
        ),
        ComplexFigureSpec(
            complex_id="16pk",
            title="16pk: 2.67 -> 1.48 A",
            pocket_pdb=data_root / "16pk" / "16pk_pocket.pdb",
            reference_sdf=data_root / "16pk" / "16pk_ligand.sdf",
            baseline_pdb=repo_root
            / "docs/training/panel20/schedule_cpu_rerun/16pk_baseline_cosine_seed42_cpu_rerun_ligand_sample.pdb",
            improved_pdb=repo_root
            / "docs/training/panel20/schedule_cpu_rerun/16pk_frame_backbone_cosine_seed42_cpu_rerun_ligand_sample.pdb",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the current showcase pose comparison figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/training/showcase/panel20_pose_comparison_current.png"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    specs = build_specs(repo_root)
    figure, axes = plt.subplots(1, len(specs), figsize=(17.64, 6.72), dpi=100)
    if len(specs) == 1:
        axes = [axes]
    for ax, spec in zip(axes, specs, strict=True):
        draw_panel(ax, spec)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, transparent=False, facecolor="white")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
