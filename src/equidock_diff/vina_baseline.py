"""Helpers for running and evaluating an AutoDock Vina baseline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from equidock_diff.utils.geometry import aligned_rmsd


@dataclass(frozen=True)
class VinaResult:
    raw_ligand_rmse: float
    aligned_ligand_rmsd: float
    runtime_seconds: float
    receptor_pdbqt: Path
    ligand_pdbqt: Path
    output_pdbqt: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight AutoDock Vina baseline")
    parser.add_argument("--protein-path", type=Path, required=True)
    parser.add_argument("--ligand-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vina-binary", default="vina")
    parser.add_argument("--receptor-pdbqt", type=Path, default=None)
    parser.add_argument("--ligand-pdbqt", type=Path, default=None)
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=1)
    return parser


def ligand_centroid_from_sdf(ligand_path: Path) -> torch.Tensor:
    ligand_positions = reference_ligand_positions_with_rdkit(ligand_path)
    return ligand_positions.mean(dim=0)


def reference_ligand_positions_with_rdkit(ligand_path: Path) -> torch.Tensor:
    from equidock_diff.utils.chemistry import featurize_ligand

    outcome = featurize_ligand(ligand_path)
    if outcome.skipped or outcome.graph is None:
        reason = outcome.skip_reason or "unknown"
        raise ValueError(f"Unable to featurize ligand {ligand_path}: {reason}")
    return outcome.graph.pos


def build_vina_command(
    *,
    vina_binary: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_pdbqt: Path,
    center: torch.Tensor,
    box_size: float,
    exhaustiveness: int,
    cpu: int,
) -> list[str]:
    return [
        vina_binary,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--out",
        str(output_pdbqt),
        "--center_x",
        f"{float(center[0]):.4f}",
        "--center_y",
        f"{float(center[1]):.4f}",
        "--center_z",
        f"{float(center[2]):.4f}",
        "--size_x",
        f"{box_size:.4f}",
        "--size_y",
        f"{box_size:.4f}",
        "--size_z",
        f"{box_size:.4f}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--cpu",
        str(cpu),
    ]


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise FileNotFoundError(f"Required executable not found on PATH: {name}")
    return binary


def prepare_pdbqt_inputs(
    *,
    protein_path: Path,
    ligand_path: Path,
    output_dir: Path,
    receptor_pdbqt: Path | None,
    ligand_pdbqt: Path | None,
    prepare_inputs: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_target = receptor_pdbqt or output_dir / f"{protein_path.stem}.pdbqt"
    ligand_target = ligand_pdbqt or output_dir / f"{ligand_path.stem}.pdbqt"

    if receptor_pdbqt is not None and ligand_pdbqt is not None:
        return receptor_pdbqt, ligand_pdbqt
    if not prepare_inputs:
        raise ValueError(
            "Pass both --receptor-pdbqt and --ligand-pdbqt, or enable --prepare-inputs."
        )

    receptor_binary = _require_binary("mk_prepare_receptor.py")
    ligand_binary = _require_binary("mk_prepare_ligand.py")
    subprocess.run(
        [
            receptor_binary,
            "-i",
            str(protein_path),
            "-o",
            str(receptor_target),
        ],
        check=True,
    )
    subprocess.run(
        [
            ligand_binary,
            "-i",
            str(ligand_path),
            "-o",
            str(ligand_target),
        ],
        check=True,
    )
    return receptor_target, ligand_target


def parse_pdbqt_positions(path: Path) -> torch.Tensor:
    coords: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            coords.append(
                [
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip()),
                ]
            )
    if not coords:
        raise ValueError(f"No docked coordinates found in {path}")
    return torch.tensor(coords, dtype=torch.float32)


def evaluate_docked_pose(
    reference_positions: torch.Tensor,
    docked_positions: torch.Tensor,
) -> tuple[float, float]:
    if reference_positions.shape != docked_positions.shape:
        raise ValueError(
            f"Reference and docked ligand shapes differ: {reference_positions.shape} vs {docked_positions.shape}"
        )
    raw_rmse = torch.sqrt(torch.mean((docked_positions - reference_positions) ** 2)).item()
    aligned = aligned_rmsd(docked_positions, reference_positions).item()
    return raw_rmse, aligned


def run_vina_baseline(args: argparse.Namespace) -> VinaResult:
    receptor_pdbqt, ligand_pdbqt = prepare_pdbqt_inputs(
        protein_path=args.protein_path,
        ligand_path=args.ligand_path,
        output_dir=args.output_dir,
        receptor_pdbqt=args.receptor_pdbqt,
        ligand_pdbqt=args.ligand_pdbqt,
        prepare_inputs=args.prepare_inputs,
    )
    center = ligand_centroid_from_sdf(args.ligand_path)
    output_pdbqt = args.output_dir / f"{args.ligand_path.stem}_vina_out.pdbqt"
    vina_binary = _require_binary(args.vina_binary)
    command = build_vina_command(
        vina_binary=vina_binary,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        output_pdbqt=output_pdbqt,
        center=center,
        box_size=args.box_size,
        exhaustiveness=args.exhaustiveness,
        cpu=args.cpu,
    )

    start = perf_counter()
    subprocess.run(command, check=True)
    runtime_seconds = perf_counter() - start

    reference_positions = reference_ligand_positions_with_rdkit(args.ligand_path)
    docked_positions = parse_pdbqt_positions(output_pdbqt)
    raw_rmse, aligned = evaluate_docked_pose(reference_positions, docked_positions)
    return VinaResult(
        raw_ligand_rmse=raw_rmse,
        aligned_ligand_rmsd=aligned,
        runtime_seconds=runtime_seconds,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        output_pdbqt=output_pdbqt,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = run_vina_baseline(args)
    print(f"receptor_pdbqt={result.receptor_pdbqt}")
    print(f"ligand_pdbqt={result.ligand_pdbqt}")
    print(f"output_pdbqt={result.output_pdbqt}")
    print(f"runtime_seconds={result.runtime_seconds:.3f}")
    print(f"raw_ligand_rmse={result.raw_ligand_rmse:.6f}")
    print(f"aligned_ligand_rmsd={result.aligned_ligand_rmsd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
