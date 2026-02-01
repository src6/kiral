"""Data I/O utilities for the local PDBbind layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Iterable


@dataclass(frozen=True)
class ProteinLigandPaths:
    protein_path: Path
    ligand_path: Path


def _collect_protein_ligand_paths(
    base_dir: Path,
    ligand_exts: Iterable[str],
) -> list[ProteinLigandPaths]:
    if not base_dir.exists():
        return []

    paths: list[ProteinLigandPaths] = []
    protein_files = sorted(base_dir.rglob("*_protein.pdb"))
    for protein_path in protein_files:
        stem = protein_path.name.replace("_protein.pdb", "")
        ligand_path = None
        for ext in ligand_exts:
            candidate = protein_path.with_name(f"{stem}_ligand.{ext}")
            if candidate.exists():
                ligand_path = candidate
                break
        if ligand_path is None:
            continue
        paths.append(
            ProteinLigandPaths(protein_path=protein_path, ligand_path=ligand_path)
        )
    return paths


def load_pdbbind_v2020_paths(
    root: Path,
    *,
    include_general_minus_refined: bool = True,
    include_refined: bool = True,
    ligand_exts: Iterable[str] = ("sdf",),
) -> list[ProteinLigandPaths]:
    """Load PDBbind v2020 protein-ligand paths from the repo layout.

    Expected layout under root:
    - protein_ligand_general_minus_refined/
    - protein_ligand_refined/
    """
    paths: list[ProteinLigandPaths] = []
    if include_general_minus_refined:
        paths.extend(
            _collect_protein_ligand_paths(
                root / "protein_ligand_general_minus_refined",
                ligand_exts,
            )
        )
    if include_refined:
        paths.extend(
            _collect_protein_ligand_paths(
                root / "protein_ligand_refined",
                ligand_exts,
            )
        )
    return paths


def load_paths(root: Path) -> list[ProteinLigandPaths]:
    """Load dataset file paths from a root directory.

    This expects the repo's PDBbind v2020 layout under `data/pdbbind_v2020/`.
    """
    return load_pdbbind_v2020_paths(root)
