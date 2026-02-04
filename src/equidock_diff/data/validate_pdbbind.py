"""Quick validation for the local PDBbind v2020 layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from equidock_diff.data.io import load_pdbbind_v2020_paths


def _list_stems(base_dir: Path, suffix: str) -> set[str]:
    return {p.name[: -len(suffix)] for p in base_dir.rglob(f"*{suffix}")}


def count_missing_pairs(base_dir: Path) -> dict[str, int]:
    if not base_dir.exists():
        return {
            "proteins": 0,
            "ligands": 0,
            "missing_ligands_for_proteins": 0,
            "missing_proteins_for_ligands": 0,
        }

    protein_stems = _list_stems(base_dir, "_protein.pdb")
    ligand_stems = _list_stems(base_dir, "_ligand.sdf")
    return {
        "proteins": len(protein_stems),
        "ligands": len(ligand_stems),
        "missing_ligands_for_proteins": len(protein_stems - ligand_stems),
        "missing_proteins_for_ligands": len(ligand_stems - protein_stems),
    }


def _read_coords_from_sdf(path: Path) -> list[tuple[float, float, float]] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 4:
            return None
        counts_line = lines[3]
        atom_count = int(counts_line[0:3].strip())
        if atom_count <= 0 or len(lines) < 4 + atom_count:
            return None

        coords: list[tuple[float, float, float]] = []
        for line in lines[4 : 4 + atom_count]:
            # V2000 atom lines start with x, y, z as fixed-width float fields.
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            coords.append((x, y, z))
        if not coords:
            return None
        return coords
    except Exception:
        return None


def _read_coords_from_pdb(path: Path) -> list[tuple[float, float, float]] | None:
    try:
        coords: list[tuple[float, float, float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                # PDB fixed-width x/y/z columns: 31-38, 39-46, 47-54 (1-based).
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append((x, y, z))
        if not coords:
            return None
        return coords
    except Exception:
        return None


def count_empty_crops(
    paths: list,
    *,
    cutoff: float,
    limit: int,
) -> dict[str, int]:
    if limit <= 0 or not paths:
        return {
            "checked_pairs": 0,
            "missing_files": 0,
            "coord_parse_failed": 0,
            "empty_crops": 0,
        }

    cutoff_sq = cutoff * cutoff
    checked_pairs = 0
    missing_files = 0
    coord_parse_failed = 0
    empty_crops = 0

    for entry in paths:
        if checked_pairs >= limit:
            break
        checked_pairs += 1

        if not entry.protein_path.exists() or not entry.ligand_path.exists():
            missing_files += 1
            continue

        protein_coords = _read_coords_from_pdb(entry.protein_path)
        ligand_coords = _read_coords_from_sdf(entry.ligand_path)
        if not protein_coords or not ligand_coords:
            coord_parse_failed += 1
            continue

        lx = sum(p[0] for p in ligand_coords) / len(ligand_coords)
        ly = sum(p[1] for p in ligand_coords) / len(ligand_coords)
        lz = sum(p[2] for p in ligand_coords) / len(ligand_coords)

        min_dist_sq = math.inf
        for px, py, pz in protein_coords:
            dx = px - lx
            dy = py - ly
            dz = pz - lz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq <= cutoff_sq:
                break

        if min_dist_sq > cutoff_sq:
            empty_crops += 1

    return {
        "checked_pairs": checked_pairs,
        "missing_files": missing_files,
        "coord_parse_failed": coord_parse_failed,
        "empty_crops": empty_crops,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local PDBbind v2020 layout",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/pdbbind_v2020"),
        help="Dataset root containing the PDBbind v2020 folders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of sample entries to print",
    )
    parser.add_argument(
        "--no-refined",
        action="store_true",
        help="Exclude the refined subset",
    )
    parser.add_argument(
        "--no-general",
        action="store_true",
        help="Exclude the general-minus-refined subset",
    )
    parser.add_argument(
        "--crop-cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff (Angstrom) for empty-crop validation",
    )
    parser.add_argument(
        "--crop-check-limit",
        type=int,
        default=500,
        help="Number of protein-ligand pairs to inspect for empty crops (0 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_pdbbind_v2020_paths(
        args.root,
        include_general_minus_refined=not args.no_general,
        include_refined=not args.no_refined,
        ligand_exts=("sdf",),
    )

    print(f"Root: {args.root}")
    print(f"Pairs found: {len(paths)}")
    if not args.no_general:
        counts = count_missing_pairs(args.root / "protein_ligand_general_minus_refined")
        print(
            "General-minus-refined:"
            f" proteins={counts['proteins']}, ligands={counts['ligands']},"
            f" missing_ligands_for_proteins={counts['missing_ligands_for_proteins']},"
            f" missing_proteins_for_ligands={counts['missing_proteins_for_ligands']}"
        )
    if not args.no_refined:
        counts = count_missing_pairs(args.root / "protein_ligand_refined")
        print(
            "Refined:"
            f" proteins={counts['proteins']}, ligands={counts['ligands']},"
            f" missing_ligands_for_proteins={counts['missing_ligands_for_proteins']},"
            f" missing_proteins_for_ligands={counts['missing_proteins_for_ligands']}"
        )

    crop_counts = count_empty_crops(
        paths,
        cutoff=args.crop_cutoff,
        limit=max(args.crop_check_limit, 0),
    )
    print(
        "Crop validation:"
        f" checked_pairs={crop_counts['checked_pairs']},"
        f" missing_files={crop_counts['missing_files']},"
        f" coord_parse_failed={crop_counts['coord_parse_failed']},"
        f" empty_crops={crop_counts['empty_crops']}"
    )

    if not paths:
        print("No protein-ligand pairs found.")
        return

    print("Samples:")
    for entry in paths[: max(args.limit, 0)]:
        print(f"- {entry.protein_path} | {entry.ligand_path}")


if __name__ == "__main__":
    main()
