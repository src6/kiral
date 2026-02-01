"""Quick validation for the local PDBbind v2020 layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from equidock_diff.data.io import load_pdbbind_v2020_paths


def count_missing_sdf(base_dir: Path) -> tuple[int, int]:
    if not base_dir.exists():
        return 0, 0
    protein_files = sorted(base_dir.rglob("*_protein.pdb"))
    missing = 0
    for protein_path in protein_files:
        stem = protein_path.name.replace("_protein.pdb", "")
        ligand_path = protein_path.with_name(f"{stem}_ligand.sdf")
        if not ligand_path.exists():
            missing += 1
    return len(protein_files), missing


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
        total, missing = count_missing_sdf(
            args.root / "protein_ligand_general_minus_refined"
        )
        print(
            f"General-minus-refined proteins: {total} (missing SDF ligands: {missing})"
        )
    if not args.no_refined:
        total, missing = count_missing_sdf(args.root / "protein_ligand_refined")
        print(f"Refined proteins: {total} (missing SDF ligands: {missing})")
    if not paths:
        print("No protein-ligand pairs found.")
        return

    print("Samples:")
    for entry in paths[: max(args.limit, 0)]:
        print(f"- {entry.protein_path} | {entry.ligand_path}")


if __name__ == "__main__":
    main()
