"""Pre-compute and cache protein-ligand graph batches for zero-CPU latency training."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from equidock_diff.data.io import filter_paths_by_complex_ids, load_paths, load_split_complex_ids
from equidock_diff.data.pipeline import load_protein_ligand_graph_cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-cache protein-ligand graphs from PDBbind")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/pdbbind_v2020"),
        help="Root directory of local PDBbind dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/evaluation/dissertation_panel20.txt"),
        help="Path to manifest containing target complex IDs",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.cache"),
        help="Target directory to write pre-computed PyTorch graph cache files",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Protein pocket crop radius in Angstroms",
    )
    parser.add_argument(
        "--edge-cutoff",
        type=float,
        default=4.5,
        help="Radius cutoff for pocket graph connectivity",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dataset_root.exists():
        fallback = Path("/Users/sadik/data/pdbbind_v2020")
        if fallback.exists():
            args.dataset_root = fallback
        else:
            raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_ids = load_split_complex_ids(args.manifest)
    print(f"Loading {len(manifest_ids)} target complexes from {args.manifest}...")

    all_paths = load_paths(args.dataset_root)
    matched_paths = filter_paths_by_complex_ids(all_paths, manifest_ids)

    print(f"Matched {len(matched_paths)} complexes in {args.dataset_root}. Pre-caching graphs...")

    t0 = perf_counter()
    for idx, item in enumerate(matched_paths, start=1):
        item_start = perf_counter()
        batch = load_protein_ligand_graph_cached(
            protein_path=item.protein_path,
            ligand_path=item.ligand_path,
            cutoff=args.cutoff,
            edge_cutoff=args.edge_cutoff,
            cache_dir=args.cache_dir,
        )
        elapsed = perf_counter() - item_start
        print(f"[{idx:02d}/{len(matched_paths):02d}] {item.complex_id}: {batch.node_features.size(0)} nodes, {batch.edge_index.size(1)} edges ({elapsed:.3f}s)")

    total_time = perf_counter() - t0
    print(f"Done! Cached {len(matched_paths)} complex graphs in {args.cache_dir} in {total_time:.2f}s.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
