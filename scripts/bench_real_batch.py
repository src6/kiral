#!/usr/bin/env python3
"""Batched sampler benchmark on REAL PDBbind complexes.

`bench_sampler.py` builds synthetic *complete* graphs, which overstate any optimisation
that thrives on dense matmuls - BF16 especially. Real complexes are pocket-cropped
(<= protein-node-budget nodes) and sparse under the 4.5 A edge cutoff. This loads real
complexes through the repo's own data pipeline and batches them into one disconnected
graph (the shape the model already accepts) and reports per-complex latency, throughput
and the true graph sizes.

Usage:
    uv run python scripts/bench_real_batch.py --batch-sizes 1,2,4,8 --count 8
    uv run python scripts/bench_real_batch.py --complicated ... --amp --compile
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from equidock_diff.data.io import load_paths  # noqa: E402
from equidock_diff.models.amp_utils import get_autocast_context, maybe_compile_model  # noqa: E402
from equidock_diff.train import (  # noqa: E402
    DEFAULT_COSINE_NU,
    DEFAULT_COSINE_OFFSET,
    load_dataset_example,
    make_model_for_node_dim,
    sample_positions,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batched sampler benchmark on real complexes")
    p.add_argument("--root", type=Path, default=Path.home() / "data" / "pdbbind_v2020")
    p.add_argument("--manifest", type=Path, default=_REPO / "config/evaluation/dissertation_panel20.txt")
    p.add_argument("--batch-sizes", default="1,2,4,8")
    p.add_argument("--count", type=int, default=None, help="Complexes to load (default: all in the manifest)")
    p.add_argument("--sample-steps", type=int, default=25)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--schedule", default="cosine", choices=("linear", "cosine"))
    p.add_argument("--device", default="auto")
    p.add_argument("--csv", type=Path, default=None)
    # pipeline / model configuration, mirroring the CLI defaults
    p.add_argument("--crop-cutoff", type=float, default=10.0)
    p.add_argument("--edge-cutoff", type=float, default=4.5)
    p.add_argument("--protein-node-budget", type=int, default=256)
    p.add_argument("--context-policy", default="fixed")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        context_policy=args.context_policy,
        crop_cutoff=args.crop_cutoff,
        edge_cutoff=args.edge_cutoff,
        protein_node_budget=args.protein_node_budget,
        dataset_cache_dir=_REPO / "data" / ".cache" / "equidock_diff_graphs",
    )


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    # the "frame backbone" arm, matching the dissertation panel
    return SimpleNamespace(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        hetero_edges=False,
        ligand_global_node=False,
        complete_frame=False,
        frame_hetero_backbone=True,
        use_edge_attention=False,
        use_cross_interface_block=False,
    )


def load_real_graphs(args, device) -> list[dict]:
    paths = load_paths(args.root)
    by_id = {p.complex_id: p for p in paths}
    ids = [
        line.strip()
        for line in args.manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.count is not None:
        ids = ids[: args.count]
    graphs = []
    for cid in ids:
        example = by_id.get(cid)
        if example is None:
            print(f"  skip {cid}: not in the dataset", flush=True)
            continue
        features, positions, edge_index, _bond, _cutoff, _retained = load_dataset_example(
            example, pipeline_args(args), device
        )
        graphs.append(
            {
                "complex": cid,
                "features": features,
                "positions": positions,
                "edge_index": edge_index,
            }
        )
        print(f"  loaded {cid}: {features.size(0)} nodes, {edge_index.size(1)} edges", flush=True)
    return graphs


def stack(graphs: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    features = torch.cat([g["features"].to(device) for g in graphs], dim=0)
    positions = torch.cat([g["positions"].to(device) for g in graphs], dim=0)
    offsets, edge_list, running = [], [], 0
    for g in graphs:
        edge_list.append(g["edge_index"].to(device) + running)
        offsets.append(running)
        running += g["features"].size(0)
    edge_index = torch.cat(edge_list, dim=1)
    return features, positions, edge_index, offsets


def bench(args, graphs, device) -> list[dict]:
    rows = []
    for batch in [int(b) for b in args.batch_sizes.split(",") if b.strip()]:
        if batch > len(graphs):
            print(f"  batch {batch}: skipped (only {len(graphs)} complexes loaded)")
            continue
        subset = graphs[:batch]
        features, positions, edge_index, _ = stack(subset, device)
        torch.manual_seed(args.seed)
        model = make_model_for_node_dim(model_args(args), device, node_dim=features.size(-1))
        model = model.to(device).eval()
        if args.compile:
            model = maybe_compile_model(model, enabled=True)

        def step():
            with torch.no_grad(), get_autocast_context(device, enabled=args.amp):
                sample_positions(
                    model, features, edge_index, positions.size(0), device,
                    args.sample_steps, 0.1, 2.0, 10.0, 50.0,
                    sample_time_power=1.0, noise_schedule=args.schedule,
                    cosine_offset=DEFAULT_COSINE_OFFSET, cosine_nu=DEFAULT_COSINE_NU,
                    reference_positions=positions, anchor_protein=True, snr_consistent=True,
                )

        for _ in range(args.warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        times.sort()
        mean = statistics.fmean(times)
        rows.append(
            {
                "batch": batch,
                "nodes": int(features.size(0)),
                "edges": int(edge_index.size(1)),
                "nodes_per_complex": round(features.size(0) / batch, 1),
                "edges_per_complex": round(edge_index.size(1) / batch, 1),
                "batch_mean_ms": round(mean * 1000, 3),
                "per_pose_mean_ms": round(mean * 1000 / batch, 3),
                "per_pose_p50_ms": round(times[len(times) // 2] * 1000 / batch, 3),
                "per_pose_p95_ms": round(times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))] * 1000 / batch, 3),
                "poses_per_min": round(batch * 60000 / (mean * 1000), 1),
                "amp": args.amp,
                "compile": args.compile,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print("=== real-complex batched sampler benchmark ===")
    print(f"torch {torch.__version__} | device {device.type}"
          + (f" | {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))
    print(f"sample-steps {args.sample_steps} | schedule {args.schedule} | amp {args.amp} | compile {args.compile}")
    print("loading complexes through the repo pipeline:")
    graphs = load_real_graphs(args, device)
    if not graphs:
        print("no complexes loaded")
        return 1
    print()
    rows = bench(args, graphs, device)
    header = (f"{'batch':>6} {'nodes':>7} {'edges':>8} {'nodes/cplx':>11} {'mean/pose ms':>13} "
              f"{'p50/pose ms':>12} {'p95/pose ms':>12} {'poses/min':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['batch']:>6} {r['nodes']:>7} {r['edges']:>8} {r['nodes_per_complex']:>11} "
              f"{r['per_pose_mean_ms']:>13} {r['per_pose_p50_ms']:>12} {r['per_pose_p95_ms']:>12} "
              f"{r['poses_per_min']:>10}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ncsv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
