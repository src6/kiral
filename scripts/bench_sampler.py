#!/usr/bin/env python3
"""Batched sampler latency benchmark for Kiral.

Measures reverse-diffusion latency and pose throughput on synthetic graphs at several
batch sizes, with BF16 autocast and torch.compile toggled, so the scale at which each
optimisation starts paying is visible rather than assumed.

Per-complex latency (a "pose") is what the production targets quote (e.g. P95 < 80 ms),
so the report is per pose, not per batch.

Usage:
    uv run python scripts/bench_sampler.py --batch-sizes 1,8,32,64 --sample-steps 25 --repeat 5
    uv run python scripts/bench_sampler.py --compile-modes off,on --csv /tmp/bench.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from kiral.models.amp_utils import get_autocast_context, maybe_compile_model
from kiral.train import (  # noqa: E402
    DEFAULT_COSINE_NU,
    DEFAULT_COSINE_OFFSET,
    build_synthetic_graph,
    make_model_for_node_dim,
    sample_positions,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batched sampler latency benchmark")
    p.add_argument("--batch-sizes", default="1,8,32,64", help="Comma-separated batch sizes")
    p.add_argument("--nodes", type=int, default=12, help="Nodes per synthetic graph")
    p.add_argument("--sample-steps", type=int, default=25, help="Reverse diffusion steps")
    p.add_argument("--repeat", type=int, default=5, help="Timed repetitions per configuration")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing")
    p.add_argument("--amp-modes", default="off,on", help="Comma-separated: off,on")
    p.add_argument("--compile-modes", default="off", help="Comma-separated: off,on")
    p.add_argument("--noise-schedule", default="cosine", choices=("linear", "cosine"))
    p.add_argument("--snr-consistent", action="store_true", default=True)
    p.add_argument("--no-snr-consistent", dest="snr_consistent", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    # model configuration: mirrors the CLI defaults (expose them so the report is self-describing)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--hetero-edges", action="store_true", default=True)
    p.add_argument("--ligand-global-node", action="store_true", default=True)
    p.add_argument("--complete-frame", action="store_true", default=False)
    p.add_argument("--frame-hetero-backbone", action="store_true", default=True)
    p.add_argument("--use-edge-attention", action="store_true", default=False)
    p.add_argument("--use-cross-interface-block", action="store_true", default=False)
    # sampler configuration
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=2.0)
    p.add_argument("--sample-score-clip", type=float, default=1000.0)
    p.add_argument("--sample-position-clip", type=float, default=50.0)
    p.add_argument("--sample-time-power", type=float, default=1.0)
    return p.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_once(args, model, graph, device) -> None:
    node_features, positions, edge_index = graph
    with torch.no_grad(), get_autocast_context(device, enabled=args._amp):
        sample_positions(
            model,
            node_features,
            edge_index,
            positions.size(0),
            device,
            args.sample_steps,
            args.beta_min,
            args.beta_max,
            args.sample_score_clip,
            args.sample_position_clip,
            sample_time_power=args.sample_time_power,
            noise_schedule=args.noise_schedule,
            cosine_offset=DEFAULT_COSINE_OFFSET,
            cosine_nu=DEFAULT_COSINE_NU,
            reference_positions=positions,
            anchor_protein=args.frame_hetero_backbone,
            snr_consistent=args.snr_consistent,
        )


def bench_config(args, batch: int, amp: bool, compile_on: bool, device) -> dict:
    args._amp = amp
    torch.manual_seed(args.seed)
    graph = build_synthetic_graph(args.nodes, batch, device)
    model = make_model_for_node_dim(args, device, node_dim=graph[0].size(-1)).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    if compile_on:
        model = maybe_compile_model(model, enabled=True)

    for _ in range(args.warmup):
        run_once(args, model, graph, device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_once(args, model, graph, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times_ms = sorted(t * 1000.0 for t in times)
    mean_ms = statistics.fmean(times_ms)
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[min(len(times_ms) - 1, int(round(0.95 * (len(times_ms) - 1))))]
    return {
        "batch": batch,
        "amp": amp,
        "compile": compile_on,
        "params": params,
        "batch_mean_ms": round(mean_ms, 3),
        "per_pose_mean_ms": round(mean_ms / batch, 3),
        "per_pose_p50_ms": round(p50 / batch, 3),
        "per_pose_p95_ms": round(p95 / batch, 3),
        "poses_per_min": round(batch * 60000.0 / mean_ms, 1),
        "runs": len(times),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    batches = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    amp_modes = [m.strip() == "on" for m in args.amp_modes.split(",") if m.strip()]
    compile_modes = [m.strip() == "on" for m in args.compile_modes.split(",") if m.strip()]

    print("=== Kiral sampler benchmark ===")
    print(f"torch {torch.__version__} | device {device.type}"
          + (f" | {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))
    print(f"nodes/graph {args.nodes} | sample-steps {args.sample_steps} | schedule {args.noise_schedule}"
          f" | snr_consistent {args.snr_consistent} | warmup {args.warmup} | repeat {args.repeat}")
    print()
    header = (f"{'batch':>6} {'amp':>4} {'compile':>8} {'params':>9} "
              f"{'mean/pose ms':>13} {'p50/pose ms':>12} {'p95/pose ms':>12} {'poses/min':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for batch in batches:
        for amp in amp_modes:
            for compile_on in compile_modes:
                row = bench_config(args, batch, amp, compile_on, device)
                rows.append(row)
                print(f"{row['batch']:>6} {('on' if amp else 'off'):>4} "
                      f"{('on' if compile_on else 'off'):>8} {row['params']:>9} "
                      f"{row['per_pose_mean_ms']:>13} {row['per_pose_p50_ms']:>12} "
                      f"{row['per_pose_p95_ms']:>12} {row['poses_per_min']:>10}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ncsv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
