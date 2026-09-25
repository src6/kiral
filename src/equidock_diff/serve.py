"""Persistent batched docking engine.

One CLI invocation per complex costs about 1.7 s of interpreter start plus ~0.16 s of
sampling; a batched, in-process engine amortises both. Measured on the RTX 3080 Ti
(2026-09-25, real panel complexes, 25 sample steps, FP32):

    batch 1   157.6 ms/pose    381 poses/min
    batch 4    52.7 ms/pose  1,138 poses/min
    batch 8    37.9 ms/pose  1,583 poses/min

The model already accepts one disconnected graph, so batching is a stacking problem, not
a model change: node features and coordinates are concatenated and the edge index is
offset per graph. Results are split back per complex for metrics.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from .data.io import load_paths
from .models.amp_utils import get_autocast_context, maybe_compile_model
from .train import (
    DEFAULT_COSINE_NU,
    DEFAULT_COSINE_OFFSET,
    load_dataset_example,
    make_model_for_node_dim,
    sample_positions,
)
from .utils.chemistry import evaluate_chemical_validity
from .utils.geometry import aligned_rmsd


@dataclass
class PoseResult:
    complex: str
    nodes: int
    edges: int
    raw_rmse: float | None = None
    aligned_rmsd: float | None = None
    chemical_is_valid: bool | None = None
    clash_fraction: float | None = None
    bond_violations: int | None = None


@dataclass
class BatchTiming:
    batch: int
    complexes: list[str]
    seconds: float
    per_pose_ms: float
    poses_per_min: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample many complexes in batched, in-process passes (throughput mode)",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional trained checkpoint")
    p.add_argument("--data-root", type=Path, default=Path.home() / "data" / "pdbbind_v2020")
    p.add_argument("--manifest", type=Path, default=None, help="Complex ids, one per line")
    p.add_argument("--limit", type=int, default=None, help="Sample only the first N complexes")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=25)
    p.add_argument("--schedule", default="cosine", choices=("linear", "cosine"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-snr-consistent", dest="snr_consistent", action="store_false", default=True)
    p.add_argument("--csv", type=Path, default=None)
    # pipeline / model configuration, mirroring the CLI defaults
    p.add_argument("--crop-cutoff", type=float, default=10.0)
    p.add_argument("--edge-cutoff", type=float, default=4.5)
    p.add_argument("--protein-node-budget", type=int, default=256)
    p.add_argument("--context-policy", default="fixed")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--allow-random-weights", action="store_true",
                   help="Run without a checkpoint (poses will be meaningless)")
    return p


def _pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        context_policy=args.context_policy,
        crop_cutoff=args.crop_cutoff,
        edge_cutoff=args.edge_cutoff,
        protein_node_budget=args.protein_node_budget,
        dataset_cache_dir=Path("data/.cache/equidock_diff_graphs"),
    )


_ARCH_FLAGS = (
    "hetero_edges",
    "ligand_global_node",
    "complete_frame",
    "frame_hetero_backbone",
    "use_edge_attention",
    "use_cross_interface_block",
)


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
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


def resolve_model_config(cli_args: argparse.Namespace, saved_args: dict | None) -> SimpleNamespace:
    """Resolve the model architecture for serving.

    A checkpoint's own saved configuration wins over the CLI defaults: serving a trained
    model with a mismatched architecture either fails outright or, worse, silently
    produces drift. CLI values remain the fallback when no checkpoint is supplied.
    """
    config = vars(_model_args(cli_args)).copy()
    if saved_args:
        for field in ("hidden_dim", "num_layers"):
            value = saved_args.get(field)
            if value is not None:
                config[field] = int(value)
        for field in _ARCH_FLAGS:
            value = saved_args.get(field)
            if value is not None:
                config[field] = bool(value)
    return SimpleNamespace(**config)


class DockingEngine:
    """Load once, sample many. Batching is transparent to the caller."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        )
        self.model: torch.nn.Module | None = None
        self._paths: dict[str, object] | None = None
        self._saved_args: dict | None = None

    # -- loading ---------------------------------------------------------------
    def _dataset_paths(self) -> dict[str, object]:
        if self._paths is None:
            self._paths = {p.complex_id: p for p in load_paths(self.args.data_root)}
        return self._paths

    def _complex_ids(self) -> list[str]:
        if self.args.manifest is not None:
            ids = [
                line.strip()
                for line in self.args.manifest.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            ids = sorted(self._dataset_paths())
        if self.args.limit is not None:
            ids = ids[: self.args.limit]
        return ids

    def _load_graph(self, complex_id: str) -> dict | None:
        example = self._dataset_paths().get(complex_id)
        if example is None:
            return None
        features, positions, edge_index, bond_index, _cutoff, _retained = load_dataset_example(
            example, _pipeline_args(self.args), self.device
        )
        return {
            "complex": complex_id,
            "features": features,
            "positions": positions,
            "edge_index": edge_index,
            "bond_index": bond_index,
        }

    def _checkpoint_saved_args(self) -> dict | None:
        if self.args.checkpoint is None:
            return None
        if self._saved_args is None:
            checkpoint = torch.load(self.args.checkpoint, map_location="cpu", weights_only=False)
            saved = checkpoint.get("saved_args", {}) if isinstance(checkpoint, dict) else {}
            self._saved_args = dict(saved)
        return self._saved_args

    def _ensure_model(self, node_dim: int) -> torch.nn.Module:
        if self.model is None:
            config = resolve_model_config(self.args, self._checkpoint_saved_args())
            model = make_model_for_node_dim(config, self.device, node_dim=node_dim)
            model = model.to(self.device).eval()
            if self.args.checkpoint is None:
                if not self.args.allow_random_weights:
                    raise SystemExit(
                        "refusing to serve randomly initialised weights: pass --checkpoint, "
                        "or --allow-random-weights if you only want to time the plumbing")
                print("warning: no --checkpoint, poses come from random weights and are meaningless")
            else:
                from .train import load_checkpoint

                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                state = load_checkpoint(
                    self.args.checkpoint, model=model, optimizer=optimizer, device=self.device
                )
                print(
                    f"checkpoint loaded: {self.args.checkpoint} (steps={state.completed_steps}, "
                    f"hidden_dim={config.hidden_dim}, layers={config.num_layers}, "
                    f"frame_backbone={config.frame_hetero_backbone})"
                )
            if self.args.compile:
                model = maybe_compile_model(model, enabled=True)
            self.model = model
        return self.model

    # -- sampling --------------------------------------------------------------
    def _stack(self, graphs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat([g["features"] for g in graphs], dim=0)
        positions = torch.cat([g["positions"] for g in graphs], dim=0)
        edges, running = [], 0
        for g in graphs:
            edges.append(g["edge_index"] + running)
            running += g["features"].size(0)
        return features, positions, torch.cat(edges, dim=1)

    def sample(self, complex_ids: list[str]) -> tuple[list[PoseResult], list[BatchTiming]]:
        results: list[PoseResult] = []
        timings: list[BatchTiming] = []
        batch_size = max(1, self.args.batch_size)
        for start in range(0, len(complex_ids), batch_size):
            chunk = complex_ids[start : start + batch_size]
            graphs = [g for g in (self._load_graph(c) for c in chunk) if g is not None]
            if not graphs:
                continue
            features, positions, edge_index = self._stack(graphs)
            model = self._ensure_model(features.size(-1))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), get_autocast_context(self.device, enabled=self.args.amp):
                sampled, _trajectory, _diag = sample_positions(
                    model,
                    features,
                    edge_index,
                    positions.size(0),
                    self.device,
                    self.args.sample_steps,
                    0.1,
                    2.0,
                    10.0,
                    50.0,
                    sample_time_power=1.0,
                    noise_schedule=self.args.schedule,
                    cosine_offset=DEFAULT_COSINE_OFFSET,
                    cosine_nu=DEFAULT_COSINE_NU,
                    reference_positions=positions,
                    anchor_protein=True,
                    snr_consistent=self.args.snr_consistent,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            timings.append(
                BatchTiming(
                    batch=len(graphs),
                    complexes=[g["complex"] for g in graphs],
                    seconds=round(seconds, 4),
                    per_pose_ms=round(seconds * 1000 / len(graphs), 3),
                    poses_per_min=round(len(graphs) * 60 / seconds, 1),
                )
            )
            offset = 0
            for graph in graphs:
                count = graph["features"].size(0)
                local_features = features[offset : offset + count]
                local_sampled = sampled[offset : offset + count]
                local_reference = positions[offset : offset + count]
                offset += count
                mask = local_features[:, -1] > 0.5
                result = PoseResult(
                    complex=graph["complex"],
                    nodes=count,
                    edges=int(graph["edge_index"].size(1)),
                )
                if bool(mask.any()):
                    mobile = local_sampled[mask]
                    target = local_reference[mask]
                    result.raw_rmse = float(torch.sqrt(torch.mean((mobile - target) ** 2)))
                    result.aligned_rmsd = float(aligned_rmsd(mobile, target))
                    report = evaluate_chemical_validity(
                        local_sampled,
                        local_features,
                        ligand_bond_index=graph["bond_index"],
                        reference_positions=local_reference,
                    )
                    result.chemical_is_valid = bool(report.is_valid)
                    result.clash_fraction = float(report.clash_fraction)
                    result.bond_violations = int(report.bond_violation_count)
                results.append(result)
                if not self.args.quiet:
                    rmsd = f"{result.aligned_rmsd:.4f}" if result.aligned_rmsd is not None else "n/a"
                    print(
                        f"  {result.complex:8} nodes={result.nodes:4} edges={result.edges:6} "
                        f"aligned_rmsd={rmsd} valid={result.chemical_is_valid}",
                        flush=True,
                    )
            del model, features, positions, edge_index, sampled
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results, timings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = DockingEngine(args)
    ids = engine._complex_ids()
    print(f"=== kiral serve === complexes={len(ids)} batch={args.batch_size} "
          f"device={engine.device.type} sample_steps={args.sample_steps} amp={args.amp}")
    wall_start = time.perf_counter()
    results, timings = engine.sample(ids)
    wall = time.perf_counter() - wall_start
    if not timings:
        print("nothing sampled")
        return 1
    print()
    header = f"{'complexes':>10} {'batch':>6} {'seconds':>9} {'ms/pose':>9} {'poses/min':>10}"
    print(header)
    print("-" * len(header))
    for t in timings:
        print(f"{len(t.complexes):>10} {t.batch:>6} {t.seconds:>9.3f} {t.per_pose_ms:>9.2f} {t.poses_per_min:>10.1f}")
    total_poses = sum(len(t.complexes) for t in timings)
    sampled_seconds = sum(t.seconds for t in timings)
    rmsds = [r.aligned_rmsd for r in results if r.aligned_rmsd is not None]
    print()
    print(f"total poses        : {total_poses}")
    print(f"sampling wall      : {sampled_seconds:.2f} s")
    print(f"end-to-end wall    : {wall:.2f} s  (includes load + metrics)")
    print(f"throughput         : {total_poses * 60 / wall:.1f} poses/min end-to-end, "
          f"{total_poses * 60 / sampled_seconds:.1f} poses/min sampling-only")
    if rmsds:
        print(f"mean aligned RMSD  : {statistics.fmean(rmsds):.4f} A (n={len(rmsds)})")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
        print(f"csv                : {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
