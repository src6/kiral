"""Coordinate local and remote research_runner jobs under one dual-host plan."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from equidock_diff.research_runner import (
    PlannedRun,
    _group_planned_runs,
    build_parser as build_research_parser,
    expand_run_specs,
    plan_runs,
)
from equidock_diff.remote_research_runner import (
    local_git_sync_state,
    resolve_sync_mode,
    sync_remote_repo,
)


DEFAULT_DUAL_ROOT = Path("runs/dual")
DEFAULT_LOCAL_RESEARCH_ROOT = Path("runs/research")
DEFAULT_REMOTE_SUMMARY_ROOT = Path("runs/remote")


@dataclass(frozen=True)
class ScheduledUnit:
    unit_id: str
    worker_pool: str
    subtag: str
    complex_id: str
    seed: int
    steps: int
    protein_node_budget: int
    crop_cutoff: float
    use_cross_interface_block: bool
    planned_runs: tuple[PlannedRun, ...]
    local_output_root: Path
    remote_output_root: Path


@dataclass(frozen=True)
class UnitStatusRow:
    unit_id: str
    worker_pool: str
    subtag: str
    complex_id: str
    seed: int
    steps: int
    protein_node_budget: int
    crop_cutoff: float
    use_cross_interface_block: bool
    planned_actions: str
    status: str
    completed_runs: int
    failed_runs: int
    mean_aligned_ligand_rmsd: float | None
    mean_training_seconds: float | None
    local_output_root: Path
    remote_output_root: Path
    last_heartbeat: str
    error: str | None = None


@dataclass(frozen=True)
class WorkerStatusRow:
    worker_name: str
    worker_pool: str
    assigned_units: int
    completed_units: int
    failed_units: int
    state: str
    current_subtag: str | None
    last_heartbeat: str


@dataclass(frozen=True)
class UnitMetrics:
    total_runs: int
    completed_runs: int
    failed_runs: int
    mean_aligned_ligand_rmsd: float | None
    mean_training_seconds: float | None


def build_parser() -> argparse.ArgumentParser:
    research_defaults = build_research_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--model",
            "baseline",
            "--noise-schedule",
            "linear",
            "--tag",
            "placeholder",
        ]
    )
    parser = argparse.ArgumentParser(
        description="Coordinate local and remote research_runner groups under one dual-host tag"
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--complex-id",
        dest="complex_ids",
        action="append",
        default=None,
        help="Complex id to include; may be passed multiple times",
    )
    scope.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest listing one complex id per line",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=research_defaults.dataset_root,
        help="Dataset root used to resolve complex ids",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "frame_backbone"),
        required=True,
        help="Research model family to run",
    )
    parser.add_argument(
        "--noise-schedule",
        choices=("linear", "cosine"),
        required=True,
        help="Noise schedule shared by the matrix",
    )
    parser.add_argument("--seed", dest="seeds", action="append", type=int, default=None)
    parser.add_argument("--steps", dest="steps_values", action="append", type=int, default=None)
    parser.add_argument(
        "--context-policy",
        dest="context_policies",
        action="append",
        choices=("fixed", "adaptive", "gated"),
        default=None,
    )
    parser.add_argument(
        "--protein-node-budget",
        dest="protein_node_budgets",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument("--crop-cutoff", dest="crop_cutoffs", action="append", type=float, default=None)
    parser.add_argument(
        "--sample-steps",
        dest="sample_steps_values",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sample-time-power",
        dest="sample_time_powers",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-shape-weight",
        dest="ligand_shape_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-protein-clash-weight",
        dest="ligand_protein_clash_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ligand-protein-contact-weight",
        dest="ligand_protein_contact_weights",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sample-score-clip",
        dest="sample_score_clips",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sample-position-clip",
        dest="sample_position_clips",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--use-edge-attention",
        action="store_true",
        help="Enable lightweight incoming-edge attention inside frame-backbone runs",
    )
    parser.add_argument(
        "--use-cross-interface-block",
        action="store_true",
        help="Enable the frame-backbone protein-to-ligand cross-message block",
    )
    parser.add_argument("--dual-tag", required=True, help="Top-level dual-host tag under runs/dual/")
    parser.add_argument(
        "--local-max-parallel",
        type=int,
        default=1,
        help="Number of local worker slots to use on the laptop",
    )
    parser.add_argument(
        "--remote-max-parallel",
        type=int,
        default=2,
        help="Number of remote worker slots to use on the Mac mini",
    )
    parser.add_argument(
        "--laptop-mode",
        choices=("smoke", "worker"),
        default="smoke",
        help="Whether the laptop is limited to smoke/debug work or participates as a worker",
    )
    parser.add_argument(
        "--routing-policy",
        choices=("size", "explicit"),
        default="size",
        help="How to route grouped runs between the laptop and the Mac mini",
    )
    parser.add_argument(
        "--local-complex-id",
        dest="local_complex_ids",
        action="append",
        default=None,
        help="Complex id to force onto the local worker when --routing-policy explicit is used",
    )
    parser.add_argument(
        "--remote-complex-id",
        dest="remote_complex_ids",
        action="append",
        default=None,
        help="Complex id to force onto the remote worker when --routing-policy explicit is used",
    )
    parser.add_argument("--remote-host", required=True, help="SSH host alias or IP for the Mac mini")
    parser.add_argument(
        "--remote-repo",
        type=Path,
        required=True,
        help="Absolute path to the dedicated repo clone on the remote host",
    )
    parser.add_argument(
        "--remote-dataset-target",
        type=Path,
        default=None,
        help="Actual dataset root on the remote host used to populate data/pdbbind_v2020",
    )
    parser.add_argument(
        "--compare-against",
        default=None,
        help="Optional prior tag to diff against within each worker subtag",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Keep pose, trajectory, and plot artifacts for each worker subtag",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("git", "rsync", "auto"),
        default="auto",
        help="How to update the remote dedicated clone before remote runs",
    )
    parser.add_argument(
        "--fetch-full-results",
        action="store_true",
        help="Fetch the whole remote tag directory instead of only compact summaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the dual-host plan without launching local or remote jobs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DUAL_ROOT,
        help="Top-level output root for dual-host plans",
    )
    return parser


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _float_token(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if rendered == "-0":
        rendered = "0"
    return rendered.replace("-", "m").replace(".", "p")


def _planned_actions_summary(planned_runs: tuple[PlannedRun, ...]) -> str:
    counts: dict[str, int] = {}
    for run in planned_runs:
        counts[run.action] = counts.get(run.action, 0) + 1
    parts = []
    for action in ("train", "resample", "reuse"):
        count = counts.get(action, 0)
        if count:
            parts.append(f"{action}:{count}")
    return ",".join(parts)


def _group_subtag(dual_tag: str, worker_pool: str, planned_runs: tuple[PlannedRun, ...]) -> str:
    base = planned_runs[0].checkpoint_name
    return f"{dual_tag}__{worker_pool}__{base}"


def _worker_slots(args: argparse.Namespace) -> list[tuple[str, str]]:
    local_slots = 1 if args.laptop_mode == "smoke" else args.local_max_parallel
    slots: list[tuple[str, str]] = []
    for index in range(max(1, local_slots)):
        slots.append((f"local_cpu_{index + 1}", "local_cpu"))
    for index in range(max(1, args.remote_max_parallel)):
        slots.append((f"remote_cpu_{index + 1}", "remote_cpu"))
    return slots


def _validate_args(args: argparse.Namespace) -> None:
    if args.local_max_parallel < 1:
        raise ValueError("--local-max-parallel must be at least 1.")
    if args.remote_max_parallel < 1:
        raise ValueError("--remote-max-parallel must be at least 1.")
    if args.use_edge_attention and args.model != "frame_backbone":
        raise ValueError("--use-edge-attention currently requires --model frame_backbone.")
    if args.use_cross_interface_block and args.model != "frame_backbone":
        raise ValueError("--use-cross-interface-block currently requires --model frame_backbone.")
    if args.use_cross_interface_block and args.use_edge_attention:
        raise ValueError("--use-cross-interface-block cannot be combined with --use-edge-attention in this cycle.")
    if args.routing_policy == "explicit":
        local_ids = set(args.local_complex_ids or [])
        remote_ids = set(args.remote_complex_ids or [])
        overlap = local_ids & remote_ids
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"Complex ids cannot be assigned to both local and remote: {joined}")
        if not local_ids and not remote_ids:
            raise ValueError(
                "--routing-policy explicit requires at least one --local-complex-id or --remote-complex-id."
            )


def _route_group(
    group: tuple[PlannedRun, ...],
    *,
    args: argparse.Namespace,
    total_complexes: int,
    total_seeds: int,
) -> str:
    spec = group[0].spec
    if args.routing_policy == "explicit":
        local_ids = set(args.local_complex_ids or [])
        remote_ids = set(args.remote_complex_ids or [])
        if spec.complex_id in local_ids:
            return "local_cpu"
        if spec.complex_id in remote_ids:
            return "remote_cpu"
        raise ValueError(
            f"Complex {spec.complex_id!r} was not assigned to local or remote under --routing-policy explicit."
        )

    if total_complexes > 1:
        return "remote_cpu"
    if total_seeds > 1:
        return "remote_cpu"
    if any(run.action == "train" for run in group) and spec.steps >= 200:
        return "remote_cpu"
    return "local_cpu"


def plan_scheduled_units(args: argparse.Namespace) -> list[ScheduledUnit]:
    scratch_root = args.output_root / args.dual_tag / "_planning_scratch"
    specs = expand_run_specs(args)
    grouped = [tuple(group) for group in _group_planned_runs(plan_runs(specs, scratch_root))]
    total_complexes = len({spec.complex_id for spec in specs})
    total_seeds = len({spec.seed for spec in specs})
    units: list[ScheduledUnit] = []
    for index, group in enumerate(grouped, start=1):
        worker_pool = _route_group(group, args=args, total_complexes=total_complexes, total_seeds=total_seeds)
        subtag = _group_subtag(args.dual_tag, worker_pool, group)
        units.append(
            ScheduledUnit(
                unit_id=f"unit_{index:03d}",
                worker_pool=worker_pool,
                subtag=subtag,
                complex_id=group[0].spec.complex_id,
                seed=group[0].spec.seed,
                steps=group[0].spec.steps,
                protein_node_budget=group[0].spec.protein_node_budget,
                crop_cutoff=group[0].spec.crop_cutoff,
                use_cross_interface_block=group[0].spec.use_cross_interface_block,
                planned_runs=group,
                local_output_root=DEFAULT_LOCAL_RESEARCH_ROOT / subtag,
                remote_output_root=DEFAULT_REMOTE_SUMMARY_ROOT / subtag,
            )
        )
    return units


def _append_repeated(argv: list[str], flag: str, values: list[object]) -> None:
    for value in values:
        argv.extend([flag, str(value)])


def _group_runner_args(unit: ScheduledUnit, *, compare_against: str | None, save_artifacts: bool) -> list[str]:
    planned_runs = unit.planned_runs
    specs = [item.spec for item in planned_runs]
    argv: list[str] = []
    argv.extend(["--complex-id", unit.complex_id])
    argv.extend(["--model", specs[0].model, "--noise-schedule", specs[0].noise_schedule, "--tag", unit.subtag])
    argv.extend(["--device-policy", "canonical"])
    if compare_against is not None:
        argv.extend(["--compare-against", compare_against])
    if save_artifacts:
        argv.append("--save-artifacts")
    _append_repeated(argv, "--seed", sorted({spec.seed for spec in specs}))
    _append_repeated(argv, "--steps", sorted({spec.steps for spec in specs}))
    _append_repeated(argv, "--context-policy", sorted({spec.context_policy for spec in specs}))
    _append_repeated(argv, "--protein-node-budget", sorted({spec.protein_node_budget for spec in specs}))
    _append_repeated(argv, "--crop-cutoff", sorted({spec.crop_cutoff for spec in specs}))
    _append_repeated(argv, "--sample-steps", sorted({spec.sample_steps for spec in specs}))
    _append_repeated(argv, "--sample-time-power", sorted({spec.sample_time_power for spec in specs}))
    _append_repeated(argv, "--ligand-shape-weight", sorted({spec.ligand_shape_weight for spec in specs}))
    _append_repeated(
        argv,
        "--ligand-protein-clash-weight",
        sorted({spec.ligand_protein_clash_weight for spec in specs}),
    )
    _append_repeated(
        argv,
        "--ligand-protein-contact-weight",
        sorted({spec.ligand_protein_contact_weight for spec in specs}),
    )
    _append_repeated(argv, "--sample-score-clip", sorted({spec.sample_score_clip for spec in specs}))
    _append_repeated(
        argv,
        "--sample-position-clip",
        sorted({spec.sample_position_clip for spec in specs}),
    )
    if specs[0].use_edge_attention:
        argv.append("--use-edge-attention")
    if specs[0].use_cross_interface_block:
        argv.append("--use-cross-interface-block")
    return argv


def build_local_runner_argv(unit: ScheduledUnit, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "equidock_diff.research_runner",
        "--dataset-root",
        str(args.dataset_root),
        * _group_runner_args(unit, compare_against=args.compare_against, save_artifacts=args.save_artifacts),
        "--max-parallel",
        "1",
    ]


def build_remote_runner_argv(unit: ScheduledUnit, args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "equidock_diff.remote_research_runner",
        "--remote-host",
        args.remote_host,
        "--remote-repo",
        str(args.remote_repo),
        "--sync-mode",
        "none",
        "--remote-dataset-target",
        str(args.remote_dataset_target) if args.remote_dataset_target is not None else "",
    ]
    argv.extend(_group_runner_args(unit, compare_against=args.compare_against, save_artifacts=args.save_artifacts))
    argv.extend(["--max-parallel", "1"])
    if args.fetch_full_results:
        argv.append("--fetch-full-results")
    if args.remote_dataset_target is None:
        argv = [item for item in argv if item != "--remote-dataset-target" and item != ""]
    return argv


def _load_run_metrics(run_index_csv: Path) -> UnitMetrics:
    if not run_index_csv.exists():
        return UnitMetrics(total_runs=0, completed_runs=0, failed_runs=0, mean_aligned_ligand_rmsd=None, mean_training_seconds=None)
    completed = 0
    failed = 0
    aligned_values: list[float] = []
    training_seconds: list[float] = []
    with run_index_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        action = row.get("action", "")
        if action.startswith("planned_"):
            continue
        if row.get("aligned_ligand_rmsd", "").strip():
            aligned_values.append(float(row["aligned_ligand_rmsd"]))
        if row.get("training_seconds", "").strip():
            training_seconds.append(float(row["training_seconds"]))
        completed += 1
    status_path = run_index_csv.with_name("status.csv")
    if status_path.exists():
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            status_reader = csv.DictReader(handle)
            failed = sum(1 for row in status_reader if row.get("status") == "failed")
    return UnitMetrics(
        total_runs=len(rows),
        completed_runs=completed,
        failed_runs=failed,
        mean_aligned_ligand_rmsd=statistics.fmean(aligned_values) if aligned_values else None,
        mean_training_seconds=statistics.fmean(training_seconds) if training_seconds else None,
    )


def _initial_unit_status(unit: ScheduledUnit, status: str) -> UnitStatusRow:
    return UnitStatusRow(
        unit_id=unit.unit_id,
        worker_pool=unit.worker_pool,
        subtag=unit.subtag,
        complex_id=unit.complex_id,
        seed=unit.seed,
        steps=unit.steps,
        protein_node_budget=unit.protein_node_budget,
        crop_cutoff=unit.crop_cutoff,
        use_cross_interface_block=unit.use_cross_interface_block,
        planned_actions=_planned_actions_summary(unit.planned_runs),
        status=status,
        completed_runs=0,
        failed_runs=0,
        mean_aligned_ligand_rmsd=None,
        mean_training_seconds=None,
        local_output_root=unit.local_output_root,
        remote_output_root=unit.remote_output_root,
        last_heartbeat=_now(),
    )


def _completed_unit_status(unit: ScheduledUnit, metrics: UnitMetrics) -> UnitStatusRow:
    return UnitStatusRow(
        unit_id=unit.unit_id,
        worker_pool=unit.worker_pool,
        subtag=unit.subtag,
        complex_id=unit.complex_id,
        seed=unit.seed,
        steps=unit.steps,
        protein_node_budget=unit.protein_node_budget,
        crop_cutoff=unit.crop_cutoff,
        use_cross_interface_block=unit.use_cross_interface_block,
        planned_actions=_planned_actions_summary(unit.planned_runs),
        status="completed" if metrics.failed_runs == 0 else "failed",
        completed_runs=metrics.completed_runs,
        failed_runs=metrics.failed_runs,
        mean_aligned_ligand_rmsd=metrics.mean_aligned_ligand_rmsd,
        mean_training_seconds=metrics.mean_training_seconds,
        local_output_root=unit.local_output_root,
        remote_output_root=unit.remote_output_root,
        last_heartbeat=_now(),
        error=None if metrics.failed_runs == 0 else "child status.csv reported failed runs",
    )


def _failed_unit_status(unit: ScheduledUnit, error: str) -> UnitStatusRow:
    return replace(_initial_unit_status(unit, "failed"), error=error)


def _idle_worker_status(worker_name: str, worker_pool: str, assigned_units: int) -> WorkerStatusRow:
    return WorkerStatusRow(
        worker_name=worker_name,
        worker_pool=worker_pool,
        assigned_units=assigned_units,
        completed_units=0,
        failed_units=0,
        state="idle",
        current_subtag=None,
        last_heartbeat=_now(),
    )


def write_plan_csv(path: Path, units: list[ScheduledUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "unit_id",
                "worker_pool",
                "subtag",
                "complex_id",
                "seed",
                "steps",
                "protein_node_budget",
                "crop_cutoff",
                "use_cross_interface_block",
                "planned_actions",
                "local_output_root",
                "remote_output_root",
            ]
        )
        for unit in units:
            writer.writerow(
                [
                    unit.unit_id,
                    unit.worker_pool,
                    unit.subtag,
                    unit.complex_id,
                    unit.seed,
                    unit.steps,
                    unit.protein_node_budget,
                    f"{unit.crop_cutoff:.6f}",
                    str(unit.use_cross_interface_block).lower(),
                    _planned_actions_summary(unit.planned_runs),
                    unit.local_output_root.as_posix(),
                    unit.remote_output_root.as_posix(),
                ]
            )


def write_status_csv(path: Path, rows: list[UnitStatusRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "unit_id",
                "worker_pool",
                "subtag",
                "complex_id",
                "seed",
                "steps",
                "protein_node_budget",
                "crop_cutoff",
                "use_cross_interface_block",
                "planned_actions",
                "status",
                "completed_runs",
                "failed_runs",
                "mean_aligned_ligand_rmsd",
                "mean_training_seconds",
                "local_output_root",
                "remote_output_root",
                "last_heartbeat",
                "error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.unit_id,
                    row.worker_pool,
                    row.subtag,
                    row.complex_id,
                    row.seed,
                    row.steps,
                    row.protein_node_budget,
                    f"{row.crop_cutoff:.6f}",
                    str(row.use_cross_interface_block).lower(),
                    row.planned_actions,
                    row.status,
                    row.completed_runs,
                    row.failed_runs,
                    "" if row.mean_aligned_ligand_rmsd is None else f"{row.mean_aligned_ligand_rmsd:.6f}",
                    "" if row.mean_training_seconds is None else f"{row.mean_training_seconds:.3f}",
                    row.local_output_root.as_posix(),
                    row.remote_output_root.as_posix(),
                    row.last_heartbeat,
                    "" if row.error is None else row.error,
                ]
            )


def write_worker_status_csv(path: Path, rows: list[WorkerStatusRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "worker_name",
                "worker_pool",
                "assigned_units",
                "completed_units",
                "failed_units",
                "state",
                "current_subtag",
                "last_heartbeat",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.worker_name,
                    row.worker_pool,
                    row.assigned_units,
                    row.completed_units,
                    row.failed_units,
                    row.state,
                    "" if row.current_subtag is None else row.current_subtag,
                    row.last_heartbeat,
                ]
            )


def write_run_index_markdown(
    path: Path,
    *,
    dual_tag: str,
    units: list[ScheduledUnit],
    unit_statuses: list[UnitStatusRow],
    worker_statuses: list[WorkerStatusRow],
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(row.status == "completed" for row in unit_statuses)
    failed = sum(row.status == "failed" for row in unit_statuses)
    running = sum(row.status == "running" for row in unit_statuses)
    queued = sum(row.status in {"queued", "planned"} for row in unit_statuses)
    lines = [
        "# Dual-Host Run Index",
        "",
        f"- Dual tag: `{dual_tag}`",
        f"- Dry run: `{'yes' if dry_run else 'no'}`",
        f"- Planned units: `{len(units)}`",
        f"- Completed units: `{completed}`",
        f"- Failed units: `{failed}`",
        f"- Running units: `{running}`",
        f"- Queued/planned units: `{queued}`",
        "",
        "| Worker | Pool | State | Assigned | Completed | Failed | Current Subtag |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in worker_statuses:
        lines.append(
            f"| `{row.worker_name}` | `{row.worker_pool}` | `{row.state}` | `{row.assigned_units}` | "
            f"`{row.completed_units}` | `{row.failed_units}` | `{row.current_subtag or '-'}` |"
        )
    lines.extend(
        [
            "",
            "| Unit | Pool | Subtag | Complex | Seed | Steps | Crop | Status | Completed Runs | Failed Runs | Mean Aligned RMSD | Local Root | Remote Root |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    status_by_id = {row.unit_id: row for row in unit_statuses}
    for unit in units:
        row = status_by_id[unit.unit_id]
        aligned = "-" if row.mean_aligned_ligand_rmsd is None else f"{row.mean_aligned_ligand_rmsd:.6f}"
        lines.append(
            f"| `{unit.unit_id}` | `{unit.worker_pool}` | `{unit.subtag}` | `{unit.complex_id}` | `{unit.seed}` | "
            f"`{unit.steps}` | `{unit.crop_cutoff:.1f}` | `{row.status}` | `{row.completed_runs}` | "
            f"`{row.failed_runs}` | `{aligned}` | `{unit.local_output_root}` | `{unit.remote_output_root}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_unit(unit: ScheduledUnit, args: argparse.Namespace) -> UnitMetrics:
    argv = build_local_runner_argv(unit, args) if unit.worker_pool == "local_cpu" else build_remote_runner_argv(unit, args)
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        details = [f"unit command failed with exit code {completed.returncode}"]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        raise RuntimeError("\n\n".join(details))
    output_root = unit.local_output_root if unit.worker_pool == "local_cpu" else unit.remote_output_root
    return _load_run_metrics(output_root / "run_index.csv")


def _unit_worker_thread(
    *,
    worker_name: str,
    worker_pool: str,
    units: list[ScheduledUnit],
    args: argparse.Namespace,
    unit_statuses: dict[str, UnitStatusRow],
    worker_statuses: dict[str, WorkerStatusRow],
    dual_root: Path,
    lock: threading.Lock,
    all_units: list[ScheduledUnit],
) -> None:
    for unit in units:
        with lock:
            worker_row = worker_statuses[worker_name]
            worker_statuses[worker_name] = replace(
                worker_row,
                state="running",
                current_subtag=unit.subtag,
                last_heartbeat=_now(),
            )
            unit_statuses[unit.unit_id] = replace(unit_statuses[unit.unit_id], status="running", last_heartbeat=_now())
            _write_dual_outputs(dual_root, unit_statuses, worker_statuses, args, all_units)
        try:
            metrics = run_unit(unit, args)
        except Exception as exc:
            with lock:
                worker_row = worker_statuses[worker_name]
                worker_statuses[worker_name] = replace(
                    worker_row,
                    failed_units=worker_row.failed_units + 1,
                    state="idle",
                    current_subtag=None,
                    last_heartbeat=_now(),
                )
                unit_statuses[unit.unit_id] = _failed_unit_status(unit, str(exc))
                _write_dual_outputs(dual_root, unit_statuses, worker_statuses, args, all_units)
            continue
        with lock:
            worker_row = worker_statuses[worker_name]
            worker_statuses[worker_name] = replace(
                worker_row,
                completed_units=worker_row.completed_units + 1,
                state="idle",
                current_subtag=None,
                last_heartbeat=_now(),
            )
            unit_statuses[unit.unit_id] = _completed_unit_status(unit, metrics)
            _write_dual_outputs(dual_root, unit_statuses, worker_statuses, args, all_units)


def _write_dual_outputs(
    dual_root: Path,
    unit_statuses: dict[str, UnitStatusRow],
    worker_statuses: dict[str, WorkerStatusRow],
    args: argparse.Namespace,
    units: list[ScheduledUnit] | None = None,
) -> None:
    rows = [unit_statuses[key] for key in sorted(unit_statuses)]
    worker_rows = [worker_statuses[key] for key in sorted(worker_statuses)]
    write_status_csv(dual_root / "status.csv", rows)
    write_worker_status_csv(dual_root / "worker_status.csv", worker_rows)
    if units is None:
        units = []
    write_run_index_markdown(
        dual_root / "run_index.md",
        dual_tag=args.dual_tag,
        units=units,
        unit_statuses=rows,
        worker_statuses=worker_rows,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    dual_root = args.output_root / args.dual_tag
    units = plan_scheduled_units(args)
    write_plan_csv(dual_root / "plan.csv", units)

    local_units = [unit for unit in units if unit.worker_pool == "local_cpu"]
    remote_units = [unit for unit in units if unit.worker_pool == "remote_cpu"]
    local_buckets = [[] for _ in range(1 if args.laptop_mode == "smoke" else args.local_max_parallel)]
    remote_buckets = [[] for _ in range(args.remote_max_parallel)]
    for index, unit in enumerate(local_units):
        local_buckets[index % len(local_buckets)].append(unit)
    for index, unit in enumerate(remote_units):
        remote_buckets[index % len(remote_buckets)].append(unit)

    worker_slots = _worker_slots(args)
    worker_statuses: dict[str, WorkerStatusRow] = {}
    local_slot_index = 0
    remote_slot_index = 0
    for worker_name, worker_pool in worker_slots:
        if worker_pool == "local_cpu":
            assigned_units = len(local_buckets[local_slot_index])
            local_slot_index += 1
        else:
            assigned_units = len(remote_buckets[remote_slot_index])
            remote_slot_index += 1
        worker_statuses[worker_name] = _idle_worker_status(worker_name, worker_pool, assigned_units)

    initial_status = "planned" if args.dry_run else "queued"
    unit_statuses = {unit.unit_id: _initial_unit_status(unit, initial_status) for unit in units}
    _write_dual_outputs(dual_root, unit_statuses, worker_statuses, args, units)

    if args.dry_run:
        print(f"dual_root={dual_root}")
        print(f"planned_units={len(units)}")
        return 0
    has_remote_units = any(u.worker_pool != "local_cpu" for u in units)
    if has_remote_units and not args.dry_run and args.sync_mode != "none":
        local_root = Path.cwd()
        git_state = local_git_sync_state(local_root)
        resolved_mode = resolve_sync_mode(args.sync_mode, git_state)
        sync_remote_repo(local_root, args, mode=resolved_mode, git_state=git_state)


    lock = threading.Lock()
    futures = []
    with ThreadPoolExecutor(max_workers=len(worker_slots)) as executor:
        local_slot_index = 0
        remote_slot_index = 0
        for worker_name, worker_pool in worker_slots:
            if worker_pool == "local_cpu":
                bucket = local_buckets[local_slot_index]
                local_slot_index += 1
            else:
                bucket = remote_buckets[remote_slot_index]
                remote_slot_index += 1
            futures.append(
                executor.submit(
                    _unit_worker_thread,
                    worker_name=worker_name,
                    worker_pool=worker_pool,
                    units=bucket,
                    args=args,
                    unit_statuses=unit_statuses,
                    worker_statuses=worker_statuses,
                    dual_root=dual_root,
                    lock=lock,
                    all_units=units,
                )
            )
        for future in futures:
            future.result()

    _write_dual_outputs(dual_root, unit_statuses, worker_statuses, args, units)
    completed = sum(row.status == "completed" for row in unit_statuses.values())
    failed = sum(row.status == "failed" for row in unit_statuses.values())
    print(f"dual_root={dual_root}")
    print(f"completed_units={completed}")
    print(f"failed_units={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
