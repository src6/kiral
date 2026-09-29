"""Launch small local research matrices with safe cache and checkpoint reuse."""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

import torch

from kiral.data.io import ProteinLigandPaths, filter_paths_by_complex_ids, load_paths, load_split_complex_ids
from kiral.evaluation_diff import main as evaluation_diff_main
from kiral.evaluation_summary import parse_experiment_log
from kiral.resample_from_checkpoint import main as resample_main
from kiral.train import build_parser as build_train_parser
from kiral.train import main as train_main
from kiral.train import resolve_device


RUNNER_MODEL_LABELS = {
    "baseline": "EGNN baseline",
    "frame_backbone": "heterogeneous frame-based backbone",
}
DEFAULT_RESEARCH_ROOT = Path("runs/research")


@dataclass(frozen=True)
class RunSpec:
    complex_id: str
    protein_path: Path
    ligand_path: Path
    model: str
    noise_schedule: str
    seed: int
    steps: int
    context_policy: str
    protein_node_budget: int
    crop_cutoff: float
    sample_steps: int
    sample_time_power: float
    ligand_shape_weight: float
    ligand_protein_clash_weight: float
    ligand_protein_contact_weight: float
    sample_score_clip: float
    sample_position_clip: float
    use_edge_attention: bool
    use_cross_interface_block: bool


@dataclass(frozen=True)
class PlannedRun:
    spec: RunSpec
    action: str
    run_name: str
    checkpoint_name: str
    checkpoint_path: Path
    log_path: Path
    loss_csv_path: Path
    sample_path: Path
    trajectory_path: Path
    plot_path: Path


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    complex_id: str
    model: str
    noise_schedule: str
    seed: int
    action: str
    device: str
    checkpoint_path: Path
    log_path: Path
    loss_csv_path: Path
    sample_path: Path | None
    trajectory_path: Path | None
    final_loss: float | None
    raw_ligand_rmse: float | None
    aligned_ligand_rmsd: float | None
    training_seconds: float | None


@dataclass(frozen=True)
class StatusRow:
    run_name: str
    complex_id: str
    model: str
    seed: int
    planned_action: str
    status: str
    device: str
    checkpoint_path: Path
    experiment_log: Path
    final_action: str | None = None
    final_loss: float | None = None
    raw_ligand_rmse: float | None = None
    aligned_ligand_rmsd: float | None = None
    training_seconds: float | None = None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    train_defaults = build_train_parser().parse_args([])
    parser = argparse.ArgumentParser(
        description="Run small local research matrices with checkpoint-aware routing"
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--complex-id",
        dest="complex_ids",
        action="append",
        default=None,
        help="Complex id to run; may be passed multiple times",
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
        default=Path("data/pdbbind_v2020"),
        help="Dataset root used to resolve complex ids into protein/ligand paths",
    )
    parser.add_argument(
        "--model",
        choices=tuple(RUNNER_MODEL_LABELS),
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
    parser.add_argument(
        "--crop-cutoff",
        dest="crop_cutoffs",
        action="append",
        type=float,
        default=None,
    )
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
        "--use-edge-attention",
        action="store_true",
        help="Enable lightweight edge attention inside the frame-backbone EGNN layers",
    )
    parser.add_argument(
        "--use-cross-interface-block",
        action="store_true",
        help="Enable the frame-backbone protein-to-ligand cross-message block",
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
        "--device-policy",
        choices=("scratch", "canonical", "explicit"),
        default="scratch",
        help="Device policy for this matrix",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device required when --device-policy explicit is used",
    )
    parser.add_argument("--tag", required=True, help="Local tag name under runs/research/")
    parser.add_argument(
        "--compare-against",
        default=None,
        help="Optional prior research tag to diff against automatically",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Keep pose, trajectory, and plot artifacts for each run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand the matrix and write the planned run table without executing runs",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of training-signature groups to execute concurrently",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="Optional delay between launching parallel groups",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue other groups after a group failure instead of failing fast",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESEARCH_ROOT,
        help="Root directory for local research outputs",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=train_defaults.dataset_cache_dir,
        help="Graph cache directory reused by train.py during research runs",
    )
    return parser


def _float_token(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if rendered == "-0":
        rendered = "0"
    return rendered.replace("-", "m").replace(".", "p")


def _model_training_args(model: str) -> list[str]:
    if model == "frame_backbone":
        return ["--frame-hetero-backbone", "--ligand-bond-weight", "0.1"]
    return []


def resolve_runner_device(device_policy: str, explicit_device: str | None) -> tuple[str, torch.device]:
    if device_policy == "canonical":
        requested = "cpu"
    elif device_policy == "explicit":
        if explicit_device is None:
            raise ValueError("--device is required when --device-policy explicit is used.")
        requested = explicit_device
    else:
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    return requested, resolve_device(requested)


def _resolve_examples(args: argparse.Namespace) -> list[ProteinLigandPaths]:
    all_paths = load_paths(args.dataset_root)
    if args.manifest is not None:
        return filter_paths_by_complex_ids(all_paths, load_split_complex_ids(args.manifest))
    assert args.complex_ids is not None
    return filter_paths_by_complex_ids(all_paths, args.complex_ids)


def _defaulted(values: list[object] | None, default: object) -> list[object]:
    if values:
        return list(values)
    return [default]


def _effective_protein_node_budget(context_policy: str, protein_node_budget: int) -> int:
    return protein_node_budget if context_policy == "gated" else 256


def expand_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    train_defaults = build_train_parser().parse_args([])
    examples = _resolve_examples(args)
    seeds = [int(value) for value in _defaulted(args.seeds, train_defaults.seed)]
    steps_values = [int(value) for value in _defaulted(args.steps_values, train_defaults.steps)]
    context_policies = [str(value) for value in _defaulted(args.context_policies, train_defaults.context_policy)]
    protein_node_budgets = [
        int(value) for value in _defaulted(args.protein_node_budgets, train_defaults.protein_node_budget)
    ]
    crop_cutoffs = [float(value) for value in _defaulted(args.crop_cutoffs, train_defaults.crop_cutoff)]
    sample_steps_values = [
        int(value) for value in _defaulted(args.sample_steps_values, train_defaults.sample_steps)
    ]
    sample_time_powers = [
        float(value) for value in _defaulted(args.sample_time_powers, train_defaults.sample_time_power)
    ]
    ligand_shape_weights = [
        float(value)
        for value in _defaulted(args.ligand_shape_weights, train_defaults.ligand_shape_weight)
    ]
    ligand_protein_clash_weights = [
        float(value)
        for value in _defaulted(
            args.ligand_protein_clash_weights,
            train_defaults.ligand_protein_clash_weight,
        )
    ]
    ligand_protein_contact_weights = [
        float(value)
        for value in _defaulted(
            args.ligand_protein_contact_weights,
            train_defaults.ligand_protein_contact_weight,
        )
    ]
    sample_score_clips = [
        float(value)
        for value in _defaulted(args.sample_score_clips, train_defaults.sample_score_clip)
    ]
    sample_position_clips = [
        float(value)
        for value in _defaulted(args.sample_position_clips, train_defaults.sample_position_clip)
    ]

    specs: list[RunSpec] = []
    for example in examples:
            for seed in sorted(seeds):
                for steps in sorted(steps_values):
                    for context_policy in sorted(context_policies):
                        budgets = (
                            sorted(protein_node_budgets)
                            if context_policy == "gated"
                            else [train_defaults.protein_node_budget]
                        )
                        for protein_node_budget in budgets:
                            for crop_cutoff in sorted(crop_cutoffs):
                                for sample_steps in sorted(sample_steps_values):
                                    for sample_time_power in sorted(sample_time_powers):
                                        for ligand_shape_weight in sorted(ligand_shape_weights):
                                            for ligand_protein_clash_weight in sorted(ligand_protein_clash_weights):
                                                for ligand_protein_contact_weight in sorted(
                                                    ligand_protein_contact_weights
                                                ):
                                                    for sample_score_clip in sorted(sample_score_clips):
                                                        for sample_position_clip in sorted(sample_position_clips):
                                                            specs.append(
                                                                RunSpec(
                                                                    complex_id=example.complex_id,
                                                                    protein_path=example.protein_path,
                                                                    ligand_path=example.ligand_path,
                                                                    model=args.model,
                                                                    noise_schedule=args.noise_schedule,
                                                                    seed=seed,
                                                                    steps=steps,
                                                                    context_policy=context_policy,
                                                                    protein_node_budget=protein_node_budget,
                                                                    crop_cutoff=crop_cutoff,
                                                                    sample_steps=sample_steps,
                                                                    sample_time_power=sample_time_power,
                                                                    ligand_shape_weight=ligand_shape_weight,
                                                                    ligand_protein_clash_weight=ligand_protein_clash_weight,
                                                                    ligand_protein_contact_weight=ligand_protein_contact_weight,
                                                                    sample_score_clip=sample_score_clip,
                                                                    sample_position_clip=sample_position_clip,
                                                                    use_edge_attention=bool(args.use_edge_attention),
                                                                    use_cross_interface_block=bool(
                                                                        args.use_cross_interface_block
                                                                    ),
                                                                )
                                                            )
    return sorted(
        specs,
        key=lambda item: (
            item.complex_id,
            item.model,
            item.noise_schedule,
            item.seed,
            item.steps,
            item.context_policy,
            item.protein_node_budget,
            item.crop_cutoff,
            item.ligand_shape_weight,
            item.ligand_protein_clash_weight,
            item.ligand_protein_contact_weight,
            item.sample_steps,
            item.sample_time_power,
            item.sample_score_clip,
            item.sample_position_clip,
            item.use_edge_attention,
            item.use_cross_interface_block,
        ),
    )


def training_signature(spec: RunSpec) -> tuple[object, ...]:
    return (
        spec.complex_id,
        spec.model,
        spec.noise_schedule,
        spec.seed,
        spec.steps,
        spec.context_policy,
        spec.protein_node_budget,
        spec.crop_cutoff,
        spec.ligand_shape_weight,
        spec.ligand_protein_clash_weight,
        spec.ligand_protein_contact_weight,
        spec.use_edge_attention,
        spec.use_cross_interface_block,
    )


def run_name_for_spec(spec: RunSpec) -> str:
    return (
        f"{spec.complex_id}_{spec.model}_{spec.noise_schedule}"
        f"_seed{spec.seed}_steps{spec.steps}"
        f"_ctx{spec.context_policy}"
        f"_pnb{spec.protein_node_budget}"
        f"_crop{_float_token(spec.crop_cutoff)}"
        f"_sample{spec.sample_steps}"
        f"_time{_float_token(spec.sample_time_power)}"
        f"_shape{_float_token(spec.ligand_shape_weight)}"
        f"_clash{_float_token(spec.ligand_protein_clash_weight)}"
        f"_contact{_float_token(spec.ligand_protein_contact_weight)}"
        f"_attn{1 if spec.use_edge_attention else 0}"
        f"_xmsg{1 if spec.use_cross_interface_block else 0}"
        f"_score{_float_token(spec.sample_score_clip)}"
        f"_pos{_float_token(spec.sample_position_clip)}"
    )


def checkpoint_name_for_spec(spec: RunSpec) -> str:
    return (
        f"{spec.complex_id}_{spec.model}_{spec.noise_schedule}"
        f"_seed{spec.seed}_steps{spec.steps}_ctx{spec.context_policy}_pnb{spec.protein_node_budget}_crop{_float_token(spec.crop_cutoff)}"
        f"_shape{_float_token(spec.ligand_shape_weight)}"
        f"_clash{_float_token(spec.ligand_protein_clash_weight)}"
        f"_contact{_float_token(spec.ligand_protein_contact_weight)}"
        f"_attn{1 if spec.use_edge_attention else 0}"
        f"_xmsg{1 if spec.use_cross_interface_block else 0}"
    )


def _run_paths(root: Path, run_name: str, checkpoint_name: str) -> dict[str, Path]:
    return {
        "checkpoint": root / "checkpoints" / f"{checkpoint_name}.pt",
        "log": root / "logs" / f"{run_name}_log.md",
        "loss_csv": root / "losses" / f"{run_name}_loss.csv",
        "sample": root / "artifacts" / f"{run_name}_sample.pdb",
        "trajectory": root / "artifacts" / f"{run_name}_traj.pdb",
        "plot": root / "plots" / f"{run_name}.png",
    }


def plan_runs(specs: list[RunSpec], root: Path) -> list[PlannedRun]:
    grouped: dict[tuple[object, ...], list[RunSpec]] = defaultdict(list)
    for spec in specs:
        grouped[training_signature(spec)].append(spec)

    planned: list[PlannedRun] = []
    for signature in sorted(grouped):
        group = sorted(
            grouped[signature],
            key=lambda item: (
                item.sample_steps,
                item.sample_time_power,
                item.sample_score_clip,
                item.sample_position_clip,
            ),
        )
        base_spec = group[0]
        base_checkpoint_name = checkpoint_name_for_spec(base_spec)
        base_paths = _run_paths(root, run_name_for_spec(base_spec), base_checkpoint_name)
        base_log_exists = base_paths["log"].exists()
        base_checkpoint_exists = base_paths["checkpoint"].exists()
        missing_non_base = any(
            not _run_paths(root, run_name_for_spec(spec), base_checkpoint_name)["log"].exists()
            for spec in group[1:]
        )
        if base_log_exists and (base_checkpoint_exists or not missing_non_base):
            base_action = "reuse"
        elif base_checkpoint_exists:
            base_action = "resample"
        else:
            base_action = "train"

        for index, spec in enumerate(group):
            run_name = run_name_for_spec(spec)
            paths = _run_paths(root, run_name, base_checkpoint_name)
            if index == 0:
                action = base_action
            elif paths["log"].exists():
                action = "reuse"
            else:
                action = "resample"
            planned.append(
                PlannedRun(
                    spec=spec,
                    action=action,
                    run_name=run_name,
                    checkpoint_name=base_checkpoint_name,
                    checkpoint_path=paths["checkpoint"],
                    log_path=paths["log"],
                    loss_csv_path=paths["loss_csv"],
                    sample_path=paths["sample"],
                    trajectory_path=paths["trajectory"],
                    plot_path=paths["plot"],
                )
            )
    return planned


def _train_argv(
    planned: PlannedRun,
    *,
    requested_device: str,
    dataset_cache_dir: Path,
    save_artifacts: bool,
) -> list[str]:
    spec = planned.spec
    argv = [
        "--device",
        requested_device,
        "--seed",
        str(spec.seed),
        "--steps",
        str(spec.steps),
        "--context-policy",
        spec.context_policy,
        "--protein-node-budget",
        str(spec.protein_node_budget),
        "--crop-cutoff",
        str(spec.crop_cutoff),
        "--sample-steps",
        str(spec.sample_steps),
        "--sample-time-power",
        str(spec.sample_time_power),
        "--sample-score-clip",
        str(spec.sample_score_clip),
        "--sample-position-clip",
        str(spec.sample_position_clip),
        "--ligand-shape-weight",
        str(spec.ligand_shape_weight),
        "--ligand-protein-clash-weight",
        str(spec.ligand_protein_clash_weight),
        "--ligand-protein-contact-weight",
        str(spec.ligand_protein_contact_weight),
        "--noise-schedule",
        spec.noise_schedule,
        "--protein-path",
        str(spec.protein_path),
        "--ligand-path",
        str(spec.ligand_path),
        "--dataset-cache-dir",
        str(dataset_cache_dir),
        "--checkpoint-path",
        str(planned.checkpoint_path),
        "--loss-csv",
        str(planned.loss_csv_path),
        "--experiment-log",
        str(planned.log_path),
        "--output",
        str(planned.sample_path),
        "--trajectory-output",
        str(planned.trajectory_path),
        "--plot-output",
        str(planned.plot_path),
    ]
    argv.extend(_model_training_args(spec.model))
    if spec.use_edge_attention:
        argv.append("--use-edge-attention")
    if spec.use_cross_interface_block:
        argv.append("--use-cross-interface-block")
    if not save_artifacts:
        argv.extend(["--skip-pose-artifacts", "--skip-plot"])
    return argv


def _resample_argv(
    planned: PlannedRun,
    *,
    requested_device: str,
    save_artifacts: bool,
) -> list[str]:
    spec = planned.spec
    argv = [
        "--checkpoint",
        str(planned.checkpoint_path),
        "--device",
        requested_device,
        "--sample-steps",
        str(spec.sample_steps),
        "--sample-time-power",
        str(spec.sample_time_power),
        "--sample-score-clip",
        str(spec.sample_score_clip),
        "--sample-position-clip",
        str(spec.sample_position_clip),
        "--loss-csv",
        str(planned.loss_csv_path),
        "--experiment-log",
        str(planned.log_path),
        "--output",
        str(planned.sample_path),
        "--trajectory-output",
        str(planned.trajectory_path),
        "--plot-output",
        str(planned.plot_path),
    ]
    if not save_artifacts:
        argv.extend(["--skip-pose-artifacts", "--skip-plot"])
    return argv


def execute_train(
    planned: PlannedRun,
    *,
    requested_device: str,
    dataset_cache_dir: Path,
    save_artifacts: bool,
) -> int:
    return train_main(
        _train_argv(
            planned,
            requested_device=requested_device,
            dataset_cache_dir=dataset_cache_dir,
            save_artifacts=save_artifacts,
        )
    )


def execute_resample(
    planned: PlannedRun,
    *,
    requested_device: str,
    save_artifacts: bool,
) -> int:
    return resample_main(
        _resample_argv(
            planned,
            requested_device=requested_device,
            save_artifacts=save_artifacts,
        )
    )


def summarize_run(
    planned: PlannedRun,
    *,
    action: str,
    actual_device: torch.device,
    save_artifacts: bool,
) -> RunSummary:
    record = parse_experiment_log(planned.log_path)
    return RunSummary(
        run_name=planned.run_name,
        complex_id=record.complex_id,
        model=record.model,
        noise_schedule=record.noise_schedule,
        seed=record.seed,
        action=action,
        device=actual_device.type,
        checkpoint_path=planned.checkpoint_path,
        log_path=planned.log_path,
        loss_csv_path=planned.loss_csv_path,
        sample_path=None if not save_artifacts else planned.sample_path,
        trajectory_path=None if not save_artifacts else planned.trajectory_path,
        final_loss=record.final_loss,
        raw_ligand_rmse=record.raw_ligand_rmse,
        aligned_ligand_rmsd=record.aligned_ligand_rmsd,
        training_seconds=record.training_seconds,
    )


def summarize_planned_run(
    planned: PlannedRun,
    *,
    actual_device: torch.device,
    save_artifacts: bool,
) -> RunSummary:
    return RunSummary(
        run_name=planned.run_name,
        complex_id=planned.spec.complex_id,
        model=RUNNER_MODEL_LABELS[planned.spec.model],
        noise_schedule=planned.spec.noise_schedule,
        seed=planned.spec.seed,
        action=f"planned_{planned.action}",
        device=actual_device.type,
        checkpoint_path=planned.checkpoint_path,
        log_path=planned.log_path,
        loss_csv_path=planned.loss_csv_path,
        sample_path=None if not save_artifacts else planned.sample_path,
        trajectory_path=None if not save_artifacts else planned.trajectory_path,
        final_loss=None,
        raw_ligand_rmse=None,
        aligned_ligand_rmsd=None,
        training_seconds=None,
    )


def write_run_index_csv(path: Path, summaries: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "complex_id",
                "model",
                "noise_schedule",
                "seed",
                "action",
                "device",
                "checkpoint_path",
                "experiment_log",
                "loss_csv",
                "sample_path",
                "trajectory_path",
                "final_loss",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.run_name,
                    summary.complex_id,
                    summary.model,
                    summary.noise_schedule,
                    summary.seed,
                    summary.action,
                    summary.device,
                    summary.checkpoint_path.as_posix(),
                    summary.log_path.as_posix(),
                    summary.loss_csv_path.as_posix(),
                    "" if summary.sample_path is None else summary.sample_path.as_posix(),
                    "" if summary.trajectory_path is None else summary.trajectory_path.as_posix(),
                    "" if summary.final_loss is None else f"{summary.final_loss:.6f}",
                    "" if summary.raw_ligand_rmse is None else f"{summary.raw_ligand_rmse:.6f}",
                    ""
                    if summary.aligned_ligand_rmsd is None
                    else f"{summary.aligned_ligand_rmsd:.6f}",
                    "" if summary.training_seconds is None else f"{summary.training_seconds:.3f}",
                ]
            )


def write_run_index_markdown(
    path: Path,
    *,
    tag: str,
    summaries: list[RunSummary],
    compare_against: str | None,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_runs = sum(1 for item in summaries if "train" in item.action)
    resample_runs = sum(1 for item in summaries if "resample" in item.action)
    reuse_runs = sum(1 for item in summaries if "reuse" in item.action)
    aligned = [item.aligned_ligand_rmsd for item in summaries if item.aligned_ligand_rmsd is not None]
    lines = [
        "# Research Run Index",
        "",
        f"- Tag: `{tag}`",
        f"- Dry run: `{'yes' if dry_run else 'no'}`",
        f"- Train actions: `{train_runs}`",
        f"- Resample actions: `{resample_runs}`",
        f"- Reused runs: `{reuse_runs}`",
    ]
    if compare_against is not None:
        lines.append(f"- Compared against: `{compare_against}`")
    if aligned:
        lines.append(f"- Mean aligned RMSD: `{statistics.fmean(aligned):.6f}`")
    lines.extend(
        [
            "",
            "| Run | Complex | Model | Seed | Action | Device | Final Loss | Raw RMSE | Aligned RMSD | Training Seconds |",
            "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        final_loss = "-" if summary.final_loss is None else f"{summary.final_loss:.6f}"
        raw_rmse = "-" if summary.raw_ligand_rmse is None else f"{summary.raw_ligand_rmse:.6f}"
        aligned = "-" if summary.aligned_ligand_rmsd is None else f"{summary.aligned_ligand_rmsd:.6f}"
        training_seconds = "-" if summary.training_seconds is None else f"{summary.training_seconds:.3f}"
        lines.append(
            f"| `{summary.run_name}` | `{summary.complex_id}` | {summary.model} | `{summary.seed}` | "
            f"`{summary.action}` | `{summary.device}` | `{final_loss}` | `{raw_rmse}` | "
            f"`{aligned}` | `{training_seconds}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status_csv(path: Path, statuses: list[StatusRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "complex_id",
                "model",
                "seed",
                "planned_action",
                "status",
                "device",
                "checkpoint_path",
                "experiment_log",
                "final_action",
                "final_loss",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
                "error",
            ]
        )
        for row in statuses:
            writer.writerow(
                [
                    row.run_name,
                    row.complex_id,
                    row.model,
                    row.seed,
                    row.planned_action,
                    row.status,
                    row.device,
                    row.checkpoint_path.as_posix(),
                    row.experiment_log.as_posix(),
                    "" if row.final_action is None else row.final_action,
                    "" if row.final_loss is None else f"{row.final_loss:.6f}",
                    "" if row.raw_ligand_rmse is None else f"{row.raw_ligand_rmse:.6f}",
                    "" if row.aligned_ligand_rmsd is None else f"{row.aligned_ligand_rmsd:.6f}",
                    "" if row.training_seconds is None else f"{row.training_seconds:.3f}",
                    "" if row.error is None else row.error,
                ]
            )


def _planned_status_row(planned: PlannedRun, *, actual_device: torch.device, status: str) -> StatusRow:
    return StatusRow(
        run_name=planned.run_name,
        complex_id=planned.spec.complex_id,
        model=RUNNER_MODEL_LABELS[planned.spec.model],
        seed=planned.spec.seed,
        planned_action=planned.action,
        status=status,
        device=actual_device.type,
        checkpoint_path=planned.checkpoint_path,
        experiment_log=planned.log_path,
    )


def _completed_status_row(
    planned: PlannedRun,
    *,
    actual_device: torch.device,
    final_action: str,
    summary: RunSummary,
) -> StatusRow:
    return StatusRow(
        run_name=planned.run_name,
        complex_id=planned.spec.complex_id,
        model=RUNNER_MODEL_LABELS[planned.spec.model],
        seed=planned.spec.seed,
        planned_action=planned.action,
        status="completed",
        device=actual_device.type,
        checkpoint_path=planned.checkpoint_path,
        experiment_log=planned.log_path,
        final_action=final_action,
        final_loss=summary.final_loss,
        raw_ligand_rmse=summary.raw_ligand_rmse,
        aligned_ligand_rmsd=summary.aligned_ligand_rmsd,
        training_seconds=summary.training_seconds,
    )


def _failed_status_row(
    planned: PlannedRun,
    *,
    actual_device: torch.device,
    error: str,
) -> StatusRow:
    row = _planned_status_row(planned, actual_device=actual_device, status="failed")
    return StatusRow(**{**row.__dict__, "error": error})


def _group_planned_runs(planned: list[PlannedRun]) -> list[list[PlannedRun]]:
    grouped: dict[tuple[object, ...], list[PlannedRun]] = defaultdict(list)
    for run in planned:
        grouped[training_signature(run.spec)].append(run)
    groups = []
    for signature in sorted(grouped):
        group = sorted(
            grouped[signature],
            key=lambda item: (
                item.spec.sample_steps,
                item.spec.sample_time_power,
                item.spec.sample_score_clip,
                item.spec.sample_position_clip,
            ),
        )
        groups.append(group)
    return groups


def _execute_group(
    group: list[PlannedRun],
    *,
    requested_device: str,
    actual_device: torch.device,
    dataset_cache_dir: Path,
    save_artifacts: bool,
    status_path: Path,
    statuses: dict[str, StatusRow],
    status_lock: threading.Lock,
) -> list[RunSummary]:
    group_summaries: list[RunSummary] = []
    for run in group:
        with status_lock:
            statuses[run.run_name] = _planned_status_row(run, actual_device=actual_device, status="running")
            write_status_csv(status_path, list(statuses.values()))
        try:
            if run.action == "reuse":
                summary = summarize_run(
                    run,
                    action="reuse_existing",
                    actual_device=actual_device,
                    save_artifacts=save_artifacts,
                )
            elif run.action == "train":
                result = execute_train(
                    run,
                    requested_device=requested_device,
                    dataset_cache_dir=dataset_cache_dir,
                    save_artifacts=save_artifacts,
                )
                if result != 0:
                    raise RuntimeError(f"train returned a non-zero status for {run.run_name}")
                summary = summarize_run(
                    run,
                    action="trained_from_scratch",
                    actual_device=actual_device,
                    save_artifacts=save_artifacts,
                )
            else:
                if not run.checkpoint_path.exists():
                    raise ValueError(f"Checkpoint required for resampling does not exist: {run.checkpoint_path}")
                result = execute_resample(
                    run,
                    requested_device=requested_device,
                    save_artifacts=save_artifacts,
                )
                if result != 0:
                    raise RuntimeError(f"resample returned a non-zero status for {run.run_name}")
                summary = summarize_run(
                    run,
                    action="resampled_from_checkpoint",
                    actual_device=actual_device,
                    save_artifacts=save_artifacts,
                )
            group_summaries.append(summary)
            with status_lock:
                statuses[run.run_name] = _completed_status_row(
                    run,
                    actual_device=actual_device,
                    final_action=summary.action,
                    summary=summary,
                )
                write_status_csv(status_path, list(statuses.values()))
        except Exception as exc:
            with status_lock:
                statuses[run.run_name] = _failed_status_row(run, actual_device=actual_device, error=str(exc))
                write_status_csv(status_path, list(statuses.values()))
            raise
    return group_summaries


def run_comparison(
    *,
    current_root: Path,
    compare_against: str,
    model: str,
    noise_schedule: str,
) -> tuple[Path, Path]:
    previous_root = current_root.parent / compare_against
    if not previous_root.exists():
        raise ValueError(f"Comparison tag {compare_against!r} does not exist under {current_root.parent}.")
    comparison_dir = current_root / "comparisons"
    markdown_path = comparison_dir / f"vs_{compare_against}.md"
    csv_path = comparison_dir / f"vs_{compare_against}.csv"
    result = evaluation_diff_main(
        [
            "--left-glob",
            str(previous_root / "logs" / "*_log.md"),
            "--right-glob",
            str(current_root / "logs" / "*_log.md"),
            "--model",
            RUNNER_MODEL_LABELS[model],
            "--schedule",
            noise_schedule,
            "--aggregate-by",
            "run",
            "--output-markdown",
            str(markdown_path),
            "--output-csv",
            str(csv_path),
        ]
    )
    if result != 0:
        raise RuntimeError("evaluation_diff returned a non-zero status.")
    return markdown_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")
    if args.stagger_seconds < 0.0:
        raise ValueError("--stagger-seconds must be non-negative.")
    if args.use_edge_attention and args.model != "frame_backbone":
        raise ValueError("--use-edge-attention currently requires --model frame_backbone.")
    if args.use_cross_interface_block and args.model != "frame_backbone":
        raise ValueError("--use-cross-interface-block currently requires --model frame_backbone.")
    if args.use_cross_interface_block and args.use_edge_attention:
        raise ValueError("--use-cross-interface-block cannot be combined with --use-edge-attention in this cycle.")
    requested_device, actual_device = resolve_runner_device(args.device_policy, args.device)
    root = args.output_root / args.tag
    specs = expand_run_specs(args)
    planned = plan_runs(specs, root)
    status_path = root / "status.csv"

    summaries: list[RunSummary] = []
    if args.dry_run:
        summaries = [
            summarize_planned_run(run, actual_device=actual_device, save_artifacts=args.save_artifacts)
            for run in planned
        ]
        write_status_csv(
            status_path,
            [_planned_status_row(run, actual_device=actual_device, status="planned") for run in planned],
        )
    else:
        grouped_runs = _group_planned_runs(planned)
        statuses = {
            run.run_name: _planned_status_row(run, actual_device=actual_device, status="queued")
            for run in planned
        }
        status_lock = threading.Lock()
        write_status_csv(status_path, list(statuses.values()))
        errors: list[Exception] = []
        if args.max_parallel == 1:
            for group in grouped_runs:
                try:
                    summaries.extend(
                        _execute_group(
                            group,
                            requested_device=requested_device,
                            actual_device=actual_device,
                            dataset_cache_dir=args.dataset_cache_dir,
                            save_artifacts=args.save_artifacts,
                            status_path=status_path,
                            statuses=statuses,
                            status_lock=status_lock,
                        )
                    )
                except Exception as exc:
                    errors.append(exc)
                    if not args.keep_going:
                        break
        else:
            with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
                futures = []
                for index, group in enumerate(grouped_runs):
                    future = executor.submit(
                        _execute_group,
                        group,
                        requested_device=requested_device,
                        actual_device=actual_device,
                        dataset_cache_dir=args.dataset_cache_dir,
                        save_artifacts=args.save_artifacts,
                        status_path=status_path,
                        statuses=statuses,
                        status_lock=status_lock,
                    )
                    futures.append(future)
                    if args.stagger_seconds > 0.0 and index != len(grouped_runs) - 1:
                        time.sleep(args.stagger_seconds)
                if args.keep_going:
                    for future in futures:
                        try:
                            summaries.extend(future.result())
                        except Exception as exc:
                            errors.append(exc)
                else:
                    done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
                    for future in done:
                        try:
                            summaries.extend(future.result())
                        except Exception as exc:
                            errors.append(exc)
                    if errors:
                        for future in not_done:
                            future.cancel()
                    else:
                        for future in not_done:
                            summaries.extend(future.result())
        summaries = sorted(summaries, key=lambda item: item.run_name)
        if errors:
            raise RuntimeError(str(errors[0]))

    summary_csv = root / "run_index.csv"
    summary_md = root / "run_index.md"
    write_run_index_csv(summary_csv, summaries)
    write_run_index_markdown(
        summary_md,
        tag=args.tag,
        summaries=summaries,
        compare_against=args.compare_against,
        dry_run=args.dry_run,
    )
    print(f"run_index_csv={summary_csv}")
    print(f"run_index_markdown={summary_md}")
    print(f"status_csv={status_path}")

    if args.compare_against is not None and not args.dry_run:
        comparison_md, comparison_csv = run_comparison(
            current_root=root,
            compare_against=args.compare_against,
            model=args.model,
            noise_schedule=args.noise_schedule,
        )
        print(f"comparison_markdown={comparison_md}")
        print(f"comparison_csv={comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
