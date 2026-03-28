"""Compare two experiment-log selections and report paired deltas."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from equidock_diff.data.io import load_split_complex_ids
from equidock_diff.evaluation_summary import (
    ExperimentRecord,
    discover_records,
    filter_records_by_manifest,
    select_records,
    success_rate,
)


METRIC_FIELDS = (
    "best_loss",
    "final_loss",
    "raw_ligand_rmse",
    "aligned_ligand_rmsd",
    "training_seconds",
)


@dataclass(frozen=True)
class RecordPair:
    complex_id: str
    model: str
    seed: int
    noise_schedule: str
    left: ExperimentRecord
    right: ExperimentRecord


@dataclass(frozen=True)
class MissingPair:
    side: str
    complex_id: str
    model: str
    seed: int
    noise_schedule: str


@dataclass(frozen=True)
class DeltaRow:
    complex_id: str
    model: str
    noise_schedule: str
    seed_label: str
    matched_runs: int
    left_best_loss: float
    right_best_loss: float
    delta_best_loss: float
    left_final_loss: float
    right_final_loss: float
    delta_final_loss: float
    left_raw_ligand_rmse: float
    right_raw_ligand_rmse: float
    delta_raw_ligand_rmse: float
    left_aligned_ligand_rmsd: float
    right_aligned_ligand_rmsd: float
    delta_aligned_ligand_rmsd: float
    left_training_seconds: float
    right_training_seconds: float
    delta_training_seconds: float


@dataclass(frozen=True)
class AggregateDelta:
    compared_runs: int
    compared_complexes: int
    mean_best_loss_delta: float
    mean_final_loss_delta: float
    mean_raw_ligand_rmse_delta: float
    mean_aligned_ligand_rmsd_delta: float
    mean_training_seconds_delta: float
    left_success_at_2a: float
    right_success_at_2a: float
    delta_success_at_2a: float
    left_success_at_5a: float
    right_success_at_5a: float
    delta_success_at_5a: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired experiment logs and export per-complex or per-run deltas"
    )
    parser.add_argument("--left-glob", required=True, help="Glob used to discover the reference logs")
    parser.add_argument("--right-glob", required=True, help="Glob used to discover the candidate logs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional text file listing the complex ids to keep in the comparison",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Optional model label filter; may be passed multiple times",
    )
    parser.add_argument(
        "--schedule",
        dest="schedules",
        action="append",
        default=None,
        help="Optional noise-schedule filter; may be passed multiple times",
    )
    parser.add_argument(
        "--aggregate-by",
        choices=("run", "complex"),
        default="run",
        help="Whether to keep one row per paired run or average rows per complex/model/schedule",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        required=True,
        help="Path for the markdown comparison report",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for the CSV comparison export",
    )
    return parser


def _pair_key(record: ExperimentRecord) -> tuple[str, str, int, str]:
    return (record.complex_id, record.model, record.seed, record.noise_schedule)


def _filter_records(
    records: list[ExperimentRecord],
    *,
    models: list[str] | None,
    schedules: list[str] | None,
) -> list[ExperimentRecord]:
    allowed_models = set(models) if models else None
    allowed_schedules = set(schedules) if schedules else None
    filtered = [
        record
        for record in records
        if (allowed_models is None or record.model in allowed_models)
        and (allowed_schedules is None or record.noise_schedule in allowed_schedules)
    ]
    return sorted(
        filtered,
        key=lambda item: (item.complex_id, item.model, item.seed, item.noise_schedule),
    )


def discover_selected_records(
    pattern: str,
    *,
    manifest_complex_ids: list[str] | None,
    models: list[str] | None,
    schedules: list[str] | None,
) -> list[ExperimentRecord]:
    records = discover_records(pattern)
    records = filter_records_by_manifest(records, manifest_complex_ids)
    records = select_records(records, models="all")
    return _filter_records(records, models=models, schedules=schedules)


def pair_records(
    left_records: list[ExperimentRecord],
    right_records: list[ExperimentRecord],
) -> tuple[list[RecordPair], list[MissingPair]]:
    left_by_key = {_pair_key(record): record for record in left_records}
    right_by_key = {_pair_key(record): record for record in right_records}
    all_keys = sorted(set(left_by_key) | set(right_by_key))
    matched: list[RecordPair] = []
    missing: list[MissingPair] = []

    for key in all_keys:
        complex_id, model, seed, noise_schedule = key
        left = left_by_key.get(key)
        right = right_by_key.get(key)
        if left is None:
            missing.append(
                MissingPair(
                    side="left",
                    complex_id=complex_id,
                    model=model,
                    seed=seed,
                    noise_schedule=noise_schedule,
                )
            )
            continue
        if right is None:
            missing.append(
                MissingPair(
                    side="right",
                    complex_id=complex_id,
                    model=model,
                    seed=seed,
                    noise_schedule=noise_schedule,
                )
            )
            continue
        matched.append(
            RecordPair(
                complex_id=complex_id,
                model=model,
                seed=seed,
                noise_schedule=noise_schedule,
                left=left,
                right=right,
            )
        )
    return matched, missing


def _metric_value(record: ExperimentRecord, field: str) -> float:
    value = getattr(record, field)
    if value is None:
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def build_delta_rows(
    pairs: list[RecordPair],
    *,
    aggregate_by: str,
) -> list[DeltaRow]:
    if aggregate_by == "run":
        return [
            DeltaRow(
                complex_id=pair.complex_id,
                model=pair.model,
                noise_schedule=pair.noise_schedule,
                seed_label=str(pair.seed),
                matched_runs=1,
                left_best_loss=pair.left.best_loss,
                right_best_loss=pair.right.best_loss,
                delta_best_loss=pair.right.best_loss - pair.left.best_loss,
                left_final_loss=pair.left.final_loss,
                right_final_loss=pair.right.final_loss,
                delta_final_loss=pair.right.final_loss - pair.left.final_loss,
                left_raw_ligand_rmse=_metric_value(pair.left, "raw_ligand_rmse"),
                right_raw_ligand_rmse=_metric_value(pair.right, "raw_ligand_rmse"),
                delta_raw_ligand_rmse=_metric_value(pair.right, "raw_ligand_rmse")
                - _metric_value(pair.left, "raw_ligand_rmse"),
                left_aligned_ligand_rmsd=_metric_value(pair.left, "aligned_ligand_rmsd"),
                right_aligned_ligand_rmsd=_metric_value(pair.right, "aligned_ligand_rmsd"),
                delta_aligned_ligand_rmsd=_metric_value(pair.right, "aligned_ligand_rmsd")
                - _metric_value(pair.left, "aligned_ligand_rmsd"),
                left_training_seconds=pair.left.training_seconds,
                right_training_seconds=pair.right.training_seconds,
                delta_training_seconds=pair.right.training_seconds - pair.left.training_seconds,
            )
            for pair in pairs
        ]

    grouped: dict[tuple[str, str, str], list[RecordPair]] = {}
    for pair in pairs:
        grouped.setdefault((pair.complex_id, pair.model, pair.noise_schedule), []).append(pair)

    rows: list[DeltaRow] = []
    for (complex_id, model, noise_schedule), group in sorted(grouped.items()):
        rows.append(
            DeltaRow(
                complex_id=complex_id,
                model=model,
                noise_schedule=noise_schedule,
                seed_label="all",
                matched_runs=len(group),
                left_best_loss=_mean([item.left.best_loss for item in group]),
                right_best_loss=_mean([item.right.best_loss for item in group]),
                delta_best_loss=_mean([item.right.best_loss - item.left.best_loss for item in group]),
                left_final_loss=_mean([item.left.final_loss for item in group]),
                right_final_loss=_mean([item.right.final_loss for item in group]),
                delta_final_loss=_mean([item.right.final_loss - item.left.final_loss for item in group]),
                left_raw_ligand_rmse=_mean(
                    [_metric_value(item.left, "raw_ligand_rmse") for item in group]
                ),
                right_raw_ligand_rmse=_mean(
                    [_metric_value(item.right, "raw_ligand_rmse") for item in group]
                ),
                delta_raw_ligand_rmse=_mean(
                    [
                        _metric_value(item.right, "raw_ligand_rmse")
                        - _metric_value(item.left, "raw_ligand_rmse")
                        for item in group
                    ]
                ),
                left_aligned_ligand_rmsd=_mean(
                    [_metric_value(item.left, "aligned_ligand_rmsd") for item in group]
                ),
                right_aligned_ligand_rmsd=_mean(
                    [_metric_value(item.right, "aligned_ligand_rmsd") for item in group]
                ),
                delta_aligned_ligand_rmsd=_mean(
                    [
                        _metric_value(item.right, "aligned_ligand_rmsd")
                        - _metric_value(item.left, "aligned_ligand_rmsd")
                        for item in group
                    ]
                ),
                left_training_seconds=_mean([item.left.training_seconds for item in group]),
                right_training_seconds=_mean([item.right.training_seconds for item in group]),
                delta_training_seconds=_mean(
                    [item.right.training_seconds - item.left.training_seconds for item in group]
                ),
            )
        )
    return rows


def summarize_pairs(pairs: list[RecordPair]) -> AggregateDelta:
    left_aligned = [_metric_value(pair.left, "aligned_ligand_rmsd") for pair in pairs]
    right_aligned = [_metric_value(pair.right, "aligned_ligand_rmsd") for pair in pairs]
    return AggregateDelta(
        compared_runs=len(pairs),
        compared_complexes=len({pair.complex_id for pair in pairs}),
        mean_best_loss_delta=_mean([pair.right.best_loss - pair.left.best_loss for pair in pairs]),
        mean_final_loss_delta=_mean([pair.right.final_loss - pair.left.final_loss for pair in pairs]),
        mean_raw_ligand_rmse_delta=_mean(
            [
                _metric_value(pair.right, "raw_ligand_rmse")
                - _metric_value(pair.left, "raw_ligand_rmse")
                for pair in pairs
            ]
        ),
        mean_aligned_ligand_rmsd_delta=_mean(
            [
                _metric_value(pair.right, "aligned_ligand_rmsd")
                - _metric_value(pair.left, "aligned_ligand_rmsd")
                for pair in pairs
            ]
        ),
        mean_training_seconds_delta=_mean(
            [pair.right.training_seconds - pair.left.training_seconds for pair in pairs]
        ),
        left_success_at_2a=success_rate(left_aligned, 2.0),
        right_success_at_2a=success_rate(right_aligned, 2.0),
        delta_success_at_2a=success_rate(right_aligned, 2.0) - success_rate(left_aligned, 2.0),
        left_success_at_5a=success_rate(left_aligned, 5.0),
        right_success_at_5a=success_rate(right_aligned, 5.0),
        delta_success_at_5a=success_rate(right_aligned, 5.0) - success_rate(left_aligned, 5.0),
    )


def write_csv(path: Path, rows: list[DeltaRow], missing: list[MissingPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_id",
                "model",
                "noise_schedule",
                "seed",
                "matched_runs",
                "left_best_loss",
                "right_best_loss",
                "delta_best_loss",
                "left_final_loss",
                "right_final_loss",
                "delta_final_loss",
                "left_raw_ligand_rmse",
                "right_raw_ligand_rmse",
                "delta_raw_ligand_rmse",
                "left_aligned_ligand_rmsd",
                "right_aligned_ligand_rmsd",
                "delta_aligned_ligand_rmsd",
                "left_training_seconds",
                "right_training_seconds",
                "delta_training_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.complex_id,
                    row.model,
                    row.noise_schedule,
                    row.seed_label,
                    row.matched_runs,
                    f"{row.left_best_loss:.6f}",
                    f"{row.right_best_loss:.6f}",
                    f"{row.delta_best_loss:.6f}",
                    f"{row.left_final_loss:.6f}",
                    f"{row.right_final_loss:.6f}",
                    f"{row.delta_final_loss:.6f}",
                    f"{row.left_raw_ligand_rmse:.6f}",
                    f"{row.right_raw_ligand_rmse:.6f}",
                    f"{row.delta_raw_ligand_rmse:.6f}",
                    f"{row.left_aligned_ligand_rmsd:.6f}",
                    f"{row.right_aligned_ligand_rmsd:.6f}",
                    f"{row.delta_aligned_ligand_rmsd:.6f}",
                    f"{row.left_training_seconds:.3f}",
                    f"{row.right_training_seconds:.3f}",
                    f"{row.delta_training_seconds:.3f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["missing_side", "complex_id", "model", "seed", "noise_schedule"])
        for item in missing:
            writer.writerow([item.side, item.complex_id, item.model, item.seed, item.noise_schedule])


def write_markdown(
    path: Path,
    rows: list[DeltaRow],
    summary: AggregateDelta,
    *,
    left_glob: str,
    right_glob: str,
    aggregate_by: str,
    missing: list[MissingPair],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Comparison",
        "",
        f"- Left logs: `{left_glob}`",
        f"- Right logs: `{right_glob}`",
        f"- Aggregate mode: `{aggregate_by}`",
        f"- Compared runs: `{summary.compared_runs}`",
        f"- Compared complexes: `{summary.compared_complexes}`",
        "",
        "## Aggregate Deltas",
        "",
        "| Metric | Left | Right | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Success@2A | `{summary.left_success_at_2a:.1f}%` | `{summary.right_success_at_2a:.1f}%` | `{summary.delta_success_at_2a:+.1f}%` |",
        f"| Success@5A | `{summary.left_success_at_5a:.1f}%` | `{summary.right_success_at_5a:.1f}%` | `{summary.delta_success_at_5a:+.1f}%` |",
        f"| Mean Best Loss Delta | `-` | `-` | `{summary.mean_best_loss_delta:+.6f}` |",
        f"| Mean Final Loss Delta | `-` | `-` | `{summary.mean_final_loss_delta:+.6f}` |",
        f"| Mean Raw Ligand RMSE Delta | `-` | `-` | `{summary.mean_raw_ligand_rmse_delta:+.6f}` |",
        f"| Mean Aligned Ligand RMSD Delta | `-` | `-` | `{summary.mean_aligned_ligand_rmsd_delta:+.6f}` |",
        f"| Mean Training Seconds Delta | `-` | `-` | `{summary.mean_training_seconds_delta:+.3f}` |",
        "",
        "## Paired Rows",
        "",
        "| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.complex_id}` | {row.model} | {row.noise_schedule} | `{row.seed_label}` | `{row.matched_runs}` | "
            f"`{row.left_raw_ligand_rmse:.6f}` | `{row.right_raw_ligand_rmse:.6f}` | `{row.delta_raw_ligand_rmse:+.6f}` | "
            f"`{row.left_aligned_ligand_rmsd:.6f}` | `{row.right_aligned_ligand_rmsd:.6f}` | `{row.delta_aligned_ligand_rmsd:+.6f}` | "
            f"`{row.left_training_seconds:.3f}` | `{row.right_training_seconds:.3f}` | `{row.delta_training_seconds:+.3f}` |"
        )

    lines.extend(["", "## Missing Pairs", ""])
    if missing:
        lines.extend(
            [
                "| Missing On | Complex | Model | Seed | Schedule |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for item in missing:
            lines.append(
                f"| {item.side} | `{item.complex_id}` | {item.model} | `{item.seed}` | {item.noise_schedule} |"
            )
    else:
        lines.append("No missing pairs.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_complex_ids = None
    if args.manifest is not None:
        manifest_complex_ids = load_split_complex_ids(args.manifest)

    left_records = discover_selected_records(
        args.left_glob,
        manifest_complex_ids=manifest_complex_ids,
        models=args.models,
        schedules=args.schedules,
    )
    right_records = discover_selected_records(
        args.right_glob,
        manifest_complex_ids=manifest_complex_ids,
        models=args.models,
        schedules=args.schedules,
    )

    pairs, missing = pair_records(left_records, right_records)
    if not pairs:
        raise ValueError("No matched experiment logs found between the selected left/right globs.")

    rows = build_delta_rows(pairs, aggregate_by=args.aggregate_by)
    summary = summarize_pairs(pairs)
    write_csv(args.output_csv, rows, missing)
    write_markdown(
        args.output_markdown,
        rows,
        summary,
        left_glob=args.left_glob,
        right_glob=args.right_glob,
        aggregate_by=args.aggregate_by,
        missing=missing,
    )
    print(f"markdown_path={args.output_markdown}")
    print(f"csv_path={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
