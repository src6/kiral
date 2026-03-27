"""Aggregate experiment logs into report-ready comparison summaries."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from equidock_diff.data.io import load_split_complex_ids


KV_LINE_RE = re.compile(r"^- (?P<key>[^:]+): `(?P<value>.*)`$")
FINAL_LOSS_RE = re.compile(r"^- Final loss: `(?P<loss>[^`]+)` at step `(?P<step>[^`]+)`$")
PROTEIN_PATH_RE = re.compile(r"/(?P<complex_id>[^/]+)/(?P=complex_id)_protein\.pdb")
PRIMARY_MODELS = ("EGNN baseline", "heterogeneous frame-based backbone")


@dataclass(frozen=True)
class ExperimentRecord:
    complex_id: str
    model: str
    seed: int
    noise_schedule: str
    log_path: Path
    command: str
    graph_source: str
    training_steps: int
    sample_steps: int
    node_count: int
    edge_count: int
    best_loss: float
    final_loss: float
    training_seconds: float
    raw_ligand_rmse: float | None
    aligned_ligand_rmsd: float | None


@dataclass(frozen=True)
class SummaryRow:
    model: str
    noise_schedule: str | None
    complexes: int
    runs: int
    mean_best_loss: float
    mean_final_loss: float
    mean_raw_ligand_rmse: float
    std_raw_ligand_rmse: float
    mean_aligned_ligand_rmsd: float
    std_aligned_ligand_rmsd: float
    success_at_2a: float
    success_at_5a: float
    mean_training_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment logs into a cross-complex comparison summary"
    )
    parser.add_argument(
        "--log-glob",
        default="docs/training/*log.md",
        help="Glob used to discover experiment logs",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/training/cross_complex_comparison.md"),
        help="Path for the markdown summary",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("docs/training/cross_complex_comparison.csv"),
        help="Path for the CSV export",
    )
    parser.add_argument(
        "--models",
        choices=("primary", "all"),
        default="primary",
        help="Whether to summarise only the baseline/final comparison or all discovered model variants",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional text file listing the complex ids to include in canonical summaries",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any expected complex/model/seed combination is missing after selection",
    )
    parser.add_argument(
        "--expected-model",
        dest="expected_models",
        action="append",
        default=None,
        help="Model label expected for each manifest complex; may be passed multiple times",
    )
    parser.add_argument(
        "--expected-seed",
        dest="expected_seeds",
        type=int,
        action="append",
        default=None,
        help="Seed expected for each manifest complex; may be passed multiple times",
    )
    parser.add_argument(
        "--expected-schedule",
        dest="expected_schedules",
        action="append",
        default=None,
        help="Noise schedule expected for each manifest complex/model/seed combination; may be passed multiple times",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=None,
        help="Optional path for a LaTeX summary table",
    )
    parser.add_argument(
        "--group-by",
        choices=("model", "model_schedule"),
        default="model",
        help="How to aggregate records in the summary outputs",
    )
    return parser


def parse_experiment_log(path: Path) -> ExperimentRecord:
    values: dict[str, str] = {}
    final_loss = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            final_match = FINAL_LOSS_RE.match(line)
            if final_match is not None:
                values["Final loss"] = final_match.group("loss")
                values["Final step"] = final_match.group("step")
                final_loss = float(final_match.group("loss"))
                continue
            kv_match = KV_LINE_RE.match(line)
            if kv_match is not None:
                values[kv_match.group("key")] = kv_match.group("value")

    if final_loss is None:
        raise ValueError(f"Could not parse final loss from {path}.")

    command = values["Command"]
    complex_id = infer_complex_id(command, path)
    model = infer_model(command)
    return ExperimentRecord(
        complex_id=complex_id,
        model=model,
        seed=int(values.get("Seed", "42")),
        noise_schedule=infer_noise_schedule(command),
        log_path=path,
        command=command,
        graph_source=values.get("Graph source", ""),
        training_steps=int(values["Training steps"]),
        sample_steps=int(values["Sample steps"]),
        node_count=int(values["Node count"]),
        edge_count=int(values["Edge count"]),
        best_loss=float(values["Best loss"]),
        final_loss=final_loss,
        training_seconds=float(values["Training seconds"]),
        raw_ligand_rmse=_optional_float(values.get("Raw Ligand Rmse")),
        aligned_ligand_rmsd=_optional_float(values.get("Aligned Ligand Rmsd")),
    )


def infer_complex_id(command: str, path: Path) -> str:
    match = PROTEIN_PATH_RE.search(command)
    if match is not None:
        return match.group("complex_id")
    return path.name.split("_", 1)[0]


def infer_model(command: str) -> str:
    if "--hetgnn-backbone" in command or "--frame-hetero-backbone" in command:
        return "heterogeneous frame-based backbone"
    if "--complete-frame" in command and "--ligand-global-node" in command:
        return "EGNN + complete frames + ligand context"
    if "--complete-frame" in command:
        return "EGNN + complete frames"
    if "--ligand-global-node" in command:
        return "EGNN + ligand context"
    if "--hetero-edges" in command:
        return "EGNN + typed edges"
    return "EGNN baseline"


def infer_noise_schedule(command: str) -> str:
    if "--noise-schedule cosine" in command:
        return "cosine"
    return "linear"


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def discover_records(pattern: str) -> list[ExperimentRecord]:
    records = [parse_experiment_log(path) for path in sorted(Path().glob(pattern))]
    return [
        record
        for record in records
        if record.graph_source == "real_pair"
        and record.raw_ligand_rmse is not None
        and record.aligned_ligand_rmsd is not None
    ]


def filter_records_by_manifest(
    records: list[ExperimentRecord],
    manifest_complex_ids: list[str] | None,
) -> list[ExperimentRecord]:
    if manifest_complex_ids is None:
        return records

    expected = set(manifest_complex_ids)
    filtered = [record for record in records if record.complex_id in expected]
    return sorted(
        filtered,
        key=lambda item: (
            manifest_complex_ids.index(item.complex_id),
            item.model,
            item.seed,
            item.log_path.as_posix(),
        ),
    )


def select_records(records: list[ExperimentRecord], *, models: str) -> list[ExperimentRecord]:
    preferred_models = set(PRIMARY_MODELS)
    deduped: dict[tuple[str, str, int, str], ExperimentRecord] = {}
    for record in records:
        if models == "primary" and record.model not in preferred_models:
            continue
        key = (record.complex_id, record.model, record.seed, record.noise_schedule)
        current = deduped.get(key)
        if current is None or _record_priority(record) < _record_priority(current):
            deduped[key] = record
    return sorted(
        deduped.values(),
        key=lambda item: (item.complex_id, item.model, item.seed, item.noise_schedule),
    )


def _record_priority(
    record: ExperimentRecord,
) -> tuple[int, int, int, int, int, int, int]:
    name = record.log_path.name
    return (
        0 if record.training_steps >= 100 else 1,
        0 if record.sample_steps >= 25 else 1,
        1 if "rerun" in name else 0,
        1 if "paper" in name else 0,
        1 if "repro_check" in name else 0,
        1 if "real_experiment" in name else 0,
        len(name),
    )


def resolve_expected_models(
    *,
    models: str,
    expected_models: list[str] | None,
) -> list[str]:
    if expected_models:
        return expected_models
    if models == "primary":
        return list(PRIMARY_MODELS)
    return []


def resolve_expected_seeds(expected_seeds: list[int] | None) -> list[int]:
    return [int(seed) for seed in expected_seeds] if expected_seeds else [42]


def resolve_expected_schedules(expected_schedules: list[str] | None) -> list[str]:
    return list(expected_schedules) if expected_schedules else ["linear"]


def assert_expected_combinations_present(
    records: list[ExperimentRecord],
    *,
    manifest_complex_ids: list[str] | None,
    expected_models: list[str],
    expected_seeds: list[int],
    expected_schedules: list[str],
) -> None:
    if manifest_complex_ids is None:
        raise ValueError("--require-complete requires --manifest.")

    present = {
        (record.complex_id, record.model, record.seed, record.noise_schedule)
        for record in records
    }
    missing: list[str] = []
    for complex_id in manifest_complex_ids:
        for model in expected_models:
            for seed in expected_seeds:
                for schedule in expected_schedules:
                    key = (complex_id, model, seed, schedule)
                    if key not in present:
                        missing.append(f"{complex_id}/{model}/seed={seed}/schedule={schedule}")

    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Missing expected experiment logs: {preview}")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def success_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    successes = sum(1 for value in values if value <= threshold)
    return 100.0 * successes / len(values)


def grouped_means(
    records: list[ExperimentRecord],
    *,
    group_by: str = "model",
) -> list[SummaryRow]:
    groups: dict[tuple[str, str | None], list[ExperimentRecord]] = {}
    for record in records:
        key = (record.model, record.noise_schedule if group_by == "model_schedule" else None)
        groups.setdefault(key, []).append(record)

    rows: list[SummaryRow] = []
    for (model, noise_schedule), group in sorted(groups.items()):
        raw_rmses = [float(item.raw_ligand_rmse or 0.0) for item in group]
        aligned_rmsds = [float(item.aligned_ligand_rmsd or 0.0) for item in group]
        rows.append(
            SummaryRow(
                model=model,
                noise_schedule=noise_schedule,
                complexes=len({item.complex_id for item in group}),
                runs=len(group),
                mean_best_loss=_mean([item.best_loss for item in group]),
                mean_final_loss=_mean([item.final_loss for item in group]),
                mean_raw_ligand_rmse=_mean(raw_rmses),
                std_raw_ligand_rmse=_stddev(raw_rmses),
                mean_aligned_ligand_rmsd=_mean(aligned_rmsds),
                std_aligned_ligand_rmsd=_stddev(aligned_rmsds),
                success_at_2a=success_rate(aligned_rmsds, 2.0),
                success_at_5a=success_rate(aligned_rmsds, 5.0),
                mean_training_seconds=_mean([item.training_seconds for item in group]),
            )
        )
    return rows


def write_csv(path: Path, records: list[ExperimentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_id",
                "model",
                "seed",
                "noise_schedule",
                "best_loss",
                "final_loss",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
                "node_count",
                "edge_count",
                "log_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.complex_id,
                    record.model,
                    record.seed,
                    record.noise_schedule,
                    f"{record.best_loss:.6f}",
                    f"{record.final_loss:.6f}",
                    f"{(record.raw_ligand_rmse or 0.0):.6f}",
                    f"{(record.aligned_ligand_rmsd or 0.0):.6f}",
                    f"{record.training_seconds:.3f}",
                    record.node_count,
                    record.edge_count,
                    record.log_path.as_posix(),
                ]
            )


def write_markdown(
    path: Path,
    records: list[ExperimentRecord],
    means: list[SummaryRow],
    *,
    group_by: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = next(
        (
            row
            for row in means
            if row.model == "EGNN baseline"
            and (group_by != "model_schedule" or row.noise_schedule == "linear")
        ),
        None,
    )
    frame_backbone = next(
        (
            row
            for row in means
            if row.model == "heterogeneous frame-based backbone"
            and (group_by != "model_schedule" or row.noise_schedule == "linear")
        ),
        None,
    )

    lines = [
        "# Cross-Complex Comparison",
        "",
        "This summary aggregates the selected real-pair CPU runs already stored in `docs/training/`.",
        "Canonical mode is deterministic once the complex manifest, model set, and seed set are fixed.",
        "",
        "## Per-Complex Results",
        "",
        "| Complex | Model | Seed | Schedule | Best Loss | Final Loss | Raw Ligand RMSE | Aligned Ligand RMSD | Training Seconds |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| `{record.complex_id}` | {record.model} | `{record.seed}` | {record.noise_schedule} | "
            f"`{record.best_loss:.6f}` | `{record.final_loss:.6f}` | "
            f"`{(record.raw_ligand_rmse or 0.0):.6f}` | `{(record.aligned_ligand_rmsd or 0.0):.6f}` | "
            f"`{record.training_seconds:.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Means",
            "",
            "| Model | Schedule | Complexes | Runs | Mean Raw Ligand RMSE | Std Raw Ligand RMSE | Mean Aligned Ligand RMSD | Std Aligned Ligand RMSD | Success@2A | Success@5A | Mean Training Seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in means:
        lines.append(
            f"| {row.model} | {row.noise_schedule or '-'} | `{row.complexes}` | `{row.runs}` | "
            f"`{row.mean_raw_ligand_rmse:.6f}` | `{row.std_raw_ligand_rmse:.6f}` | "
            f"`{row.mean_aligned_ligand_rmsd:.6f}` | `{row.std_aligned_ligand_rmsd:.6f}` | "
            f"`{row.success_at_2a:.1f}%` | `{row.success_at_5a:.1f}%` | `{row.mean_training_seconds:.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    if baseline is not None and frame_backbone is not None:
        raw_reduction = percent_reduction(
            baseline.mean_raw_ligand_rmse,
            frame_backbone.mean_raw_ligand_rmse,
        )
        aligned_reduction = percent_reduction(
            baseline.mean_aligned_ligand_rmsd,
            frame_backbone.mean_aligned_ligand_rmsd,
        )
        speed_multiplier = frame_backbone.mean_training_seconds / baseline.mean_training_seconds
        lines.extend(
            [
                f"- Across the selected comparison runs, the heterogeneous frame-based backbone reduces mean raw ligand RMSE by `{raw_reduction:.1f}%` relative to the EGNN baseline.",
                f"- Across the same runs, the heterogeneous frame-based backbone reduces mean aligned ligand RMSD by `{aligned_reduction:.1f}%`.",
                f"- Success@2A changes from `{baseline.success_at_2a:.1f}%` to `{frame_backbone.success_at_2a:.1f}%`.",
                f"- The improvement is not free: the heterogeneous frame-based backbone is roughly `{speed_multiplier:.2f}x` slower in mean training time.",
                "- This is still a small-sample comparison, so it strengthens the project narrative but does not justify benchmark-scale claims.",
                "",
                "## Source Logs",
                "",
            ]
        )

    for record in records:
        lines.append(
            f"- `{record.complex_id}` {record.model} seed `{record.seed}` ({record.noise_schedule}): "
            f"`{record.log_path.as_posix()}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, means: list[SummaryRow], *, group_by: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Model & Schedule & Complexes & Runs & Mean Raw RMSE & Std Raw RMSE & Mean Aligned RMSD & Success@2\\AA{} & Success@5\\AA{} & Mean Train s \\\\",
        "\\midrule",
    ]
    for row in means:
        schedule_label = row.noise_schedule if group_by == "model_schedule" else "-"
        lines.append(
            f"{_latex_escape(row.model)} & {_latex_escape(schedule_label or '-')} & "
            f"{row.complexes} & {row.runs} & {row.mean_raw_ligand_rmse:.6f} & "
            f"{row.std_raw_ligand_rmse:.6f} & {row.mean_aligned_ligand_rmsd:.6f} & "
            f"{row.success_at_2a:.1f}\\% & {row.success_at_5a:.1f}\\% & {row.mean_training_seconds:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def percent_reduction(baseline: float, improved: float) -> float:
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - improved) / baseline


def main() -> int:
    args = build_parser().parse_args()
    manifest_complex_ids = None
    if args.manifest is not None:
        manifest_complex_ids = load_split_complex_ids(args.manifest)

    records = discover_records(args.log_glob)
    records = filter_records_by_manifest(records, manifest_complex_ids)
    records = select_records(records, models=args.models)
    if args.require_complete:
        assert_expected_combinations_present(
            records,
            manifest_complex_ids=manifest_complex_ids,
            expected_models=resolve_expected_models(
                models=args.models,
                expected_models=args.expected_models,
            ),
            expected_seeds=resolve_expected_seeds(args.expected_seeds),
            expected_schedules=resolve_expected_schedules(args.expected_schedules),
        )
    if not records:
        raise ValueError(f"No comparable experiment logs found for pattern {args.log_glob!r}.")
    means = grouped_means(records, group_by=args.group_by)
    write_csv(args.output_csv, records)
    write_markdown(args.output_markdown, records, means, group_by=args.group_by)
    if args.output_latex is not None:
        write_latex(args.output_latex, means, group_by=args.group_by)
    print(f"markdown_path={args.output_markdown}")
    print(f"csv_path={args.output_csv}")
    if args.output_latex is not None:
        print(f"latex_path={args.output_latex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
