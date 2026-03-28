"""Aggregate sampler diagnostics JSON outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from equidock_diff.data.io import load_split_complex_ids
from equidock_diff.evaluation_summary import parse_experiment_log


@dataclass(frozen=True)
class SamplerRunSummary:
    diagnostics_path: Path
    complex_id: str
    seed: int
    sample_steps: int
    sample_time_power: float
    sample_score_clip: float
    sample_position_clip: float
    mean_late_score_norm_before_clip: float
    max_late_score_norm_before_clip: float
    mean_late_score_norm_after_clip: float
    max_late_score_norm_after_clip: float
    mean_late_clipped_fraction: float
    mean_late_center_displacement: float
    mean_late_ligand_radius: float
    raw_ligand_rmse: float | None
    aligned_ligand_rmsd: float | None
    training_seconds: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-step sampler diagnostics JSON")
    parser.add_argument("--glob", default=None, help="Single diagnostics glob to summarize")
    parser.add_argument("--left-glob", default=None, help="Reference diagnostics glob")
    parser.add_argument("--right-glob", default=None, help="Candidate diagnostics glob")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional complex manifest filter")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV output path")
    parser.add_argument("--output-markdown", type=Path, required=True, help="Markdown output path")
    return parser


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _load_paths(pattern: str) -> list[Path]:
    return sorted(Path(item) for item in glob.glob(pattern))


def _resolve_log_path(payload: dict[str, object], diagnostics_path: Path) -> Path | None:
    experiment_log = payload.get("experiment_log")
    if isinstance(experiment_log, str) and experiment_log:
        return Path(experiment_log)
    candidate = diagnostics_path.with_suffix(".md")
    return candidate if candidate.exists() else None


def load_summary(path: Path) -> SamplerRunSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = list(payload.get("steps", []))
    if not steps:
        raise ValueError(f"Sampler diagnostics {path} contains no step records.")
    late_count = max(len(steps) // 5, 1)
    late_steps = steps[-late_count:]
    log_path = _resolve_log_path(payload, path)
    parsed_log = parse_experiment_log(log_path) if log_path is not None and log_path.exists() else None
    complex_id = str(payload.get("complex_id") or (parsed_log.complex_id if parsed_log is not None else path.name.split("_", 1)[0]))
    return SamplerRunSummary(
        diagnostics_path=path,
        complex_id=complex_id,
        seed=int(payload.get("seed", parsed_log.seed if parsed_log is not None else 0)),
        sample_steps=int(payload.get("sample_steps", 0)),
        sample_time_power=float(payload.get("sample_time_power", 1.0)),
        sample_score_clip=float(payload.get("sample_score_clip", 0.0)),
        sample_position_clip=float(payload.get("sample_position_clip", 0.0)),
        mean_late_score_norm_before_clip=_mean([float(step["mean_score_norm_before_clip"]) for step in late_steps]),
        max_late_score_norm_before_clip=max(float(step["max_score_norm_before_clip"]) for step in late_steps),
        mean_late_score_norm_after_clip=_mean([float(step["mean_score_norm_after_clip"]) for step in late_steps]),
        max_late_score_norm_after_clip=max(float(step["max_score_norm_after_clip"]) for step in late_steps),
        mean_late_clipped_fraction=_mean([float(step["clipped_coordinate_fraction"]) for step in late_steps]),
        mean_late_center_displacement=_mean([float(step["ligand_center_displacement"]) for step in late_steps]),
        mean_late_ligand_radius=_mean([float(step["ligand_radius"]) for step in late_steps]),
        raw_ligand_rmse=parsed_log.raw_ligand_rmse if parsed_log is not None else None,
        aligned_ligand_rmsd=parsed_log.aligned_ligand_rmsd if parsed_log is not None else None,
        training_seconds=parsed_log.training_seconds if parsed_log is not None else None,
    )


def load_summaries(pattern: str, manifest: list[str] | None) -> list[SamplerRunSummary]:
    summaries = [load_summary(path) for path in _load_paths(pattern)]
    if manifest is not None:
        allowed = set(manifest)
        summaries = [summary for summary in summaries if summary.complex_id in allowed]
    return sorted(summaries, key=lambda item: (item.complex_id, item.seed, item.diagnostics_path.as_posix()))


def write_single_csv(path: Path, rows: list[SamplerRunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_id",
                "seed",
                "sample_steps",
                "sample_time_power",
                "sample_score_clip",
                "sample_position_clip",
                "mean_late_score_norm_before_clip",
                "max_late_score_norm_before_clip",
                "mean_late_score_norm_after_clip",
                "max_late_score_norm_after_clip",
                "mean_late_clipped_fraction",
                "mean_late_center_displacement",
                "mean_late_ligand_radius",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.complex_id,
                    row.seed,
                    row.sample_steps,
                    f"{row.sample_time_power:.3f}",
                    f"{row.sample_score_clip:.3f}",
                    f"{row.sample_position_clip:.3f}",
                    f"{row.mean_late_score_norm_before_clip:.6f}",
                    f"{row.max_late_score_norm_before_clip:.6f}",
                    f"{row.mean_late_score_norm_after_clip:.6f}",
                    f"{row.max_late_score_norm_after_clip:.6f}",
                    f"{row.mean_late_clipped_fraction:.6f}",
                    f"{row.mean_late_center_displacement:.6f}",
                    f"{row.mean_late_ligand_radius:.6f}",
                    "" if row.raw_ligand_rmse is None else f"{row.raw_ligand_rmse:.6f}",
                    "" if row.aligned_ligand_rmsd is None else f"{row.aligned_ligand_rmsd:.6f}",
                    "" if row.training_seconds is None else f"{row.training_seconds:.3f}",
                ]
            )


def write_single_markdown(path: Path, rows: list[SamplerRunSummary], pattern: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sampler Diagnostics",
        "",
        f"- Diagnostics glob: `{pattern}`",
        f"- Runs: `{len(rows)}`",
        f"- Complexes: `{len({row.complex_id for row in rows})}`",
        "",
        "| Complex | Seed | Time Power | Late Score Norm (Before) | Late Score Norm (After) | Late Clip Fraction | Late Center Disp. | Late Ligand Radius | Raw RMSE | Aligned RMSD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.complex_id}` | `{row.seed}` | `{row.sample_time_power:.3f}` | "
            f"`{row.mean_late_score_norm_before_clip:.6f}` | `{row.mean_late_score_norm_after_clip:.6f}` | "
            f"`{row.mean_late_clipped_fraction:.6f}` | `{row.mean_late_center_displacement:.6f}` | "
            f"`{row.mean_late_ligand_radius:.6f}` | "
            f"`{0.0 if row.raw_ligand_rmse is None else row.raw_ligand_rmse:.6f}` | "
            f"`{0.0 if row.aligned_ligand_rmsd is None else row.aligned_ligand_rmsd:.6f}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_key(row: SamplerRunSummary) -> tuple[str, int]:
    return (row.complex_id, row.seed)


def write_comparison_outputs(
    markdown_path: Path,
    csv_path: Path,
    *,
    left_glob: str,
    right_glob: str,
    left_rows: list[SamplerRunSummary],
    right_rows: list[SamplerRunSummary],
) -> None:
    left_by_key = {_pair_key(row): row for row in left_rows}
    right_by_key = {_pair_key(row): row for row in right_rows}
    keys = sorted(set(left_by_key) & set(right_by_key))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_id",
                "seed",
                "delta_mean_late_score_norm_before_clip",
                "delta_mean_late_score_norm_after_clip",
                "delta_mean_late_clipped_fraction",
                "delta_mean_late_center_displacement",
                "delta_mean_late_ligand_radius",
                "delta_raw_ligand_rmse",
                "delta_aligned_ligand_rmsd",
            ]
        )
        for key in keys:
            left = left_by_key[key]
            right = right_by_key[key]
            writer.writerow(
                [
                    left.complex_id,
                    left.seed,
                    f"{right.mean_late_score_norm_before_clip - left.mean_late_score_norm_before_clip:.6f}",
                    f"{right.mean_late_score_norm_after_clip - left.mean_late_score_norm_after_clip:.6f}",
                    f"{right.mean_late_clipped_fraction - left.mean_late_clipped_fraction:.6f}",
                    f"{right.mean_late_center_displacement - left.mean_late_center_displacement:.6f}",
                    f"{right.mean_late_ligand_radius - left.mean_late_ligand_radius:.6f}",
                    f"{(right.raw_ligand_rmse or 0.0) - (left.raw_ligand_rmse or 0.0):.6f}",
                    f"{(right.aligned_ligand_rmsd or 0.0) - (left.aligned_ligand_rmsd or 0.0):.6f}",
                ]
            )

    aggregate_deltas = []
    for key in keys:
        left = left_by_key[key]
        right = right_by_key[key]
        aggregate_deltas.append(
            (
                right.mean_late_score_norm_before_clip - left.mean_late_score_norm_before_clip,
                right.mean_late_clipped_fraction - left.mean_late_clipped_fraction,
                (right.raw_ligand_rmse or 0.0) - (left.raw_ligand_rmse or 0.0),
                (right.aligned_ligand_rmsd or 0.0) - (left.aligned_ligand_rmsd or 0.0),
            )
        )
    markdown_lines = [
        "# Sampler Diagnostics Comparison",
        "",
        f"- Left diagnostics: `{left_glob}`",
        f"- Right diagnostics: `{right_glob}`",
        f"- Matched runs: `{len(keys)}`",
        "",
    ]
    if aggregate_deltas:
        markdown_lines.extend(
            [
                "## Aggregate Deltas",
                "",
                f"- Mean late score norm before clip delta: `{_mean([row[0] for row in aggregate_deltas]):+.6f}`",
                f"- Mean late clipped fraction delta: `{_mean([row[1] for row in aggregate_deltas]):+.6f}`",
                f"- Mean raw RMSE delta: `{_mean([row[2] for row in aggregate_deltas]):+.6f}`",
                f"- Mean aligned RMSD delta: `{_mean([row[3] for row in aggregate_deltas]):+.6f}`",
                "",
                "## Paired Rows",
                "",
                "| Complex | Seed | Delta Late Score Norm | Delta Clip Fraction | Delta Raw RMSE | Delta Aligned RMSD |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in keys:
            left = left_by_key[key]
            right = right_by_key[key]
            markdown_lines.append(
                f"| `{left.complex_id}` | `{left.seed}` | "
                f"`{right.mean_late_score_norm_before_clip - left.mean_late_score_norm_before_clip:+.6f}` | "
                f"`{right.mean_late_clipped_fraction - left.mean_late_clipped_fraction:+.6f}` | "
                f"`{(right.raw_ligand_rmse or 0.0) - (left.raw_ligand_rmse or 0.0):+.6f}` | "
                f"`{(right.aligned_ligand_rmsd or 0.0) - (left.aligned_ligand_rmsd or 0.0):+.6f}` |"
            )
    else:
        markdown_lines.append("No matched diagnostics pairs were found.")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.glob is None and (args.left_glob is None or args.right_glob is None):
        raise ValueError("Pass either --glob or both --left-glob and --right-glob.")
    manifest = load_split_complex_ids(args.manifest) if args.manifest is not None else None
    if args.glob is not None:
        rows = load_summaries(args.glob, manifest)
        write_single_csv(args.output_csv, rows)
        write_single_markdown(args.output_markdown, rows, args.glob)
    else:
        left_rows = load_summaries(args.left_glob, manifest)
        right_rows = load_summaries(args.right_glob, manifest)
        write_comparison_outputs(
            args.output_markdown,
            args.output_csv,
            left_glob=args.left_glob,
            right_glob=args.right_glob,
            left_rows=left_rows,
            right_rows=right_rows,
        )
    print(f"csv_path={args.output_csv}")
    print(f"markdown_path={args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
