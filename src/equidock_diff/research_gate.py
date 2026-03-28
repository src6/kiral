"""Summarize a research tag against a control tag using hard-case gate metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    complex_id: str
    run_name: str
    raw_ligand_rmse: float
    aligned_ligand_rmsd: float
    training_seconds: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a completed research tag against a control tag"
    )
    parser.add_argument("--control-csv", type=Path, required=True, help="Control run_index.csv path")
    parser.add_argument("--candidate-csv", type=Path, required=True, help="Candidate run_index.csv path")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional markdown summary output path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional per-complex comparison CSV output path",
    )
    parser.add_argument(
        "--max-mean-raw-rmse-delta",
        type=float,
        default=None,
        help="Optional threshold: mean raw RMSE delta must be <= this value",
    )
    parser.add_argument(
        "--max-mean-aligned-rmsd-delta",
        type=float,
        default=None,
        help="Optional threshold: mean aligned RMSD delta must be <= this value",
    )
    parser.add_argument(
        "--min-aligned-improved",
        type=int,
        default=None,
        help="Optional threshold: at least this many complexes must improve on aligned RMSD",
    )
    return parser


def _load_best_by_complex(path: Path) -> dict[str, RunRecord]:
    best: dict[str, RunRecord] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aligned = row.get("aligned_ligand_rmsd", "").strip()
            raw = row.get("raw_ligand_rmse", "").strip()
            if not aligned or not raw:
                continue
            record = RunRecord(
                complex_id=row["complex_id"],
                run_name=row["run_name"],
                raw_ligand_rmse=float(raw),
                aligned_ligand_rmsd=float(aligned),
                training_seconds=(
                    None
                    if not row.get("training_seconds", "").strip()
                    else float(row["training_seconds"])
                ),
            )
            current = best.get(record.complex_id)
            if current is None or record.aligned_ligand_rmsd < current.aligned_ligand_rmsd:
                best[record.complex_id] = record
    return best


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def write_comparison_csv(
    path: Path,
    *,
    control: dict[str, RunRecord],
    candidate: dict[str, RunRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complex_ids = sorted(set(control) & set(candidate))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_id",
                "control_run",
                "candidate_run",
                "control_raw_ligand_rmse",
                "candidate_raw_ligand_rmse",
                "delta_raw_ligand_rmse",
                "control_aligned_ligand_rmsd",
                "candidate_aligned_ligand_rmsd",
                "delta_aligned_ligand_rmsd",
            ]
        )
        for complex_id in complex_ids:
            control_record = control[complex_id]
            candidate_record = candidate[complex_id]
            writer.writerow(
                [
                    complex_id,
                    control_record.run_name,
                    candidate_record.run_name,
                    _format_float(control_record.raw_ligand_rmse),
                    _format_float(candidate_record.raw_ligand_rmse),
                    _format_float(candidate_record.raw_ligand_rmse - control_record.raw_ligand_rmse),
                    _format_float(control_record.aligned_ligand_rmsd),
                    _format_float(candidate_record.aligned_ligand_rmsd),
                    _format_float(
                        candidate_record.aligned_ligand_rmsd - control_record.aligned_ligand_rmsd
                    ),
                ]
            )


def write_summary_markdown(
    path: Path,
    *,
    control_csv: Path,
    candidate_csv: Path,
    control: dict[str, RunRecord],
    candidate: dict[str, RunRecord],
    gate_lines: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complex_ids = sorted(set(control) & set(candidate))
    raw_deltas = [candidate[c].raw_ligand_rmse - control[c].raw_ligand_rmse for c in complex_ids]
    aligned_deltas = [candidate[c].aligned_ligand_rmsd - control[c].aligned_ligand_rmsd for c in complex_ids]
    improved = sum(delta < 0 for delta in aligned_deltas)
    lines = [
        "# Research Gate Summary",
        "",
        f"- Control CSV: `{control_csv}`",
        f"- Candidate CSV: `{candidate_csv}`",
        f"- Compared complexes: `{len(complex_ids)}`",
        f"- Mean raw RMSE delta: `{statistics.fmean(raw_deltas):.6f}`",
        f"- Mean aligned RMSD delta: `{statistics.fmean(aligned_deltas):.6f}`",
        f"- Complexes improved on aligned RMSD: `{improved}/{len(complex_ids)}`",
    ]
    if gate_lines:
        lines.extend(gate_lines)
    lines.extend(
        [
            "",
            "| Complex | Control Run | Candidate Run | Delta Raw RMSE | Delta Aligned RMSD |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for complex_id in complex_ids:
        control_record = control[complex_id]
        candidate_record = candidate[complex_id]
        lines.append(
            f"| `{complex_id}` | `{control_record.run_name}` | `{candidate_record.run_name}` | "
            f"`{candidate_record.raw_ligand_rmse - control_record.raw_ligand_rmse:.6f}` | "
            f"`{candidate_record.aligned_ligand_rmsd - control_record.aligned_ligand_rmsd:.6f}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    control = _load_best_by_complex(args.control_csv)
    candidate = _load_best_by_complex(args.candidate_csv)
    overlap = sorted(set(control) & set(candidate))
    if not overlap:
        raise ValueError("No overlapping completed complex records were found between the two CSV files.")
    raw_deltas = [candidate[c].raw_ligand_rmse - control[c].raw_ligand_rmse for c in overlap]
    aligned_deltas = [candidate[c].aligned_ligand_rmsd - control[c].aligned_ligand_rmsd for c in overlap]
    improved = sum(delta < 0 for delta in aligned_deltas)
    mean_raw_delta = statistics.fmean(raw_deltas)
    mean_aligned_delta = statistics.fmean(aligned_deltas)
    threshold_results: list[tuple[str, bool, str]] = []
    if args.max_mean_raw_rmse_delta is not None:
        threshold_results.append(
            (
                "max_mean_raw_rmse_delta",
                mean_raw_delta <= args.max_mean_raw_rmse_delta,
                f"{mean_raw_delta:.6f} <= {args.max_mean_raw_rmse_delta:.6f}",
            )
        )
    if args.max_mean_aligned_rmsd_delta is not None:
        threshold_results.append(
            (
                "max_mean_aligned_rmsd_delta",
                mean_aligned_delta <= args.max_mean_aligned_rmsd_delta,
                f"{mean_aligned_delta:.6f} <= {args.max_mean_aligned_rmsd_delta:.6f}",
            )
        )
    if args.min_aligned_improved is not None:
        threshold_results.append(
            (
                "min_aligned_improved",
                improved >= args.min_aligned_improved,
                f"{improved} >= {args.min_aligned_improved}",
            )
        )
    gate_pass = all(result for _, result, _ in threshold_results) if threshold_results else None
    if args.output_csv is not None:
        write_comparison_csv(args.output_csv, control=control, candidate=candidate)
    if args.output_markdown is not None:
        gate_lines = None
        if gate_pass is not None:
            gate_lines = [f"- Gate pass: `{'yes' if gate_pass else 'no'}`"]
            gate_lines.extend(
                [
                    f"- `{name}`: `{'pass' if result else 'fail'}` ({detail})"
                    for name, result, detail in threshold_results
                ]
            )
        write_summary_markdown(
            args.output_markdown,
            control_csv=args.control_csv,
            candidate_csv=args.candidate_csv,
            control=control,
            candidate=candidate,
            gate_lines=gate_lines,
        )
    print(f"compared_complexes={len(overlap)}")
    print(f"mean_raw_ligand_rmse_delta={mean_raw_delta:.6f}")
    print(f"mean_aligned_ligand_rmsd_delta={mean_aligned_delta:.6f}")
    print(f"aligned_improved_complexes={improved}/{len(overlap)}")
    if gate_pass is not None:
        print(f"gate_pass={'yes' if gate_pass else 'no'}")
        for name, result, detail in threshold_results:
            print(f"gate_check_{name}={'pass' if result else 'fail'} ({detail})")
    if args.output_csv is not None:
        print(f"comparison_csv={args.output_csv}")
    if args.output_markdown is not None:
        print(f"summary_markdown={args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
