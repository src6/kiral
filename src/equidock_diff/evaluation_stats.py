"""Run paired significance tests on experiment-diff CSV outputs."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


METRIC_CHOICES = ("aligned_ligand_rmsd", "raw_ligand_rmse")


@dataclass(frozen=True)
class ComparisonRow:
    complex_id: str
    model: str
    noise_schedule: str
    seed: str
    matched_runs: int
    left_value: float
    right_value: float
    delta_value: float


@dataclass(frozen=True)
class SignificanceResult:
    comparison_csv: Path
    metric: str
    alternative: str
    observations: int
    p_value: float
    mean_delta: float
    median_delta: float
    mean_delta_ci_low: float
    mean_delta_ci_high: float
    median_delta_ci_low: float
    median_delta_ci_high: float
    wins: int
    losses: int
    ties: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired significance tests on experiment comparison CSV outputs"
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        required=True,
        help="CSV output produced by python -m equidock_diff.evaluation_diff",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        required=True,
        help="Metric column to test",
    )
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "less", "greater"),
        default="less",
        help="Alternative hypothesis for the paired permutation test",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional markdown report output path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV summary output path",
    )
    return parser


def load_comparison_rows(path: Path) -> list[ComparisonRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Comparison CSV {path} is empty.") from exc

        if "complex_id" not in header:
            raise ValueError(f"Comparison CSV {path} is missing the complex_id header.")
        index = {name: idx for idx, name in enumerate(header)}
        rows: list[ComparisonRow] = []
        for raw_row in reader:
            if not raw_row or all(cell == "" for cell in raw_row):
                break
            complex_id = raw_row[index["complex_id"]]
            if complex_id == "" or complex_id == "missing_side":
                break
            rows.append(
                ComparisonRow(
                    complex_id=complex_id,
                    model=raw_row[index["model"]],
                    noise_schedule=raw_row[index["noise_schedule"]],
                    seed=raw_row[index["seed"]],
                    matched_runs=int(raw_row[index["matched_runs"]]),
                    left_value=float(raw_row[index["left_aligned_ligand_rmsd"]])
                    if "left_aligned_ligand_rmsd" in index
                    else float(raw_row[index["left_raw_ligand_rmse"]]),
                    right_value=float(raw_row[index["right_aligned_ligand_rmsd"]])
                    if "right_aligned_ligand_rmsd" in index
                    else float(raw_row[index["right_raw_ligand_rmse"]]),
                    delta_value=float(raw_row[index["delta_aligned_ligand_rmsd"]])
                    if "delta_aligned_ligand_rmsd" in index
                    else float(raw_row[index["delta_raw_ligand_rmse"]]),
                )
            )
    if not rows:
        raise ValueError(f"Comparison CSV {path} contains no paired rows.")
    return rows


def _metric_columns(metric: str) -> tuple[str, str, str]:
    return (
        f"left_{metric}",
        f"right_{metric}",
        f"delta_{metric}",
    )


def load_metric_rows(path: Path, *, metric: str) -> list[ComparisonRow]:
    left_col, right_col, delta_col = _metric_columns(metric)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Comparison CSV {path} is empty.") from exc

        required = {
            "complex_id",
            "model",
            "noise_schedule",
            "seed",
            "matched_runs",
            left_col,
            right_col,
            delta_col,
        }
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(
                f"Comparison CSV {path} is missing required columns: {', '.join(missing)}"
            )

        index = {name: idx for idx, name in enumerate(header)}
        rows: list[ComparisonRow] = []
        for raw_row in reader:
            if not raw_row or all(cell == "" for cell in raw_row):
                break
            complex_id = raw_row[index["complex_id"]]
            if complex_id == "" or complex_id == "missing_side":
                break
            rows.append(
                ComparisonRow(
                    complex_id=complex_id,
                    model=raw_row[index["model"]],
                    noise_schedule=raw_row[index["noise_schedule"]],
                    seed=raw_row[index["seed"]],
                    matched_runs=int(raw_row[index["matched_runs"]]),
                    left_value=float(raw_row[index[left_col]]),
                    right_value=float(raw_row[index[right_col]]),
                    delta_value=float(raw_row[index[delta_col]]),
                )
            )
    if not rows:
        raise ValueError(f"Comparison CSV {path} contains no paired rows.")
    return rows


def filter_seed_all_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    filtered = [row for row in rows if row.seed == "all"]
    if not filtered:
        raise ValueError("Comparison CSV does not contain any per-complex aggregate rows with seed=all.")
    return filtered


def bootstrap_confidence_interval(
    values: list[float],
    *,
    statistic: str,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("At least one value is required for bootstrap confidence intervals.")
    rng = random.Random(seed)
    estimates: list[float] = []
    sample_size = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        if statistic == "mean":
            estimates.append(statistics.fmean(sample))
        elif statistic == "median":
            estimates.append(statistics.median(sample))
        else:
            raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(int(tail * iterations), 0)
    upper_index = min(int((1.0 - tail) * iterations), iterations - 1)
    return estimates[lower_index], estimates[upper_index]


def exact_sign_permutation_p_value(
    deltas: list[float],
    *,
    alternative: str,
) -> float:
    if not deltas:
        raise ValueError("At least one paired delta is required for significance testing.")

    observed = statistics.fmean(deltas)
    abs_deltas = [abs(delta) for delta in deltas]
    permutations = 1 << len(abs_deltas)
    extreme = 0

    for signs in itertools.product((-1.0, 1.0), repeat=len(abs_deltas)):
        statistic = statistics.fmean(
            sign * magnitude for sign, magnitude in zip(signs, abs_deltas, strict=True)
        )
        if alternative == "less":
            if statistic <= observed:
                extreme += 1
        elif alternative == "greater":
            if statistic >= observed:
                extreme += 1
        else:
            if abs(statistic) >= abs(observed):
                extreme += 1

    return extreme / permutations


def summarize_deltas(
    path: Path,
    *,
    metric: str,
    alternative: str,
) -> SignificanceResult:
    rows = filter_seed_all_rows(load_metric_rows(path, metric=metric))
    deltas = [row.delta_value for row in rows]
    wins = sum(1 for delta in deltas if delta < 0.0)
    losses = sum(1 for delta in deltas if delta > 0.0)
    ties = len(deltas) - wins - losses
    return SignificanceResult(
        comparison_csv=path,
        metric=metric,
        alternative=alternative,
        observations=len(deltas),
        p_value=exact_sign_permutation_p_value(deltas, alternative=alternative),
        mean_delta=statistics.fmean(deltas),
        median_delta=statistics.median(deltas),
        mean_delta_ci_low=bootstrap_confidence_interval(deltas, statistic="mean")[0],
        mean_delta_ci_high=bootstrap_confidence_interval(deltas, statistic="mean")[1],
        median_delta_ci_low=bootstrap_confidence_interval(deltas, statistic="median")[0],
        median_delta_ci_high=bootstrap_confidence_interval(deltas, statistic="median")[1],
        wins=wins,
        losses=losses,
        ties=ties,
    )


def write_markdown(path: Path, result: SignificanceResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Significance Test",
        "",
        f"- Comparison CSV: `{result.comparison_csv}`",
        f"- Metric: `{result.metric}`",
        f"- Alternative hypothesis: `{result.alternative}`",
        f"- Paired observations: `{result.observations}`",
        "",
        "## Result",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| p-value | `{result.p_value:.6f}` |",
        f"| Mean delta | `{result.mean_delta:+.6f}` |",
        f"| Mean delta 95% CI | `[{result.mean_delta_ci_low:+.6f}, {result.mean_delta_ci_high:+.6f}]` |",
        f"| Median delta | `{result.median_delta:+.6f}` |",
        f"| Median delta 95% CI | `[{result.median_delta_ci_low:+.6f}, {result.median_delta_ci_high:+.6f}]` |",
        f"| Wins | `{result.wins}` |",
        f"| Losses | `{result.losses}` |",
        f"| Ties | `{result.ties}` |",
        "",
        "Negative deltas mean the candidate outperformed the control for lower-is-better metrics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, result: SignificanceResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison_csv",
                "metric",
                "alternative",
                "observations",
                "p_value",
                "mean_delta",
                "mean_delta_ci_low",
                "mean_delta_ci_high",
                "median_delta",
                "median_delta_ci_low",
                "median_delta_ci_high",
                "wins",
                "losses",
                "ties",
            ]
        )
        writer.writerow(
            [
                result.comparison_csv.as_posix(),
                result.metric,
                result.alternative,
                result.observations,
                f"{result.p_value:.6f}",
                f"{result.mean_delta:+.6f}",
                f"{result.mean_delta_ci_low:+.6f}",
                f"{result.mean_delta_ci_high:+.6f}",
                f"{result.median_delta:+.6f}",
                f"{result.median_delta_ci_low:+.6f}",
                f"{result.median_delta_ci_high:+.6f}",
                result.wins,
                result.losses,
                result.ties,
            ]
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize_deltas(
        args.comparison_csv,
        metric=args.metric,
        alternative=args.alternative,
    )
    if args.output_markdown is not None:
        write_markdown(args.output_markdown, result)
        print(f"markdown_path={args.output_markdown}")
    if args.output_csv is not None:
        write_csv(args.output_csv, result)
        print(f"csv_path={args.output_csv}")
    print(f"metric={result.metric}")
    print(f"observations={result.observations}")
    print(f"p_value={result.p_value:.6f}")
    print(f"mean_delta={result.mean_delta:+.6f}")
    print(f"mean_delta_ci=[{result.mean_delta_ci_low:+.6f},{result.mean_delta_ci_high:+.6f}]")
    print(f"median_delta={result.median_delta:+.6f}")
    print(f"median_delta_ci=[{result.median_delta_ci_low:+.6f},{result.median_delta_ci_high:+.6f}]")
    print(f"wins={result.wins}")
    print(f"losses={result.losses}")
    print(f"ties={result.ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
