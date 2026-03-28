from __future__ import annotations

import csv
from pathlib import Path

import pytest

from equidock_diff.evaluation_stats import (
    exact_sign_permutation_p_value,
    filter_seed_all_rows,
    load_metric_rows,
    main,
    summarize_deltas,
)


def _write_comparison_csv(path: Path, rows: list[dict[str, str]]) -> None:
    header = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])
        writer.writerow([])
        writer.writerow(["missing_side", "complex_id", "model", "seed", "noise_schedule"])


def _row(
    *,
    complex_id: str = "10gs",
    seed: str = "all",
    left_aligned: str = "1.200000",
    right_aligned: str = "1.000000",
    delta_aligned: str = "-0.200000",
    left_raw: str = "1.500000",
    right_raw: str = "1.300000",
    delta_raw: str = "-0.200000",
) -> dict[str, str]:
    return {
        "complex_id": complex_id,
        "model": "heterogeneous frame-based backbone",
        "noise_schedule": "cosine",
        "seed": seed,
        "matched_runs": "3",
        "left_best_loss": "0.010000",
        "right_best_loss": "0.008000",
        "delta_best_loss": "-0.002000",
        "left_final_loss": "0.020000",
        "right_final_loss": "0.010000",
        "delta_final_loss": "-0.010000",
        "left_raw_ligand_rmse": left_raw,
        "right_raw_ligand_rmse": right_raw,
        "delta_raw_ligand_rmse": delta_raw,
        "left_aligned_ligand_rmsd": left_aligned,
        "right_aligned_ligand_rmsd": right_aligned,
        "delta_aligned_ligand_rmsd": delta_aligned,
        "left_training_seconds": "3.000",
        "right_training_seconds": "6.000",
        "delta_training_seconds": "3.000",
    }


def test_load_metric_rows_stops_before_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_comparison_csv(path, [_row(complex_id="10gs"), _row(complex_id="11gs")])

    rows = load_metric_rows(path, metric="aligned_ligand_rmsd")

    assert [row.complex_id for row in rows] == ["10gs", "11gs"]
    assert rows[0].delta_value == -0.2


def test_filter_seed_all_rows_keeps_only_complex_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_comparison_csv(
        path,
        [
            _row(complex_id="10gs", seed="all"),
            _row(complex_id="10gs", seed="42"),
        ],
    )

    rows = filter_seed_all_rows(load_metric_rows(path, metric="aligned_ligand_rmsd"))

    assert len(rows) == 1
    assert rows[0].seed == "all"


def test_exact_sign_permutation_p_value_supports_less_and_two_sided() -> None:
    deltas = [-1.0, -2.0, -3.0]

    assert exact_sign_permutation_p_value(deltas, alternative="less") == pytest.approx(0.125)
    assert exact_sign_permutation_p_value(deltas, alternative="two-sided") == pytest.approx(0.25)


def test_summarize_deltas_reports_effect_size_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_comparison_csv(
        path,
        [
            _row(complex_id="10gs", delta_aligned="-0.200000"),
            _row(complex_id="11gs", delta_aligned="0.100000", left_aligned="1.0", right_aligned="1.1"),
            _row(complex_id="12gs", delta_aligned="0.000000", left_aligned="1.0", right_aligned="1.0"),
        ],
    )

    result = summarize_deltas(path, metric="aligned_ligand_rmsd", alternative="less")

    assert result.observations == 3
    assert result.mean_delta == pytest.approx(-0.0333333333)
    assert result.median_delta == pytest.approx(0.0)
    assert result.wins == 1
    assert result.losses == 1
    assert result.ties == 1


def test_main_writes_stats_outputs_for_significant_pattern(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "comparison.csv"
    _write_comparison_csv(
        path,
        [
            _row(complex_id="10gs", delta_aligned="-0.5", left_aligned="1.2", right_aligned="0.7"),
            _row(complex_id="11gs", delta_aligned="-0.4", left_aligned="1.4", right_aligned="1.0"),
            _row(complex_id="12gs", delta_aligned="-0.3", left_aligned="1.1", right_aligned="0.8"),
            _row(complex_id="13gs", delta_aligned="-0.2", left_aligned="1.0", right_aligned="0.8"),
        ],
    )
    markdown_path = tmp_path / "stats.md"
    csv_path = tmp_path / "stats.csv"

    monkeypatch.chdir(tmp_path)
    result = main(
        [
            "--comparison-csv",
            str(path),
            "--metric",
            "aligned_ligand_rmsd",
            "--alternative",
            "less",
            "--output-markdown",
            str(markdown_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    assert result == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Comparison CSV" in markdown
    assert "| p-value | `0.062500` |" in markdown
    assert "| Wins | `4` |" in markdown

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][1] == "aligned_ligand_rmsd"
    assert rows[1][4] == "0.062500"


def test_main_handles_non_significant_mixed_pattern(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    _write_comparison_csv(
        path,
        [
            _row(complex_id="10gs", delta_aligned="-0.2"),
            _row(complex_id="11gs", delta_aligned="0.2", left_aligned="1.0", right_aligned="1.2"),
            _row(complex_id="12gs", delta_aligned="-0.1", left_aligned="1.1", right_aligned="1.0"),
            _row(complex_id="13gs", delta_aligned="0.1", left_aligned="0.9", right_aligned="1.0"),
        ],
    )

    result = summarize_deltas(path, metric="aligned_ligand_rmsd", alternative="less")

    assert result.p_value == pytest.approx(0.625)


def test_load_metric_rows_rejects_empty_or_malformed_csv(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("foo,bar\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_metric_rows(empty, metric="aligned_ligand_rmsd")
    with pytest.raises(ValueError, match="missing required columns"):
        load_metric_rows(malformed, metric="aligned_ligand_rmsd")
