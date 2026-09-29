from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kiral.evaluation_diff import (
    build_delta_rows,
    discover_selected_records,
    main,
    pair_records,
    summarize_pairs,
)
from kiral.evaluation_summary import ExperimentRecord


def _record(
    *,
    complex_id: str = "10gs",
    model: str = "heterogeneous frame-based backbone",
    seed: int = 42,
    noise_schedule: str = "cosine",
    best_loss: float = 0.1,
    final_loss: float = 0.2,
    raw_ligand_rmse: float = 1.5,
    aligned_ligand_rmsd: float = 1.2,
    training_seconds: float = 3.0,
    log_name: str = "a.md",
) -> ExperimentRecord:
    return ExperimentRecord(
        complex_id=complex_id,
        model=model,
        seed=seed,
        noise_schedule=noise_schedule,
        log_path=Path(log_name),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=1,
        edge_count=1,
        best_loss=best_loss,
        final_loss=final_loss,
        training_seconds=training_seconds,
        raw_ligand_rmse=raw_ligand_rmse,
        aligned_ligand_rmsd=aligned_ligand_rmsd,
    )


def _write_log(
    path: Path,
    *,
    command: str,
    graph_source: str = "real_pair",
    training_steps: int = 100,
    sample_steps: int = 25,
    best_loss: float = 0.05,
    final_loss: float = 0.10,
    training_seconds: float = 1.0,
    raw_ligand_rmse: float = 1.5,
    aligned_ligand_rmsd: float = 1.2,
    seed: int = 42,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                f"- Command: `{command}`",
                f"- Seed: `{seed}`",
                f"- Graph source: `{graph_source}`",
                f"- Training steps: `{training_steps}`",
                f"- Sample steps: `{sample_steps}`",
                "- Node count: `147`",
                "- Edge count: `2992`",
                f"- Final loss: `{final_loss:.6f}` at step `{training_steps}`",
                f"- Best loss: `{best_loss:.6f}`",
                f"- Training seconds: `{training_seconds:.3f}`",
                f"- Raw Ligand Rmse: `{raw_ligand_rmse:.6f}`",
                f"- Aligned Ligand Rmsd: `{aligned_ligand_rmsd:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_pair_records_reports_missing_pairs() -> None:
    left = [_record(log_name="left.md")]
    right = [_record(seed=43, log_name="right.md")]

    pairs, missing = pair_records(left, right)

    assert pairs == []
    assert {item.side for item in missing} == {"left", "right"}


def test_build_delta_rows_supports_run_and_complex_aggregation() -> None:
    pairs = [
        pair_records(
            [_record(seed=42, raw_ligand_rmse=1.0, aligned_ligand_rmsd=1.1, log_name="l1.md")],
            [_record(seed=42, raw_ligand_rmse=0.8, aligned_ligand_rmsd=0.9, log_name="r1.md")],
        )[0][0],
        pair_records(
            [_record(seed=43, raw_ligand_rmse=1.2, aligned_ligand_rmsd=1.3, log_name="l2.md")],
            [_record(seed=43, raw_ligand_rmse=1.1, aligned_ligand_rmsd=1.0, log_name="r2.md")],
        )[0][0],
    ]

    run_rows = build_delta_rows(pairs, aggregate_by="run")
    complex_rows = build_delta_rows(pairs, aggregate_by="complex")

    assert len(run_rows) == 2
    assert run_rows[0].seed_label == "42"
    assert len(complex_rows) == 1
    assert complex_rows[0].seed_label == "all"
    assert complex_rows[0].matched_runs == 2
    assert complex_rows[0].delta_aligned_ligand_rmsd == pytest.approx(-0.25)


def test_summarize_pairs_computes_success_rate_deltas() -> None:
    pairs = [
        pair_records(
            [_record(seed=42, aligned_ligand_rmsd=2.5, log_name="left_42.md")],
            [_record(seed=42, aligned_ligand_rmsd=1.5, log_name="right_42.md")],
        )[0][0],
        pair_records(
            [_record(seed=43, aligned_ligand_rmsd=1.0, log_name="left_43.md")],
            [_record(seed=43, aligned_ligand_rmsd=1.1, log_name="right_43.md")],
        )[0][0],
    ]

    summary = summarize_pairs(pairs)

    assert summary.left_success_at_2a == 50.0
    assert summary.right_success_at_2a == 100.0
    assert summary.delta_success_at_2a == 50.0


def test_discover_selected_records_filters_model_and_schedule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cosine = tmp_path / "10gs_cosine_log.md"
    linear = tmp_path / "10gs_linear_log.md"
    _write_log(
        cosine,
        command=(
            "uv run python -m kiral.train "
            "--frame-hetero-backbone --noise-schedule cosine "
            "--protein-path data/x/10gs/10gs_protein.pdb"
        ),
    )
    _write_log(
        linear,
        command=(
            "uv run python -m kiral.train "
            "--protein-path data/x/10gs/10gs_protein.pdb"
        ),
    )

    monkeypatch.chdir(tmp_path)
    records = discover_selected_records(
        "*_log.md",
        manifest_complex_ids=None,
        models=["heterogeneous frame-based backbone"],
        schedules=["cosine"],
    )

    assert [record.log_path.name for record in records] == ["10gs_cosine_log.md"]


def test_main_writes_diff_report_and_missing_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    _write_log(
        left_dir / "10gs_log.md",
        command=(
            "uv run python -m kiral.train "
            "--frame-hetero-backbone --noise-schedule cosine "
            "--protein-path data/x/10gs/10gs_protein.pdb"
        ),
        raw_ligand_rmse=1.4,
        aligned_ligand_rmsd=1.1,
        training_seconds=3.2,
    )
    _write_log(
        left_dir / "11gs_log.md",
        command=(
            "uv run python -m kiral.train "
            "--frame-hetero-backbone --noise-schedule cosine "
            "--protein-path data/x/11gs/11gs_protein.pdb"
        ),
        raw_ligand_rmse=1.8,
        aligned_ligand_rmsd=1.7,
        training_seconds=3.5,
    )
    _write_log(
        right_dir / "10gs_log.md",
        command=(
            "uv run python -m kiral.train "
            "--frame-hetero-backbone --noise-schedule cosine "
            "--protein-path data/x/10gs/10gs_protein.pdb"
        ),
        raw_ligand_rmse=1.1,
        aligned_ligand_rmsd=0.9,
        training_seconds=3.0,
    )

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("10gs\n11gs\n", encoding="utf-8")
    markdown_path = tmp_path / "comparison.md"
    csv_path = tmp_path / "comparison.csv"

    monkeypatch.chdir(tmp_path)
    result = main(
        [
            "--left-glob",
            "left/*_log.md",
            "--right-glob",
            "right/*_log.md",
            "--manifest",
            str(manifest),
            "--model",
            "heterogeneous frame-based backbone",
            "--schedule",
            "cosine",
            "--aggregate-by",
            "complex",
            "--output-markdown",
            str(markdown_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    assert result == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Compared runs: `1`" in markdown
    assert "Compared complexes: `1`" in markdown
    assert "| `10gs` | heterogeneous frame-based backbone | cosine | `all` | `1` |" in markdown
    assert "| right | `11gs` | heterogeneous frame-based backbone | `42` | cosine |" in markdown

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][0] == "10gs"
    assert rows[1][13] == "-0.300000"
    assert rows[-1][0] == "right"
