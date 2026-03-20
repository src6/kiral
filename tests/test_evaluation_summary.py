from __future__ import annotations

from pathlib import Path

from equidock_diff.evaluation_summary import (
    ExperimentRecord,
    discover_records,
    grouped_means,
    infer_model,
    parse_experiment_log,
    percent_reduction,
    select_records,
)


def test_infer_model_detects_primary_variants() -> None:
    assert infer_model("uv run python -m equidock_diff.train") == "EGNN baseline"
    assert (
        infer_model("uv run python -m equidock_diff.train --hetgnn-backbone")
        == "heterogeneous frame-based backbone"
    )
    assert (
        infer_model("uv run python -m equidock_diff.train --complete-frame --ligand-global-node")
        == "EGNN + complete frames + ligand context"
    )


def test_parse_experiment_log_extracts_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "10gs_hetgnn_log.md"
    log_path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                "- Command: `uv run python -m equidock_diff.train --hetgnn-backbone --protein-path data/x/10gs/10gs_protein.pdb`",
                "- Graph source: `real_pair`",
                "- Training steps: `100`",
                "- Sample steps: `25`",
                "- Node count: `147`",
                "- Edge count: `2992`",
                "- Final loss: `0.061780` at step `100`",
                "- Best loss: `0.039261`",
                "- Training seconds: `3.665`",
                "- Raw Ligand Rmse: `1.637888`",
                "- Aligned Ligand Rmsd: `1.420224`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = parse_experiment_log(log_path)

    assert record.complex_id == "10gs"
    assert record.model == "heterogeneous frame-based backbone"
    assert record.graph_source == "real_pair"
    assert record.final_loss == 0.061780
    assert record.best_loss == 0.039261
    assert record.raw_ligand_rmse == 1.637888
    assert record.aligned_ligand_rmsd == 1.420224


def test_select_records_prefers_non_rerun_logs() -> None:
    preferred = ExperimentRecord(
        complex_id="10gs",
        model="heterogeneous frame-based backbone",
        log_path=Path("docs/training/10gs_hetgnn_compare_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=147,
        edge_count=2992,
        best_loss=0.01,
        final_loss=0.02,
        training_seconds=3.0,
        raw_ligand_rmse=1.6,
        aligned_ligand_rmsd=1.4,
    )
    rerun = ExperimentRecord(
        complex_id="10gs",
        model="heterogeneous frame-based backbone",
        log_path=Path("docs/training/10gs_hetgnn_compare_rerun_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=147,
        edge_count=2992,
        best_loss=0.01,
        final_loss=0.02,
        training_seconds=3.0,
        raw_ligand_rmse=1.6,
        aligned_ligand_rmsd=1.4,
    )

    records = select_records([rerun, preferred], models="primary")

    assert records == [preferred]


def test_select_records_prefers_full_runs_over_repro_checks() -> None:
    repro = ExperimentRecord(
        complex_id="10gs",
        model="EGNN baseline",
        log_path=Path("docs/training/10gs_repro_check_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=3,
        sample_steps=5,
        node_count=147,
        edge_count=2992,
        best_loss=1.0,
        final_loss=1.1,
        training_seconds=0.06,
        raw_ligand_rmse=2.8,
        aligned_ligand_rmsd=2.7,
    )
    full = ExperimentRecord(
        complex_id="10gs",
        model="EGNN baseline",
        log_path=Path("docs/training/10gs_hetgnn_compare_baseline_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=147,
        edge_count=2992,
        best_loss=0.09,
        final_loss=0.69,
        training_seconds=1.529,
        raw_ligand_rmse=3.6,
        aligned_ligand_rmsd=2.6,
    )

    records = select_records([repro, full], models="primary")

    assert records == [full]


def test_grouped_means_and_percent_reduction() -> None:
    records = [
        ExperimentRecord(
            complex_id="10gs",
            model="EGNN baseline",
            log_path=Path("a"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.1,
            final_loss=0.2,
            training_seconds=2.0,
            raw_ligand_rmse=3.0,
            aligned_ligand_rmsd=2.0,
        ),
        ExperimentRecord(
            complex_id="11gs",
            model="heterogeneous frame-based backbone",
            log_path=Path("b"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.05,
            final_loss=0.1,
            training_seconds=4.0,
            raw_ligand_rmse=1.5,
            aligned_ligand_rmsd=1.0,
        ),
    ]

    means = grouped_means(records)

    baseline = next(row for row in means if row["model"] == "EGNN baseline")
    frame_backbone = next(
        row for row in means if row["model"] == "heterogeneous frame-based backbone"
    )

    assert float(baseline["mean_raw_ligand_rmse"]) == 3.0
    assert float(frame_backbone["mean_aligned_ligand_rmsd"]) == 1.0
    assert percent_reduction(3.0, 1.5) == 50.0


def test_discover_records_filters_non_real_pair_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_log = tmp_path / "10gs_baseline_log.md"
    real_log.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                "- Command: `uv run python -m equidock_diff.train --protein-path data/x/10gs/10gs_protein.pdb`",
                "- Graph source: `real_pair`",
                "- Training steps: `100`",
                "- Sample steps: `25`",
                "- Node count: `147`",
                "- Edge count: `2992`",
                "- Final loss: `0.100000` at step `100`",
                "- Best loss: `0.050000`",
                "- Training seconds: `1.000`",
                "- Raw Ligand Rmse: `3.000000`",
                "- Aligned Ligand Rmsd: `2.000000`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_log = tmp_path / "dataset_mode_log.md"
    dataset_log.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                "- Command: `uv run python -m equidock_diff.train --hetgnn-backbone`",
                "- Graph source: `dataset`",
                "- Training steps: `9`",
                "- Sample steps: `10`",
                "- Node count: `174`",
                "- Edge count: `3554`",
                "- Final loss: `0.338327` at step `9`",
                "- Best loss: `0.338327`",
                "- Training seconds: `1.320`",
                "- Raw Ligand Rmse: `2.276072`",
                "- Aligned Ligand Rmsd: `2.142486`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    records = discover_records("*_log.md")

    assert [record.log_path.name for record in records] == ["10gs_baseline_log.md"]
