from __future__ import annotations

from pathlib import Path

import pytest

from kiral.evaluation_summary import (
    ExperimentRecord,
    assert_expected_combinations_present,
    discover_records,
    filter_records_by_manifest,
    grouped_means,
    infer_model,
    infer_noise_schedule,
    parse_experiment_log,
    percent_reduction,
    resolve_expected_models,
    resolve_expected_seeds,
    resolve_expected_schedules,
    select_records,
    success_rate,
    write_latex,
)


def test_infer_model_detects_primary_variants() -> None:
    assert infer_model("uv run python -m kiral.train") == "EGNN baseline"
    assert (
        infer_model("uv run python -m kiral.train --hetgnn-backbone")
        == "heterogeneous frame-based backbone"
    )
    assert (
        infer_model("uv run python -m kiral.train --complete-frame --ligand-global-node")
        == "EGNN + complete frames + ligand context"
    )


def test_infer_model_detects_combined_egnn_variants() -> None:
    assert (
        infer_model("uv run python -m kiral.train --hetero-edges --ligand-global-node")
        == "EGNN + typed edges + ligand context"
    )
    assert (
        infer_model("uv run python -m kiral.train --hetero-edges --complete-frame")
        == "EGNN + typed edges + complete frames"
    )
    assert (
        infer_model(
            "uv run python -m kiral.train --hetero-edges --complete-frame --ligand-global-node"
        )
        == "EGNN + typed edges + complete frames + ligand context"
    )
    assert (
        infer_model(
            "uv run python -m kiral.train --frame-hetero-backbone --hetero-edges --complete-frame --ligand-global-node"
        )
        == "heterogeneous frame-based backbone"
    )


def test_infer_model_and_schedule_detect_checkpoint_resample_logs() -> None:
    command = (
        "uv run python -m kiral.resample_from_checkpoint "
        "--checkpoint docs/training/panel20/frame_backbone_cosine_inference_diag/checkpoints/"
        "184l_frame_backbone_cosine_longer_training_seed42.pt"
    )

    assert infer_model(command) == "heterogeneous frame-based backbone"
    assert infer_noise_schedule(command) == "cosine"


def test_parse_experiment_log_extracts_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "10gs_hetgnn_log.md"
    log_path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                "- Command: `uv run python -m kiral.train --hetgnn-backbone --protein-path data/x/10gs/10gs_protein.pdb`",
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
    assert record.seed == 42
    assert record.noise_schedule == "linear"
    assert record.graph_source == "real_pair"
    assert record.final_loss == 0.061780
    assert record.best_loss == 0.039261
    assert record.raw_ligand_rmse == 1.637888
    assert record.aligned_ligand_rmsd == 1.420224


def test_select_records_prefers_non_rerun_logs() -> None:
    preferred = ExperimentRecord(
        complex_id="10gs",
        model="heterogeneous frame-based backbone",
        seed=42,
        noise_schedule="linear",
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
        seed=42,
        noise_schedule="linear",
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
        seed=42,
        noise_schedule="linear",
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
        seed=42,
        noise_schedule="linear",
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


def test_select_records_keeps_distinct_schedules_for_same_seed() -> None:
    linear = ExperimentRecord(
        complex_id="10gs",
        model="EGNN baseline",
        seed=42,
        noise_schedule="linear",
        log_path=Path("docs/training/10gs_baseline_linear_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=147,
        edge_count=2992,
        best_loss=0.09,
        final_loss=0.69,
        training_seconds=1.5,
        raw_ligand_rmse=3.6,
        aligned_ligand_rmsd=2.6,
    )
    cosine = ExperimentRecord(
        complex_id="10gs",
        model="EGNN baseline",
        seed=42,
        noise_schedule="cosine",
        log_path=Path("docs/training/10gs_baseline_cosine_log.md"),
        command="",
        graph_source="real_pair",
        training_steps=100,
        sample_steps=25,
        node_count=147,
        edge_count=2992,
        best_loss=0.08,
        final_loss=0.40,
        training_seconds=1.7,
        raw_ligand_rmse=4.2,
        aligned_ligand_rmsd=3.8,
    )

    records = select_records([cosine, linear], models="primary")

    assert records == [cosine, linear]


def test_grouped_means_and_percent_reduction() -> None:
    records = [
        ExperimentRecord(
            complex_id="10gs",
            model="EGNN baseline",
            seed=42,
            noise_schedule="linear",
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
            seed=42,
            noise_schedule="linear",
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

    baseline = next(row for row in means if row.model == "EGNN baseline")
    frame_backbone = next(
        row for row in means if row.model == "heterogeneous frame-based backbone"
    )

    assert baseline.mean_raw_ligand_rmse == 3.0
    assert frame_backbone.mean_aligned_ligand_rmsd == 1.0
    assert baseline.success_at_2a == 100.0
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
                "- Command: `uv run python -m kiral.train --protein-path data/x/10gs/10gs_protein.pdb`",
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
                "- Command: `uv run python -m kiral.train --hetgnn-backbone`",
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


def test_filter_records_by_manifest_preserves_manifest_order() -> None:
    records = [
        ExperimentRecord(
            complex_id="10gs",
            model="EGNN baseline",
            seed=42,
            noise_schedule="linear",
            log_path=Path("10gs_log.md"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.1,
            final_loss=0.2,
            training_seconds=1.0,
            raw_ligand_rmse=3.0,
            aligned_ligand_rmsd=2.0,
        ),
        ExperimentRecord(
            complex_id="11gs",
            model="EGNN baseline",
            seed=42,
            noise_schedule="linear",
            log_path=Path("11gs_log.md"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.2,
            final_loss=0.3,
            training_seconds=1.0,
            raw_ligand_rmse=4.0,
            aligned_ligand_rmsd=3.0,
        ),
    ]

    filtered = filter_records_by_manifest(records, ["11gs", "10gs"])

    assert [record.complex_id for record in filtered] == ["11gs", "10gs"]


def test_assert_expected_combinations_present_checks_models_and_seeds() -> None:
    records = [
        ExperimentRecord(
            complex_id="10gs",
            model="EGNN baseline",
            seed=42,
            noise_schedule="linear",
            log_path=Path("10gs_baseline_log.md"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.1,
            final_loss=0.2,
            training_seconds=1.0,
            raw_ligand_rmse=3.0,
            aligned_ligand_rmsd=2.0,
        )
    ]

    with pytest.raises(ValueError, match="Missing expected experiment logs"):
        assert_expected_combinations_present(
            records,
            manifest_complex_ids=["10gs"],
            expected_models=["EGNN baseline", "heterogeneous frame-based backbone"],
            expected_seeds=[42],
            expected_schedules=["linear"],
        )


def test_resolve_expected_models_and_seeds_defaults() -> None:
    assert resolve_expected_models(models="primary", expected_models=None) == [
        "EGNN baseline",
        "heterogeneous frame-based backbone",
    ]
    assert resolve_expected_models(models="all", expected_models=None) == []
    assert resolve_expected_seeds(None) == [42]
    assert resolve_expected_seeds([43, 44]) == [43, 44]
    assert resolve_expected_schedules(None) == ["linear"]
    assert resolve_expected_schedules(["linear", "cosine"]) == ["linear", "cosine"]


def test_infer_noise_schedule_detects_cosine_flag() -> None:
    assert infer_noise_schedule("uv run python -m kiral.train") == "linear"
    assert (
        infer_noise_schedule("uv run python -m kiral.train --noise-schedule cosine")
        == "cosine"
    )


def test_grouped_means_supports_schedule_split_and_success_rates(tmp_path: Path) -> None:
    records = [
        ExperimentRecord(
            complex_id="10gs",
            model="EGNN baseline",
            seed=42,
            noise_schedule="linear",
            log_path=Path("linear_a.md"),
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
            aligned_ligand_rmsd=1.5,
        ),
        ExperimentRecord(
            complex_id="11gs",
            model="EGNN baseline",
            seed=43,
            noise_schedule="cosine",
            log_path=Path("cosine_a.md"),
            command="",
            graph_source="real_pair",
            training_steps=100,
            sample_steps=25,
            node_count=1,
            edge_count=1,
            best_loss=0.2,
            final_loss=0.3,
            training_seconds=3.0,
            raw_ligand_rmse=4.0,
            aligned_ligand_rmsd=5.5,
        ),
    ]

    means = grouped_means(records, group_by="model_schedule")

    linear = next(row for row in means if row.noise_schedule == "linear")
    cosine = next(row for row in means if row.noise_schedule == "cosine")

    assert linear.success_at_2a == 100.0
    assert linear.success_at_5a == 100.0
    assert cosine.success_at_2a == 0.0
    assert cosine.success_at_5a == 0.0
    assert success_rate([1.5, 5.5], 5.0) == 50.0

    latex_path = tmp_path / "summary.tex"
    write_latex(latex_path, means, group_by="model_schedule")
    latex = latex_path.read_text(encoding="utf-8")
    assert "Success@2\\AA{}" in latex
    assert "Mean Train s" in latex
    assert "cosine" in latex
