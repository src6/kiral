from __future__ import annotations

from pathlib import Path

import pytest

from equidock_diff.research_runner import (
    build_parser,
    expand_run_specs,
    main,
    plan_runs,
    resolve_runner_device,
)


def _write_dataset_pair(root: Path, complex_id: str) -> None:
    pair_dir = root / "protein_ligand_general_minus_refined" / "chunk" / complex_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / f"{complex_id}_protein.pdb").write_text("END\n", encoding="utf-8")
    (pair_dir / f"{complex_id}_ligand.sdf").write_text("$$$$\n", encoding="utf-8")


def _write_log(
    path: Path,
    *,
    command: str,
    seed: int,
    training_steps: int,
    sample_steps: int,
    final_loss: float,
    best_loss: float,
    training_seconds: float,
    raw_ligand_rmse: float,
    aligned_ligand_rmsd: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                f"- Command: `{command}`",
                f"- Seed: `{seed}`",
                "- Device: `cpu`",
                "- Graph source: `real_pair`",
                f"- Training steps: `{training_steps}`",
                f"- Sample steps: `{sample_steps}`",
                "- Node count: `20`",
                "- Edge count: `200`",
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


def test_expand_run_specs_resolves_complex_ids_and_builds_matrix(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    _write_dataset_pair(dataset_root, "11gs")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("11gs\n10gs\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--seed",
            "42",
            "--seed",
            "43",
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--tag",
            "probe",
        ]
    )

    specs = expand_run_specs(args)

    assert len(specs) == 8
    assert specs[0].complex_id == "10gs"
    assert {item.seed for item in specs} == {42, 43}
    assert {item.sample_steps for item in specs} == {25, 50}


def test_plan_runs_routes_inference_only_variants_to_resample(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--tag",
            "probe",
        ]
    )

    planned = plan_runs(expand_run_specs(args), tmp_path / "runs" / "probe")

    assert [item.action for item in planned] == ["train", "resample"]


def test_plan_runs_trains_again_when_training_signature_changes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    parser = build_parser()
    args = parser.parse_args(
        [
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--steps",
            "100",
            "--steps",
            "200",
            "--tag",
            "probe",
        ]
    )

    planned = plan_runs(expand_run_specs(args), tmp_path / "runs" / "probe")

    assert [item.action for item in planned] == ["train", "train"]


def test_resolve_runner_device_prefers_mps_for_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    requested, actual = resolve_runner_device("scratch", None)

    assert requested == "mps"
    assert actual.type == "mps"


def test_main_dry_run_writes_planned_run_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")

    result = main(
        [
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--output-root",
            str(tmp_path / "runs"),
            "--tag",
            "dry_probe",
            "--dry-run",
        ]
    )

    assert result == 0
    markdown = (tmp_path / "runs" / "dry_probe" / "run_index.md").read_text(encoding="utf-8")
    csv_text = (tmp_path / "runs" / "dry_probe" / "run_index.csv").read_text(encoding="utf-8")
    assert "planned_train" in markdown
    assert "planned_resample" in markdown
    assert "planned_train" in csv_text


def test_main_executes_train_resample_and_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    compare_root = tmp_path / "runs" / "prior" / "logs"
    compare_root.mkdir(parents=True, exist_ok=True)
    _write_log(
        compare_root / "10gs_frame_backbone_cosine_seed42_steps100_sample25_time1_shape0_score10_pos50_log.md",
        command=(
            "uv run python -m equidock_diff.train --frame-hetero-backbone "
            "--noise-schedule cosine --protein-path data/x/10gs/10gs_protein.pdb"
        ),
        seed=42,
        training_steps=100,
        sample_steps=25,
        final_loss=0.2,
        best_loss=0.1,
        training_seconds=2.0,
        raw_ligand_rmse=1.5,
        aligned_ligand_rmsd=1.2,
    )

    calls: list[str] = []

    def _fake_execute_train(planned, *, requested_device: str, dataset_cache_dir: Path, save_artifacts: bool) -> int:
        calls.append(f"train:{planned.run_name}:{requested_device}:{save_artifacts}")
        planned.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        planned.checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
        _write_log(
            planned.log_path,
            command=(
                "uv run python -m equidock_diff.train --frame-hetero-backbone "
                f"--noise-schedule {planned.spec.noise_schedule} "
                f"--protein-path data/x/{planned.spec.complex_id}/{planned.spec.complex_id}_protein.pdb"
            ),
            seed=planned.spec.seed,
            training_steps=planned.spec.steps,
            sample_steps=planned.spec.sample_steps,
            final_loss=0.2,
            best_loss=0.1,
            training_seconds=2.0,
            raw_ligand_rmse=1.5,
            aligned_ligand_rmsd=1.2,
        )
        planned.loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
        planned.loss_csv_path.write_text("step,loss,beta_t\n", encoding="utf-8")
        return 0

    def _fake_execute_resample(planned, *, requested_device: str, save_artifacts: bool) -> int:
        calls.append(f"resample:{planned.run_name}:{requested_device}:{save_artifacts}")
        _write_log(
            planned.log_path,
            command=(
                "uv run python -m equidock_diff.resample_from_checkpoint "
                f"--checkpoint {planned.checkpoint_path}"
            ),
            seed=planned.spec.seed,
            training_steps=planned.spec.steps,
            sample_steps=planned.spec.sample_steps,
            final_loss=0.2,
            best_loss=0.1,
            training_seconds=2.0,
            raw_ligand_rmse=1.4,
            aligned_ligand_rmsd=1.1,
        )
        planned.loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
        planned.loss_csv_path.write_text("step,loss,beta_t\n", encoding="utf-8")
        return 0

    def _fake_run_comparison(*, current_root: Path, compare_against: str, model: str, noise_schedule: str):
        comparison_dir = current_root / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        markdown = comparison_dir / "vs_prior.md"
        csv_path = comparison_dir / "vs_prior.csv"
        markdown.write_text("# Comparison\n", encoding="utf-8")
        csv_path.write_text("a,b\n", encoding="utf-8")
        calls.append(f"compare:{compare_against}:{model}:{noise_schedule}")
        return markdown, csv_path

    monkeypatch.setattr("equidock_diff.research_runner.execute_train", _fake_execute_train)
    monkeypatch.setattr("equidock_diff.research_runner.execute_resample", _fake_execute_resample)
    monkeypatch.setattr("equidock_diff.research_runner.run_comparison", _fake_run_comparison)

    result = main(
        [
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--output-root",
            str(tmp_path / "runs"),
            "--tag",
            "current",
            "--compare-against",
            "prior",
        ]
    )

    assert result == 0
    assert any(entry.startswith("train:") for entry in calls)
    assert any(entry.startswith("resample:") for entry in calls)
    assert any(entry.startswith("compare:prior:frame_backbone:cosine") for entry in calls)
    csv_text = (tmp_path / "runs" / "current" / "run_index.csv").read_text(encoding="utf-8")
    assert "trained_from_scratch" in csv_text
    assert "resampled_from_checkpoint" in csv_text
