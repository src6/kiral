from __future__ import annotations

import csv
from pathlib import Path

from equidock_diff.dual_host_runner import (
    UnitMetrics,
    _load_run_metrics,
    build_local_runner_argv,
    build_parser,
    build_remote_runner_argv,
    main,
    plan_scheduled_units,
)


def _write_dataset_pair(root: Path, complex_id: str) -> None:
    pair_dir = root / "protein_ligand_general_minus_refined" / "chunk" / complex_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / f"{complex_id}_protein.pdb").write_text("END\n", encoding="utf-8")
    (pair_dir / f"{complex_id}_ligand.sdf").write_text("$$$$\n", encoding="utf-8")


def test_plan_scheduled_units_routes_single_complex_smoke_local(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    args = build_parser().parse_args(
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
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--dual-tag",
            "probe",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    units = plan_scheduled_units(args)

    assert len(units) == 1
    assert units[0].worker_pool == "local_cpu"


def test_plan_scheduled_units_routes_multi_complex_remote(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    _write_dataset_pair(dataset_root, "11gs")
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--complex-id",
            "11gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--steps",
            "200",
            "--dual-tag",
            "probe",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    units = plan_scheduled_units(args)

    assert len(units) == 2
    assert {unit.worker_pool for unit in units} == {"remote_cpu"}


def test_plan_scheduled_units_explicit_split_by_complex(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    for complex_id in ("13gs", "16pk", "184l", "186l", "187l", "188l", "1a28", "10gs"):
        _write_dataset_pair(dataset_root, complex_id)
    args = build_parser().parse_args(
        [
            "--complex-id",
            "13gs",
            "--complex-id",
            "16pk",
            "--complex-id",
            "184l",
            "--complex-id",
            "186l",
            "--complex-id",
            "187l",
            "--complex-id",
            "188l",
            "--complex-id",
            "1a28",
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--steps",
            "200",
            "--dual-tag",
            "probe",
            "--routing-policy",
            "explicit",
            "--local-complex-id",
            "13gs",
            "--local-complex-id",
            "16pk",
            "--remote-complex-id",
            "184l",
            "--remote-complex-id",
            "186l",
            "--remote-complex-id",
            "187l",
            "--remote-complex-id",
            "188l",
            "--remote-complex-id",
            "1a28",
            "--remote-complex-id",
            "10gs",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    units = plan_scheduled_units(args)

    local_ids = {unit.complex_id for unit in units if unit.worker_pool == "local_cpu"}
    remote_ids = {unit.complex_id for unit in units if unit.worker_pool == "remote_cpu"}
    assert local_ids == {"13gs", "16pk"}
    assert remote_ids == {"184l", "186l", "187l", "188l", "1a28", "10gs"}


def test_build_runner_argvs_include_crop_cutoff_and_cpu_policy(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    _write_dataset_pair(dataset_root, "10gs")
    args = build_parser().parse_args(
        [
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--context-policy",
            "gated",
            "--protein-node-budget",
            "256",
            "--crop-cutoff",
            "8.0",
            "--crop-cutoff",
            "10.0",
            "--ligand-protein-contact-weight",
            "0.05",
            "--sample-steps",
            "25",
            "--sample-steps",
            "50",
            "--dual-tag",
            "probe",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
        ]
    )

    unit = plan_scheduled_units(args)[0]
    local_argv = build_local_runner_argv(unit, args)
    remote_argv = build_remote_runner_argv(unit, args)

    assert "--device-policy" in local_argv
    assert "canonical" in local_argv
    assert "--context-policy" in local_argv
    assert "--protein-node-budget" in local_argv
    assert "--crop-cutoff" in local_argv
    assert "--protein-node-budget" in remote_argv
    assert "--crop-cutoff" in remote_argv
    assert "--ligand-protein-contact-weight" in local_argv
    assert "--ligand-protein-contact-weight" in remote_argv
    assert "equidock_diff.research_runner" in " ".join(local_argv)
    assert "equidock_diff.remote_research_runner" in " ".join(remote_argv)


def test_main_dry_run_writes_combined_plan_and_worker_status(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    for complex_id in ("13gs", "16pk", "184l", "186l", "187l", "188l", "1a28", "10gs"):
        _write_dataset_pair(dataset_root, complex_id)

    result = main(
        [
            "--complex-id",
            "13gs",
            "--complex-id",
            "16pk",
            "--complex-id",
            "184l",
            "--complex-id",
            "186l",
            "--complex-id",
            "187l",
            "--complex-id",
            "188l",
            "--complex-id",
            "1a28",
            "--complex-id",
            "10gs",
            "--dataset-root",
            str(dataset_root),
            "--model",
            "frame_backbone",
            "--noise-schedule",
            "cosine",
            "--steps",
            "200",
            "--dual-tag",
            "mixed_probe",
            "--routing-policy",
            "explicit",
            "--local-complex-id",
            "13gs",
            "--local-complex-id",
            "16pk",
            "--remote-complex-id",
            "184l",
            "--remote-complex-id",
            "186l",
            "--remote-complex-id",
            "187l",
            "--remote-complex-id",
            "188l",
            "--remote-complex-id",
            "1a28",
            "--remote-complex-id",
            "10gs",
            "--remote-host",
            "macmini-tailscale",
            "--remote-repo",
            "/Users/sadik/Projects/equidock-diff",
            "--output-root",
            str(tmp_path / "dual_runs"),
            "--dry-run",
        ]
    )

    assert result == 0
    dual_root = tmp_path / "dual_runs" / "mixed_probe"
    plan_csv = dual_root / "plan.csv"
    status_csv = dual_root / "status.csv"
    worker_status_csv = dual_root / "worker_status.csv"
    run_index_md = dual_root / "run_index.md"
    assert plan_csv.exists()
    assert status_csv.exists()
    assert worker_status_csv.exists()
    assert run_index_md.exists()
    assert "remote_cpu" in plan_csv.read_text(encoding="utf-8")
    assert "local_cpu" in plan_csv.read_text(encoding="utf-8")
    assert "planned" in status_csv.read_text(encoding="utf-8")
    assert "remote_cpu_1" in worker_status_csv.read_text(encoding="utf-8")
    assert "local_cpu_1" in worker_status_csv.read_text(encoding="utf-8")


def test_load_run_metrics_uses_status_csv_for_failed_counts(tmp_path: Path) -> None:
    run_index_csv = tmp_path / "run_index.csv"
    status_csv = tmp_path / "status.csv"
    with run_index_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "complex_id",
                "model",
                "noise_schedule",
                "seed",
                "action",
                "device",
                "checkpoint_path",
                "experiment_log",
                "loss_csv",
                "sample_path",
                "trajectory_path",
                "final_loss",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
            ]
        )
        writer.writerow(
            [
                "run_a",
                "10gs",
                "heterogeneous frame-based backbone",
                "cosine",
                "42",
                "trained_from_scratch",
                "cpu",
                "ckpt.pt",
                "log.md",
                "loss.csv",
                "",
                "",
                "0.5",
                "1.1",
                "0.9",
                "3.0",
            ]
        )
    with status_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_name",
                "complex_id",
                "model",
                "seed",
                "planned_action",
                "status",
                "device",
                "checkpoint_path",
                "experiment_log",
                "final_action",
                "final_loss",
                "raw_ligand_rmse",
                "aligned_ligand_rmsd",
                "training_seconds",
                "error",
            ]
        )
        writer.writerow(["run_a", "10gs", "model", "42", "train", "failed", "cpu", "ckpt.pt", "log.md", "", "", "", "", "", "boom"])

    metrics = _load_run_metrics(run_index_csv)

    assert metrics == UnitMetrics(
        total_runs=1,
        completed_runs=1,
        failed_runs=1,
        mean_aligned_ligand_rmsd=0.9,
        mean_training_seconds=3.0,
    )
