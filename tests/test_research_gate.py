from __future__ import annotations

import csv
from pathlib import Path

from kiral.research_gate import main


def _write_run_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_research_gate_reports_best_per_complex_and_writes_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    control_csv = tmp_path / "control.csv"
    candidate_csv = tmp_path / "candidate.csv"
    summary_md = tmp_path / "summary.md"
    comparison_csv = tmp_path / "comparison.csv"
    _write_run_index(
        control_csv,
        [
            {
                "run_name": "c10gs_seed42",
                "complex_id": "10gs",
                "model": "heterogeneous frame-based backbone",
                "noise_schedule": "cosine",
                "seed": "42",
                "action": "train",
                "device": "cpu",
                "checkpoint_path": "ckpt.pt",
                "experiment_log": "log.md",
                "loss_csv": "loss.csv",
                "sample_path": "",
                "trajectory_path": "",
                "final_loss": "0.5",
                "raw_ligand_rmse": "1.2",
                "aligned_ligand_rmsd": "0.9",
                "training_seconds": "2.0",
            },
            {
                "run_name": "c11gs_seed42",
                "complex_id": "11gs",
                "model": "heterogeneous frame-based backbone",
                "noise_schedule": "cosine",
                "seed": "42",
                "action": "train",
                "device": "cpu",
                "checkpoint_path": "ckpt.pt",
                "experiment_log": "log.md",
                "loss_csv": "loss.csv",
                "sample_path": "",
                "trajectory_path": "",
                "final_loss": "0.5",
                "raw_ligand_rmse": "1.0",
                "aligned_ligand_rmsd": "0.8",
                "training_seconds": "2.0",
            },
        ],
    )
    _write_run_index(
        candidate_csv,
        [
            {
                "run_name": "cand10gs_seed42",
                "complex_id": "10gs",
                "model": "heterogeneous frame-based backbone",
                "noise_schedule": "cosine",
                "seed": "42",
                "action": "train",
                "device": "cpu",
                "checkpoint_path": "ckpt.pt",
                "experiment_log": "log.md",
                "loss_csv": "loss.csv",
                "sample_path": "",
                "trajectory_path": "",
                "final_loss": "0.5",
                "raw_ligand_rmse": "1.1",
                "aligned_ligand_rmsd": "0.7",
                "training_seconds": "2.0",
            },
            {
                "run_name": "cand10gs_seed43",
                "complex_id": "10gs",
                "model": "heterogeneous frame-based backbone",
                "noise_schedule": "cosine",
                "seed": "43",
                "action": "train",
                "device": "cpu",
                "checkpoint_path": "ckpt.pt",
                "experiment_log": "log.md",
                "loss_csv": "loss.csv",
                "sample_path": "",
                "trajectory_path": "",
                "final_loss": "0.5",
                "raw_ligand_rmse": "1.4",
                "aligned_ligand_rmsd": "0.95",
                "training_seconds": "2.0",
            },
            {
                "run_name": "cand11gs_seed42",
                "complex_id": "11gs",
                "model": "heterogeneous frame-based backbone",
                "noise_schedule": "cosine",
                "seed": "42",
                "action": "train",
                "device": "cpu",
                "checkpoint_path": "ckpt.pt",
                "experiment_log": "log.md",
                "loss_csv": "loss.csv",
                "sample_path": "",
                "trajectory_path": "",
                "final_loss": "0.5",
                "raw_ligand_rmse": "1.2",
                "aligned_ligand_rmsd": "0.85",
                "training_seconds": "2.0",
            },
        ],
    )

    result = main(
        [
            "--control-csv",
            str(control_csv),
            "--candidate-csv",
            str(candidate_csv),
            "--output-markdown",
            str(summary_md),
            "--output-csv",
            str(comparison_csv),
            "--max-mean-raw-rmse-delta",
            "0.1",
            "--max-mean-aligned-rmsd-delta",
            "0.0",
            "--min-aligned-improved",
            "1",
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert "compared_complexes=2" in stdout
    assert "mean_raw_ligand_rmse_delta=0.050000" in stdout
    assert "mean_aligned_ligand_rmsd_delta=-0.075000" in stdout
    assert "aligned_improved_complexes=1/2" in stdout
    assert "gate_pass=yes" in stdout
    assert "gate_check_max_mean_raw_rmse_delta=pass" in stdout
    assert "gate_check_max_mean_aligned_rmsd_delta=pass" in stdout
    assert "gate_check_min_aligned_improved=pass" in stdout
    assert "cand10gs_seed42" in summary_md.read_text(encoding="utf-8")
    assert "cand11gs_seed42" in summary_md.read_text(encoding="utf-8")
    assert "Gate pass: `yes`" in summary_md.read_text(encoding="utf-8")
    assert comparison_csv.exists()
