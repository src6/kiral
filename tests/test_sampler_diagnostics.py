from __future__ import annotations

import json
from pathlib import Path

from equidock_diff.sampler_diagnostics import load_summary, main


def _write_log(path: Path, *, complex_id: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Experiment Log",
                "",
                f"- Command: `uv run python -m equidock_diff.train --frame-hetero-backbone --noise-schedule cosine --protein-path data/x/{complex_id}/{complex_id}_protein.pdb`",
                "- Seed: `42`",
                "- Graph source: `real_pair`",
                "- Training steps: `200`",
                "- Sample steps: `25`",
                "- Node count: `20`",
                "- Edge count: `200`",
                "- Final loss: `0.020000` at step `200`",
                "- Best loss: `0.010000`",
                "- Training seconds: `3.000`",
                "- Raw Ligand Rmse: `1.200000`",
                "- Aligned Ligand Rmsd: `1.000000`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_diagnostics(path: Path, *, complex_id: str, base_value: float) -> None:
    payload = {
        "complex_id": complex_id,
        "seed": 42,
        "sample_steps": 25,
        "sample_time_power": 1.0,
        "sample_score_clip": 10.0,
        "sample_position_clip": 50.0,
        "anchor_protein": True,
        "experiment_log": str(path.with_suffix(".md")),
        "steps": [
            {
                "step_index": idx,
                "t": 1.0 - 0.1 * idx,
                "dt": 0.1,
                "mean_score_norm_before_clip": base_value + idx,
                "max_score_norm_before_clip": base_value + idx + 0.5,
                "mean_score_norm_after_clip": base_value + idx - 0.1,
                "max_score_norm_after_clip": base_value + idx + 0.4,
                "clipped_coordinate_fraction": 0.0,
                "ligand_center_displacement": 0.2 + 0.01 * idx,
                "ligand_radius": 1.0 + 0.05 * idx,
            }
            for idx in range(5)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_summary_extracts_late_step_aggregates(tmp_path: Path) -> None:
    diag_path = tmp_path / "10gs.json"
    log_path = tmp_path / "10gs.md"
    _write_log(log_path, complex_id="10gs")
    _write_diagnostics(diag_path, complex_id="10gs", base_value=1.0)

    summary = load_summary(diag_path)

    assert summary.complex_id == "10gs"
    assert summary.mean_late_score_norm_before_clip > 0.0
    assert summary.aligned_ligand_rmsd == 1.0


def test_main_writes_single_and_comparison_reports(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    for root, base_value in ((left_dir, 1.0), (right_dir, 0.8)):
        diag_path = root / "10gs.json"
        log_path = root / "10gs.md"
        _write_log(log_path, complex_id="10gs")
        _write_diagnostics(diag_path, complex_id="10gs", base_value=base_value)

    single_csv = tmp_path / "single.csv"
    single_md = tmp_path / "single.md"
    compare_csv = tmp_path / "compare.csv"
    compare_md = tmp_path / "compare.md"

    single_result = main(
        [
            "--glob",
            str(left_dir / "*.json"),
            "--output-csv",
            str(single_csv),
            "--output-markdown",
            str(single_md),
        ]
    )
    compare_result = main(
        [
            "--left-glob",
            str(left_dir / "*.json"),
            "--right-glob",
            str(right_dir / "*.json"),
            "--output-csv",
            str(compare_csv),
            "--output-markdown",
            str(compare_md),
        ]
    )

    assert single_result == 0
    assert compare_result == 0
    assert "Sampler Diagnostics" in single_md.read_text(encoding="utf-8")
    assert "Sampler Diagnostics Comparison" in compare_md.read_text(encoding="utf-8")
