from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.vina_baseline import (
    build_vina_command,
    evaluate_docked_pose,
    parse_pdbqt_positions,
    prepare_pdbqt_inputs,
)


def test_build_vina_command_includes_box_and_runtime_flags(tmp_path: Path) -> None:
    command = build_vina_command(
        vina_binary="vina",
        receptor_pdbqt=tmp_path / "receptor.pdbqt",
        ligand_pdbqt=tmp_path / "ligand.pdbqt",
        output_pdbqt=tmp_path / "out.pdbqt",
        center=torch.tensor([1.0, 2.0, 3.0]),
        box_size=20.0,
        exhaustiveness=8,
        cpu=1,
    )

    assert command[:2] == ["vina", "--receptor"]
    assert "--center_x" in command
    assert "--size_z" in command
    assert "--exhaustiveness" in command


def test_parse_pdbqt_positions_reads_atom_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "pose.pdbqt"
    path.write_text(
        "\n".join(
            [
                "ATOM      1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00      A    C",
                "ATOM      2  O1  LIG A   1       2.000   3.000   4.000  1.00  0.00      A    O",
                "ENDMDL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    positions = parse_pdbqt_positions(path)

    assert positions.shape == (2, 3)
    assert torch.allclose(positions[0], torch.tensor([1.0, 2.0, 3.0]))


def test_parse_pdbqt_positions_uses_first_model_when_multiple_poses_exist(tmp_path: Path) -> None:
    path = tmp_path / "multi_pose.pdbqt"
    path.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  C1  LIG A   1       1.000   2.000   3.000  1.00  0.00      A    C",
                "ENDMDL",
                "MODEL        2",
                "ATOM      1  C1  LIG A   1       9.000   8.000   7.000  1.00  0.00      A    C",
                "ENDMDL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    positions = parse_pdbqt_positions(path)

    assert positions.shape == (1, 3)
    assert torch.allclose(positions[0], torch.tensor([1.0, 2.0, 3.0]))


def test_parse_pdbqt_positions_preserves_single_pose_files_without_model_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "single_pose_no_model.pdbqt"
    path.write_text(
        "\n".join(
            [
                "REMARK  single-pose output",
                "ATOM      1  C1  LIG A   1       4.000   5.000   6.000  1.00  0.00      A    C",
                "ATOM      2  O1  LIG A   1       7.000   8.000   9.000  1.00  0.00      A    O",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    positions = parse_pdbqt_positions(path)

    assert positions.shape == (2, 3)
    assert torch.allclose(positions[1], torch.tensor([7.0, 8.0, 9.0]))


def test_evaluate_docked_pose_returns_zero_for_identical_inputs() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

    raw_rmse, aligned = evaluate_docked_pose(positions, positions.clone())

    assert raw_rmse == 0.0
    assert aligned == pytest.approx(0.0, abs=1e-6)


def test_prepare_pdbqt_inputs_requires_paths_or_preparation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass both --receptor-pdbqt and --ligand-pdbqt"):
        prepare_pdbqt_inputs(
            protein_path=tmp_path / "protein.pdb",
            ligand_path=tmp_path / "ligand.sdf",
            output_dir=tmp_path / "out",
            receptor_pdbqt=None,
            ligand_pdbqt=None,
            prepare_inputs=False,
        )
