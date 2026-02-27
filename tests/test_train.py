from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.utils.artifacts import (
    ligand_mask_from_features,
    write_experiment_log,
    write_ligand_artifacts,
)
from equidock_diff.train import (
    build_synthetic_graph,
    load_graph_inputs,
    make_model,
    sample_positions,
    training_step,
)


class _Args:
    hidden_dim = 32
    num_layers = 2


def test_training_step_is_finite() -> None:
    device = torch.device("cpu")
    model = make_model(_Args(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_load_graph_inputs_requires_both_real_paths() -> None:
    class _RealArgs(_Args):
        protein_path = Path("only_protein.pdb")
        ligand_path = None
        crop_cutoff = 10.0
        num_nodes = 8
        batch_size = 1

    with pytest.raises(ValueError, match="Pass both --protein-path and --ligand-path"):
        load_graph_inputs(_RealArgs(), torch.device("cpu"))


def test_training_step_is_finite_for_real_pair_graph(tmp_path: Path) -> None:
    rdkit = pytest.importorskip("rdkit")
    assert rdkit is not None

    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    ligand_path = tmp_path / "ethanol.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    writer = Chem.SDWriter(str(ligand_path))
    writer.write(mol)
    writer.close()

    protein_path = tmp_path / "toy_protein.pdb"
    protein_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   4.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       0.000   6.000   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _RealArgs(_Args):
        crop_cutoff = 10.0
        num_nodes = 8
        batch_size = 1

    real_args = _RealArgs()
    real_args.protein_path = protein_path
    real_args.ligand_path = ligand_path

    device = torch.device("cpu")
    node_features, positions, edge_index = load_graph_inputs(real_args, device)
    model = make_model(real_args, device, node_dim=node_features.shape[1])

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert node_features.shape[1] == 17
    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_sample_positions_stays_finite_with_clipping() -> None:
    class _SampleArgs(_Args):
        hidden_dim = 32
        num_layers = 2

    device = torch.device("cpu")
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)
    model = make_model(_SampleArgs(), device, node_dim=node_features.shape[1])

    sampled_positions, trajectory = sample_positions(
        model,
        node_features,
        edge_index,
        positions.size(0),
        device,
        sample_steps=8,
        beta_min=0.1,
        beta_max=2.0,
        score_clip=10.0,
        position_clip=50.0,
    )

    assert torch.isfinite(sampled_positions).all()
    assert len(trajectory) == 9
    assert torch.isfinite(trajectory[-1]).all()


def test_write_experiment_log_records_run_metadata(tmp_path: Path) -> None:
    class _LogArgs(_Args):
        seed = 7
        steps = 5
        sample_steps = 10
        crop_cutoff = 8.0
        edge_cutoff = 4.5

    log_path = tmp_path / "experiment.md"
    output_path = tmp_path / "sample.pdb"
    trajectory_path = tmp_path / "trajectory.pdb"
    loss_csv_path = tmp_path / "loss.csv"
    plot_path = tmp_path / "plot.png"

    write_experiment_log(
        log_path,
        command="uv run python -m equidock_diff.train --steps 5",
        device=torch.device("cpu"),
        graph_source="real_pair",
        args=_LogArgs(),
        loss_rows=[(1, 1.5, 0.2), (5, 0.25, 0.9)],
        training_seconds=0.42,
        node_count=147,
        edge_count=2992,
        sample_path=output_path,
        trajectory_path=trajectory_path,
        loss_csv_path=loss_csv_path,
        plot_path=plot_path,
    )

    contents = log_path.read_text(encoding="utf-8")
    assert "uv run python -m equidock_diff.train --steps 5" in contents
    assert "- Seed: `7`" in contents
    assert "- Graph source: `real_pair`" in contents
    assert "- Node count: `147`" in contents
    assert "- Edge count: `2992`" in contents
    assert "- Final loss: `0.250000` at step `5`" in contents
    assert str(output_path) in contents


def test_ligand_mask_from_features_uses_indicator_column() -> None:
    node_features = torch.tensor(
        [
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    mask = ligand_mask_from_features(node_features)

    assert mask.tolist() == [True, True, False]


def test_write_ligand_artifacts_filters_to_ligand_nodes(tmp_path: Path) -> None:
    node_features = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sampled_positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    trajectory = [
        sampled_positions.clone(),
        sampled_positions + 1.0,
    ]
    ligand_output = tmp_path / "ligand_sample.pdb"
    ligand_traj_output = tmp_path / "ligand_traj.pdb"

    written_sample, written_trajectory = write_ligand_artifacts(
        node_features=node_features,
        sampled_positions=sampled_positions,
        trajectory=trajectory,
        ligand_output=ligand_output,
        ligand_trajectory_output=ligand_traj_output,
    )

    assert written_sample == ligand_output
    assert written_trajectory == ligand_traj_output
    sample_lines = ligand_output.read_text(encoding="utf-8").splitlines()
    trajectory_lines = ligand_traj_output.read_text(encoding="utf-8").splitlines()
    assert len([line for line in sample_lines if line.startswith("ATOM")]) == 2
    assert len([line for line in trajectory_lines if line.startswith("MODEL")]) == 2
    assert "   9.000" not in ligand_output.read_text(encoding="utf-8")
