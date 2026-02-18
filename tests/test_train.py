from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.train import build_synthetic_graph, load_graph_inputs, make_model, training_step


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
