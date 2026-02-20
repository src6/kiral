from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.data.pipeline import (
    build_complete_edge_index,
    build_graph_batch,
    build_radius_edge_index,
    load_protein_graph,
    load_protein_ligand_graph,
)


def test_build_graph_batch_centers_ligand_and_crops_far_protein() -> None:
    node_features = torch.zeros(4, 16, dtype=torch.float32)
    positions = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 25.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ligand_mask = torch.tensor([True, True, False, False])
    edge_index = build_complete_edge_index(4, device=torch.device("cpu"))

    batch = build_graph_batch(node_features, positions, edge_index, ligand_mask, cutoff=10.0)

    assert batch.node_features.shape == (3, 17)
    assert batch.positions.shape == (3, 3)
    assert batch.edge_index.shape[0] == 2
    assert batch.mask is not None
    assert int(batch.mask.sum().item()) == 3
    assert torch.allclose(batch.positions[:2].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert batch.node_features[:2, -1].tolist() == [1.0, 1.0]
    assert batch.node_features[2, -1].item() == 0.0


def test_load_protein_graph_reads_atom_coordinates_and_features(tmp_path: Path) -> None:
    pdb_path = tmp_path / "toy_protein.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       1.500   0.000   0.000  1.00  0.00           C",
                "ATOM      3  O   GLY A   1       2.500   0.000   0.000  1.00  0.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    node_features, positions = load_protein_graph(pdb_path)

    assert node_features.shape == (3, 16)
    assert positions.shape == (3, 3)
    assert torch.isfinite(node_features).all()
    assert torch.isfinite(positions).all()
    assert node_features[0, 1].item() == 1.0  # N
    assert node_features[1, 0].item() == 1.0  # C
    assert node_features[2, 2].item() == 1.0  # O


def test_build_radius_edge_index_keeps_local_neighbors() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_index = build_radius_edge_index(positions, cutoff=1.6)
    edges = {tuple(edge) for edge in edge_index.t().tolist()}

    assert edges == {(0, 1), (1, 0)}


def test_load_protein_ligand_graph_from_files(tmp_path: Path) -> None:
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

    pdb_path = tmp_path / "toy_protein.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   4.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       0.000  30.000   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    batch = load_protein_ligand_graph(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
    )

    assert batch.node_features.shape[1] == 17
    assert batch.positions.shape[1] == 3
    assert batch.mask is not None
    assert int(batch.mask.sum().item()) >= 4
    ligand_count = int(batch.node_features[:, -1].sum().item())
    assert ligand_count >= 3
    assert batch.node_features[ligand_count:, -1].sum().item() == 0.0
    assert batch.edge_index.shape[1] > 0
