from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.data.pipeline import (
    build_complete_edge_index,
    build_graph_batch,
    build_radius_edge_index,
    graph_cache_path,
    ligand_max_span,
    load_protein_graph,
    load_protein_ligand_graph,
    load_protein_ligand_graph_cached,
    resolve_context_crop_cutoff,
)
from equidock_diff.data.io import (
    ProteinLigandPaths,
    filter_paths_by_complex_ids,
    load_split_complex_ids,
)


def _write_test_pair(tmp_path: Path) -> tuple[Path, Path]:
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
    return pdb_path, ligand_path


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
    assert batch.crop_mask is not None
    assert batch.mask is not None
    assert torch.equal(batch.crop_mask, batch.mask)
    assert int(batch.mask.sum().item()) == 3
    assert torch.allclose(batch.positions[:2].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert batch.node_features[:2, -1].tolist() == [1.0, 1.0]
    assert batch.node_features[2, -1].item() == 0.0
    assert batch.ligand_bond_index is None


def test_build_graph_batch_is_translation_consistent() -> None:
    node_features = torch.arange(64, dtype=torch.float32).view(4, 16)
    positions = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 11.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ligand_mask = torch.tensor([True, True, False, False])
    translation = torch.tensor([7.5, -3.0, 2.25], dtype=torch.float32)

    batch = build_graph_batch(
        node_features,
        positions,
        build_radius_edge_index(positions, cutoff=4.5),
        ligand_mask,
        cutoff=8.0,
    )
    translated_batch = build_graph_batch(
        node_features,
        positions + translation,
        build_radius_edge_index(positions + translation, cutoff=4.5),
        ligand_mask,
        cutoff=8.0,
    )

    assert torch.equal(batch.crop_mask, translated_batch.crop_mask)
    assert torch.equal(batch.mask, translated_batch.mask)
    assert torch.equal(batch.edge_index, translated_batch.edge_index)
    assert torch.allclose(batch.node_features, translated_batch.node_features)
    assert torch.allclose(batch.positions, translated_batch.positions, atol=1e-6)


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
    pdb_path, ligand_path = _write_test_pair(tmp_path)

    batch = load_protein_ligand_graph(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
    )

    assert batch.node_features.shape[1] == 17
    assert batch.positions.shape[1] == 3
    assert batch.crop_mask is not None
    assert batch.mask is not None
    assert int(batch.mask.sum().item()) >= 4
    ligand_count = int(batch.node_features[:, -1].sum().item())
    assert ligand_count >= 3
    assert batch.node_features[ligand_count:, -1].sum().item() == 0.0
    assert batch.edge_index.shape[1] > 0
    assert batch.ligand_bond_index is not None
    assert batch.ligand_bond_index.shape[0] == 2
    assert torch.all(batch.ligand_bond_index < ligand_count)
    assert batch.resolved_crop_cutoff == pytest.approx(10.0)


def test_resolve_context_crop_cutoff_is_clamped() -> None:
    compact = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    wide = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32)

    assert ligand_max_span(compact) == pytest.approx(0.0)
    assert resolve_context_crop_cutoff(compact, context_policy="adaptive", cutoff=8.0) == pytest.approx(6.0)
    assert resolve_context_crop_cutoff(wide, context_policy="adaptive", cutoff=8.0) == pytest.approx(10.0)
    assert resolve_context_crop_cutoff(wide, context_policy="fixed", cutoff=8.0) == pytest.approx(8.0)


def test_load_protein_ligand_graph_adaptive_context_records_resolved_cutoff(tmp_path: Path) -> None:
    pdb_path, ligand_path = _write_test_pair(tmp_path)

    batch = load_protein_ligand_graph(
        pdb_path,
        ligand_path,
        cutoff=8.0,
        edge_cutoff=4.5,
        context_policy="adaptive",
    )

    assert batch.resolved_crop_cutoff is not None
    assert 6.0 <= batch.resolved_crop_cutoff <= 10.0


def test_load_split_complex_ids_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    split_path = tmp_path / "train.txt"
    split_path.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "10gs",
                "11gs extra_token_ignored",
                "  ",
                "1a30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    complex_ids = load_split_complex_ids(split_path)

    assert complex_ids == ["10gs", "11gs", "1a30"]


def test_filter_paths_by_complex_ids_preserves_requested_order() -> None:
    paths = [
        ProteinLigandPaths(Path("root/10gs_protein.pdb"), Path("root/10gs_ligand.sdf")),
        ProteinLigandPaths(Path("root/11gs_protein.pdb"), Path("root/11gs_ligand.sdf")),
        ProteinLigandPaths(Path("root/1a30_protein.pdb"), Path("root/1a30_ligand.sdf")),
    ]

    selected = filter_paths_by_complex_ids(paths, ["1a30", "10gs"])

    assert [entry.complex_id for entry in selected] == ["1a30", "10gs"]


def test_filter_paths_by_complex_ids_raises_for_missing_complex() -> None:
    paths = [
        ProteinLigandPaths(Path("root/10gs_protein.pdb"), Path("root/10gs_ligand.sdf")),
    ]

    with pytest.raises(ValueError, match="Missing complexes"):
        filter_paths_by_complex_ids(paths, ["10gs", "11gs"])


def test_load_protein_ligand_graph_cached_reuses_saved_graph(tmp_path: Path) -> None:
    pdb_path, ligand_path = _write_test_pair(tmp_path)
    cache_dir = tmp_path / "graph_cache"

    first_batch = load_protein_ligand_graph_cached(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
        cache_dir=cache_dir,
    )
    cache_path = graph_cache_path(
        cache_dir,
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
    )
    assert cache_path.exists()

    second_batch = load_protein_ligand_graph_cached(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
        cache_dir=cache_dir,
    )

    assert len(list(cache_dir.glob("*.pt"))) == 1
    assert torch.equal(first_batch.node_features, second_batch.node_features)
    assert torch.equal(first_batch.edge_index, second_batch.edge_index)
    assert torch.equal(first_batch.crop_mask, second_batch.crop_mask)
    assert torch.allclose(first_batch.positions, second_batch.positions)


def test_load_protein_ligand_graph_cached_invalidates_when_input_changes(tmp_path: Path) -> None:
    pdb_path, ligand_path = _write_test_pair(tmp_path)
    cache_dir = tmp_path / "graph_cache"

    original_batch = load_protein_ligand_graph_cached(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
        cache_dir=cache_dir,
    )

    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   2.500   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       0.000  30.000   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    updated_batch = load_protein_ligand_graph_cached(
        pdb_path,
        ligand_path,
        cutoff=10.0,
        edge_cutoff=4.5,
        cache_dir=cache_dir,
    )

    assert len(list(cache_dir.glob("*.pt"))) == 2
    assert not torch.allclose(original_batch.positions, updated_batch.positions)
