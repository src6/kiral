from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.utils.chemistry import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    ATOM_SYMBOLS,
    ChemicalHealthReport,
    FeaturizeOutcome,
    LigandSkipCounter,
    evaluate_bond_lengths,
    evaluate_chemical_validity,
    evaluate_steric_clashes,
    featurize_ligand,
)


def _rdkit_modules():
    pytest.importorskip("rdkit")
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    return Chem, AllChem


def _write_sdf(path: Path, mol) -> None:
    chem, _ = _rdkit_modules()
    writer = chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_rejects_non_sdf_extension() -> None:
    counter = LigandSkipCounter()
    outcome = featurize_ligand("/tmp/not_a_ligand.mol2", skip_counter=counter)

    assert isinstance(outcome, FeaturizeOutcome)
    assert outcome.skipped
    assert outcome.skip_reason == "parse_failed"
    assert counter.parse_failed == 1


def test_invalid_sdf_is_parse_failed(tmp_path: Path) -> None:
    bad_sdf = tmp_path / "bad.sdf"
    bad_sdf.write_text("this is not valid sdf\n", encoding="utf-8")

    counter = LigandSkipCounter()
    outcome = featurize_ligand(bad_sdf, skip_counter=counter)

    assert outcome.skipped
    assert outcome.skip_reason == "parse_failed"
    assert counter.parse_failed == 1


def test_valid_sdf_tensor_contract(tmp_path: Path) -> None:
    Chem, AllChem = _rdkit_modules()

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    sdf_path = tmp_path / "ethanol.sdf"
    _write_sdf(sdf_path, mol)

    counter = LigandSkipCounter()
    outcome = featurize_ligand(sdf_path, skip_counter=counter)

    assert not outcome.skipped
    assert outcome.graph is not None

    graph = outcome.graph
    assert graph.x.dtype == torch.float32
    assert graph.pos.dtype == torch.float32
    assert graph.edge_index.dtype == torch.int64
    assert graph.edge_attr.dtype == torch.float32

    assert graph.x.shape[1] == ATOM_FEATURE_DIM
    assert graph.pos.shape[1] == 3
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == BOND_FEATURE_DIM
    assert graph.x.shape[0] == graph.pos.shape[0]
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]

    assert torch.isfinite(graph.x).all()
    assert torch.isfinite(graph.pos).all()
    assert torch.isfinite(graph.edge_attr).all()
    assert counter.report() == {
        "parse_failed": 0,
        "sanitize_failed": 0,
        "no_conformer": 0,
        "empty_after_h_removal": 0,
        "no_bonds": 0,
    }


def test_ligand_without_3d_conformer_is_rejected(tmp_path: Path) -> None:
    Chem, _ = _rdkit_modules()

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None

    sdf_path = tmp_path / "ethanol_2d.sdf"
    _write_sdf(sdf_path, mol)

    counter = LigandSkipCounter()
    outcome = featurize_ligand(sdf_path, skip_counter=counter)

    assert outcome.skipped
    assert outcome.skip_reason == "no_conformer"
    assert counter.no_conformer == 1


def test_single_heavy_atom_ligand_is_no_bonds(tmp_path: Path) -> None:
    Chem, AllChem = _rdkit_modules()

    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    sdf_path = tmp_path / "methane.sdf"
    _write_sdf(sdf_path, mol)

    counter = LigandSkipCounter()
    outcome = featurize_ligand(sdf_path, skip_counter=counter)

    assert outcome.skipped
    assert outcome.skip_reason == "no_bonds"
    assert counter.no_bonds == 1


def test_unknown_atom_symbol_maps_to_other_channel(tmp_path: Path) -> None:
    Chem, AllChem = _rdkit_modules()

    mol = Chem.AddHs(Chem.MolFromSmiles("[Xe]C"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    sdf_path = tmp_path / "xenon_methyl.sdf"
    _write_sdf(sdf_path, mol)

    outcome = featurize_ligand(sdf_path)

    assert not outcome.skipped
    assert outcome.graph is not None

    other_atom_index = ATOM_SYMBOLS.index("OTHER")
    assert outcome.graph.x.shape[1] == ATOM_FEATURE_DIM
    assert outcome.graph.x[0, other_atom_index].item() == 1.0


def test_evaluate_steric_clashes_detects_overlap() -> None:
    # 2 protein atoms, 2 ligand atoms
    # Node features: last column is ligand indicator (0.0 for protein, 1.0 for ligand)
    node_features = torch.zeros(4, ATOM_FEATURE_DIM, dtype=torch.float32)
    node_features[:2, -1] = 0.0  # protein
    node_features[2:, -1] = 1.0  # ligand

    # Case 1: Overlapping (clashing) coordinates
    clashing_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # protein 0
            [5.0, 0.0, 0.0],  # protein 1
            [0.4, 0.0, 0.0],  # ligand 2 (overlaps with protein 0, dist 0.4 < 1.7+1.7)
            [10.0, 0.0, 0.0],  # ligand 3 (far away)
        ],
        dtype=torch.float32,
    )
    clash_count, clash_fraction, pairs = evaluate_steric_clashes(clashing_positions, node_features)
    assert clash_count == 1
    assert clash_fraction == 0.5
    assert len(pairs) == 1
    assert pairs[0][0] == 2 and pairs[0][1] == 0

    # Case 2: Well-separated coordinates
    clean_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    clash_count, clash_fraction, pairs = evaluate_steric_clashes(clean_positions, node_features)
    assert clash_count == 0
    assert clash_fraction == 0.0
    assert len(pairs) == 0


def test_evaluate_bond_lengths_detects_distortion() -> None:
    # 3 atoms with 2 bonds: (0, 1) and (1, 2)
    bond_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # Normal bond lengths ~ 1.4 Å
    valid_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]], dtype=torch.float32
    )
    violations, max_dev = evaluate_bond_lengths(valid_positions, bond_index)
    assert violations == 0
    assert max_dev < 0.1

    # Severely stretched bond (0, 1) at 4.0 Å
    stretched_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.4, 0.0, 0.0]], dtype=torch.float32
    )
    violations, max_dev = evaluate_bond_lengths(stretched_positions, bond_index)
    assert violations >= 1
    assert max_dev >= 2.0


def test_evaluate_chemical_validity_report() -> None:
    node_features = torch.zeros(4, ATOM_FEATURE_DIM, dtype=torch.float32)
    node_features[:2, -1] = 0.0  # protein
    node_features[2:, -1] = 1.0  # ligand

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ligand_bonds = torch.tensor([[2, 3], [3, 2]], dtype=torch.long)

    report = evaluate_chemical_validity(positions, node_features, ligand_bonds)
    assert isinstance(report, ChemicalHealthReport)
    assert report.total_atoms == 4
    assert report.ligand_atoms == 2
    assert report.protein_atoms == 2
    assert report.clash_count == 0
    assert report.bond_violation_count == 0
    assert report.is_valid is True
