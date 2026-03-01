from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.utils.chemistry import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    ATOM_SYMBOLS,
    FeaturizeOutcome,
    LigandSkipCounter,
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
