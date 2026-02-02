from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.utils.chemistry import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    FeaturizeOutcome,
    LigandSkipCounter,
    featurize_ligand,
)


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
    rdkit = pytest.importorskip("rdkit")
    assert rdkit is not None

    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0

    sdf_path = tmp_path / "ethanol.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

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
