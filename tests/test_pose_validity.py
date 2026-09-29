"""Tests for the reference PoseBusters validity path.

These replace the hand-rolled gate guard: the acceptance criterion is that the *crystal* poses on
the dissertation panel pass, which the homegrown approximation failed (clash fractions 0.46-0.58).
"""

from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("posebusters")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from kiral.pose_validity import evaluate_pose, evaluate_pose_ligand_only  # noqa: E402

PANEL = ("10gs", "11gs", "1a30")
DATA_ROOT_CANDIDATES = (Path("data/pdbbind_v2020"), Path.home() / "data" / "pdbbind_v2020")
PANEL_SUBDIR = Path("protein_ligand_general_minus_refined") / "1981-2000"


def _panel_dir() -> Path | None:
    for candidate in DATA_ROOT_CANDIDATES:
        panel = candidate / PANEL_SUBDIR
        if panel.is_dir():
            return panel
    return None


def _butane_with_a_stretched_bond(path: Path, *, stretch: float) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(molecule, randomSeed=42)
    conformer = molecule.GetConformer()
    first, second = conformer.GetAtomPosition(0), conformer.GetAtomPosition(1)
    direction = (second - first)
    direction = direction / direction.Length()
    conformer.SetAtomPosition(1, first + direction * (first.Distance(second) + stretch))
    Chem.MolToMolFile(molecule, str(path))


def test_a_clean_ligand_passes(tmp_path):
    path = tmp_path / "clean.sdf"
    _butane_with_a_stretched_bond(path, stretch=0.0)

    assert evaluate_pose_ligand_only(path).passed


def test_a_stretched_bond_is_rejected(tmp_path):
    path = tmp_path / "stretched.sdf"
    _butane_with_a_stretched_bond(path, stretch=0.6)

    validity = evaluate_pose_ligand_only(path)

    assert not validity.passed
    assert any("bond" in check.lower() for check in validity.failing_checks), validity.failing_checks


def test_crystal_poses_pass_the_reference_checks(tmp_path):
    """The acceptance criterion: ground truth must satisfy the gate."""
    panel = _panel_dir()
    if panel is None:
        pytest.skip(f"no PDBbind dataset at {DATA_ROOT_CANDIDATES}")

    for complex_id in PANEL:
        ligand = panel / complex_id / f"{complex_id}_ligand.sdf"
        protein = panel / complex_id / f"{complex_id}_pocket.pdb"
        if not ligand.exists() or not protein.exists():
            pytest.skip(f"{complex_id} missing from {panel}")

        validity = evaluate_pose(ligand, ligand, protein)

        assert validity.passed, f"{complex_id}: crystal pose rejected ({validity.describe()})"
        assert validity.checks_run > 10, "the docking configuration should run the full check set"


def test_a_scrambled_pose_is_rejected(tmp_path):
    """The other side of the calibration: the check must still discriminate."""
    panel = _panel_dir()
    if panel is None:
        pytest.skip(f"no PDBbind dataset at {DATA_ROOT_CANDIDATES}")

    complex_id = "10gs"
    ligand = panel / complex_id / f"{complex_id}_ligand.sdf"
    protein = panel / complex_id / f"{complex_id}_pocket.pdb"
    if not ligand.exists() or not protein.exists():
        pytest.skip(f"{complex_id} missing from {panel}")

    molecule = Chem.MolFromMolFile(str(ligand), removeHs=False)
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (index * 0.7, index * 0.3, index * 0.5))
    scrambled = tmp_path / "scrambled.sdf"
    writer = Chem.SDWriter(str(scrambled))
    writer.write(molecule)
    writer.close()

    assert not evaluate_pose(scrambled, ligand, protein).passed


def test_a_reference_ligand_selects_the_redocking_configuration(monkeypatch, tmp_path):
    """Passing a reference must actually enable the reference-dependent checks."""
    import pandas as pd

    from kiral import pose_validity

    seen: dict[str, str] = {}

    class StubBuster:
        def bust(self, **_kwargs):
            return pd.DataFrame({"file": ["x"], "molecule": ["y"], "Bond lengths": [True]})

    def fake_buster(config):
        seen["config"] = config
        return StubBuster()

    monkeypatch.setattr(pose_validity, "_buster", fake_buster)
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")

    assert pose_validity.evaluate_pose(ligand).passed
    assert seen["config"] == "dock"

    assert pose_validity.evaluate_pose(ligand, ligand, ligand).passed
    assert seen["config"] == "redock"


def test_validate_cli_propagates_the_verdict(monkeypatch, tmp_path):
    """The command must exit non-zero on an invalid pose so scripts can gate on it."""
    from kiral import pose_validity

    ligand = tmp_path / "ligand.sdf"
    ligand.write_text("")

    monkeypatch.setattr(
        pose_validity, "evaluate_pose", lambda *a, **k: pose_validity.PoseValidity(False, 22, 20, ("Bond lengths",))
    )
    assert pose_validity.main(["--sampled", str(ligand)]) == 1

    monkeypatch.setattr(
        pose_validity, "evaluate_pose", lambda *a, **k: pose_validity.PoseValidity(True, 22, 22, ())
    )
    assert pose_validity.main(["--sampled", str(ligand)]) == 0


def test_write_posed_ligand_refuses_when_the_mapping_cannot_be_verified(tmp_path):
    """A mis-paired pose would give plausible numbers from the wrong atoms: refuse instead."""
    panel = _panel_dir()
    if panel is None:
        pytest.skip(f"no PDBbind dataset at {DATA_ROOT_CANDIDATES}")
    import torch

    from kiral.pose_validity import write_posed_ligand

    ligand = panel / "10gs" / "10gs_ligand.sdf"
    if not ligand.exists():
        pytest.skip(f"10gs missing from {panel}")

    molecule = Chem.RemoveHs(Chem.MolFromMolFile(str(ligand), removeHs=False))
    count = molecule.GetNumAtoms()
    conformer = molecule.GetConformer()
    file_positions = torch.tensor(
        [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
         for i in range(count)],
        dtype=torch.float64,
    )
    assert write_posed_ligand(ligand, file_positions, file_positions, tmp_path / "ok.sdf")

    permuted = file_positions[torch.roll(torch.arange(count), 1)]
    assert not write_posed_ligand(ligand, file_positions, permuted, tmp_path / "bad.sdf")
    assert not (tmp_path / "bad.sdf").exists()


def test_unrunnable_check_is_excluded_from_denominator_and_does_not_fail(tmp_path):
    """When a PoseBusters check cannot run (e.g. UFF parameters missing for Fe2+),
    it does not count as a failure and is excluded from the denominator (checks_run).
    """
    # Create ferrocene or a simple iron complex: Fe2+ with no UFF parameters
    metal_mol = Chem.MolFromSmiles("[Fe+2]")
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    metal_mol.AddConformer(conf)
    metal_sdf = tmp_path / "iron.sdf"
    writer = Chem.SDWriter(str(metal_sdf))
    writer.write(metal_mol)
    writer.close()

    validity = evaluate_pose_ligand_only(metal_sdf)
    # Internal energy cannot run because UFF lacks Fe2+ parameters.
    # It should pass (not fail the gate), and the skipped check should be excluded from denominator.
    assert validity.passed, f"Skipped checks must not cause gate failure: {validity.describe()}"
    assert "internal_energy" not in validity.failing_checks
    # In 'mol' config, 12 checks total exist; with internal_energy NaN/skipped, exactly 11 run.
    assert validity.checks_run == 11
    assert validity.checks_passed == 11
