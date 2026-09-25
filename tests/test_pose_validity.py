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

from equidock_diff.pose_validity import evaluate_pose, evaluate_pose_ligand_only  # noqa: E402

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

    from equidock_diff import pose_validity

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
    from equidock_diff import pose_validity

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
