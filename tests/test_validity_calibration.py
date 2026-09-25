"""Calibration guard for the ligand-protein clash criterion.

A validity gate must not flag ground-truth crystal poses. If it does, it cannot
discriminate a good pose from a bad one, and any validity metric derived from it is noise
that invites chasing phantom geometry problems.

Measured 2026-09-25 through the repo pipeline on three panel complexes: the *crystal* pose
reports clash fractions of 0.545 (10gs), 0.462 (11gs) and 0.577 (1a30), so ``is_valid`` is
False for all three. The criterion flags anything below ``max(0.75 * sum_radii,
sum_radii - 0.2)`` -- for a carbon pair 3.2 A -- which a real binding interface violates
routinely because hydrogen bonds at 2.8-3.0 A and tight van der Waals packing are
attractive contacts, not clashes.

These tests document the defect. They skip when the PDBbind dataset is absent, which is the
case in CI, and xfail where the data is present. Remove the xfail markers once the
criterion handles attractive contacts.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from equidock_diff.data.io import load_paths  # noqa: E402
from equidock_diff.train import load_dataset_example  # noqa: E402
from equidock_diff.utils.chemistry import evaluate_chemical_validity  # noqa: E402

DATA_ROOT = Path.home() / "data" / "pdbbind_v2020"
PANEL = ("10gs", "11gs", "1a30")


def _pipeline_args() -> SimpleNamespace:
    return SimpleNamespace(
        context_policy="fixed",
        crop_cutoff=10.0,
        edge_cutoff=4.5,
        protein_node_budget=256,
        dataset_cache_dir=Path("data/.cache/equidock_diff_graphs"),
    )


@pytest.fixture(scope="module")
def loaded():
    if not DATA_ROOT.exists():
        pytest.skip(f"no PDBbind dataset at {DATA_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {p.complex_id: p for p in load_paths(DATA_ROOT)}
    graphs = {}
    for complex_id in PANEL:
        if complex_id not in paths:
            pytest.skip(f"{complex_id} not in the local dataset")
        features, positions, edge_index, bond_index, _cutoff, _retained = load_dataset_example(
            paths[complex_id], _pipeline_args(), device
        )
        graphs[complex_id] = (features, positions, edge_index, bond_index)
    return device, graphs


@pytest.mark.xfail(
    reason="gate flags crystal poses (clash fractions 0.46-0.58 measured 2026-09-25)",
    strict=False,
)
def test_crystal_pose_passes_the_validity_gate(loaded):
    device, graphs = loaded
    for complex_id, (features, positions, _edge_index, bond_index) in graphs.items():
        report = evaluate_chemical_validity(
            positions, features, ligand_bond_index=bond_index, reference_positions=positions
        )
        assert report.is_valid, (
            f"{complex_id}: the crystal pose is reported invalid "
            f"(clash fraction {report.clash_fraction:.3f}, {report.clash_count} clashes)"
        )


def test_crystal_pose_has_no_bond_violations(loaded):
    """This half of the gate is correctly calibrated: a crystal pose has no bond defects."""
    device, graphs = loaded
    for complex_id, (features, positions, _edge_index, bond_index) in graphs.items():
        report = evaluate_chemical_validity(
            positions, features, ligand_bond_index=bond_index, reference_positions=positions
        )
        assert report.bond_violation_count == 0
        assert report.max_bond_deviation == pytest.approx(0.0, abs=1e-6)
