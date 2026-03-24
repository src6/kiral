from __future__ import annotations

from pathlib import Path

from equidock_diff.data.io import ProteinLigandPaths
from equidock_diff.experiment_panel import (
    DEFAULT_REQUIRED_COMPLEX_IDS,
    PanelSelectionConfig,
    select_panel,
    write_manifest,
)


def _entry(complex_id: str) -> ProteinLigandPaths:
    return ProteinLigandPaths(
        protein_path=Path(f"root/{complex_id}_protein.pdb"),
        ligand_path=Path(f"root/{complex_id}_ligand.sdf"),
    )


def test_select_panel_keeps_required_complexes_first(monkeypatch, tmp_path: Path) -> None:
    config = PanelSelectionConfig(
        root=tmp_path / "data",
        output=tmp_path / "panel.txt",
        target_size=5,
        crop_cutoff=8.0,
        edge_cutoff=4.5,
        smoke_steps=20,
        seed=42,
        device="cpu",
    )
    paths = [_entry("13gs"), _entry("10gs"), _entry("11gs"), _entry("1a30"), _entry("16pk"), _entry("184l")]

    monkeypatch.setattr("equidock_diff.experiment_panel.passes_crop_validation", lambda entry, cutoff: True)
    monkeypatch.setattr("equidock_diff.experiment_panel.ligand_featurizes", lambda entry: True)
    monkeypatch.setattr("equidock_diff.experiment_panel.smoke_run_is_finite", lambda entry, config: True)

    selected = select_panel(paths, config)

    assert [entry.complex_id for entry in selected[: len(DEFAULT_REQUIRED_COMPLEX_IDS)]] == list(
        DEFAULT_REQUIRED_COMPLEX_IDS
    )
    assert len(selected) == 5


def test_write_manifest_emits_comment_header(tmp_path: Path) -> None:
    path = tmp_path / "panel.txt"
    write_manifest(path, [_entry("10gs"), _entry("11gs")])

    text = path.read_text(encoding="utf-8")

    assert "# Fixed dissertation evaluation panel" in text
    assert "10gs" in text
    assert "11gs" in text
