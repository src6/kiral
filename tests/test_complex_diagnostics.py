from __future__ import annotations

from pathlib import Path

from equidock_diff.complex_diagnostics import compute_row, main
from equidock_diff.data.io import ProteinLigandPaths


def _write_toy_sdf(path: Path) -> None:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        path.write_text(
            "\n".join(
                [
                    "toy",
                    "  codex",
                    "",
                    "  2  1  0  0  0  0            999 V2000",
                    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                    "    1.2000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
                    "  1  2  1  0  0  0  0",
                    "M  END",
                    "$$$$",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return

    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _write_toy_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   3.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       0.000   4.500   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_compute_row_extracts_expected_descriptors(tmp_path: Path) -> None:
    complex_dir = tmp_path / "10gs"
    complex_dir.mkdir()
    protein = complex_dir / "10gs_protein.pdb"
    ligand = complex_dir / "10gs_ligand.sdf"
    _write_toy_pdb(protein)
    _write_toy_sdf(ligand)

    row = compute_row(ProteinLigandPaths(protein_path=protein, ligand_path=ligand))

    assert row.complex_id == "10gs"
    assert row.ligand_heavy_atoms == 2
    assert row.ligand_bond_count == 1
    assert row.ligand_max_degree >= 1


def test_main_writes_focus_and_comparison_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "pdbbind_v2020" / "protein_ligand_general_minus_refined" / "1981-2000"
    for complex_id in ("10gs", "11gs"):
        complex_dir = dataset_root / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)
        _write_toy_pdb(complex_dir / f"{complex_id}_protein.pdb")
        _write_toy_sdf(complex_dir / f"{complex_id}_ligand.sdf")

    focus_manifest = tmp_path / "focus.txt"
    focus_manifest.write_text("10gs\n", encoding="utf-8")
    comparison_manifest = tmp_path / "comparison.txt"
    comparison_manifest.write_text("11gs\n", encoding="utf-8")
    csv_path = tmp_path / "diag.csv"
    markdown_path = tmp_path / "diag.md"

    result = main(
        [
            "--dataset-root",
            str(tmp_path / "pdbbind_v2020"),
            "--manifest",
            str(focus_manifest),
            "--comparison-manifest",
            str(comparison_manifest),
            "--output-csv",
            str(csv_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert result == 0
    assert csv_path.exists()
    assert markdown_path.exists()
    assert "Hard-Case Rows" in markdown_path.read_text(encoding="utf-8")
