"""Tests for the held-out validation-loss instrument.

Aggregation and CSV formatting are pure functions and are tested directly. The wiring into ``main``
is tested on a two-complex toy dataset built in ``tmp_path`` — a self-contained fixture, so the
suite stays green on a machine without PDBbind and without a GPU.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from equidock_diff import train as train_module
from equidock_diff.train import (
    build_parser,
    build_validation_examples,
    evaluate_validation_loss,
    main,
    validation_mode_enabled,
)
from equidock_diff.utils.artifacts import (
    LOSS_TERM_COLUMNS,
    mean_loss_terms,
    write_loss_csv,
    write_loss_terms_csv,
    write_validation_loss_csv,
)

VALIDATION_HEADER = ["step", "loss", "score", "bond", "shape", "clash"]
DATA_ROOT_CANDIDATES = (Path("data/pdbbind_v2020"), Path.home() / "data" / "pdbbind_v2020")
PANEL_SUBDIR = Path("protein_ligand_general_minus_refined") / "1981-2000"

TOY_PROTEIN_ATOMS = (
    "ATOM      1  N   GLY A   1       0.000   4.000   0.000  1.00  0.00           N",
    "ATOM      2  CA  GLY A   1       0.000   6.000   0.000  1.00  0.00           C",
)


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_mean_loss_terms_averages_every_column() -> None:
    rows = [(1.0, 3.0, 0.0, 0.0, 0.0), (3.0, 5.0, 2.0, 1.0, 4.0)]

    means = mean_loss_terms(rows)

    assert means == pytest.approx((2.0, 4.0, 1.0, 0.5, 2.0))


def test_mean_loss_terms_rejects_an_empty_pass() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        mean_loss_terms([])


def test_mean_loss_terms_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="same number of loss terms"):
        mean_loss_terms([(1.0, 2.0), (1.0,)])


def test_write_validation_loss_csv_writes_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "validation_loss.csv"

    write_validation_loss_csv(path, [(20, 2.0, 1.0, 0.5, 0.25, 0.125)])

    rows = _read_rows(path)
    assert rows[0] == VALIDATION_HEADER
    assert rows[1] == ["20", "2.0", "1.0", "0.5", "0.25", "0.125"]


def test_validation_header_is_built_from_the_term_columns() -> None:
    """The header a reader opens is the same list the run log prints its terms from."""
    assert ["step", *LOSS_TERM_COLUMNS] == VALIDATION_HEADER


def test_existing_loss_csv_headers_are_unchanged(tmp_path: Path) -> None:
    """The instrument is additive: the columns existing consumers read must not move."""
    write_loss_csv(tmp_path / "loss_trace.csv", [(1, 0.5, 0.25)])
    write_loss_terms_csv(tmp_path / "loss_terms.csv", [(1, 0.1, 0.2, 0.3, 0.4)])

    assert _read_rows(tmp_path / "loss_trace.csv")[0] == ["step", "loss", "beta_t"]
    assert _read_rows(tmp_path / "loss_terms.csv")[0] == ["step", "score", "bond", "shape", "clash"]


def test_parser_validation_flags_default_to_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.validation_split is None
    assert args.validation_every == 0
    assert not validation_mode_enabled(args)


def test_validation_mode_requires_both_the_split_and_an_interval(tmp_path: Path) -> None:
    parser = build_parser()
    split = tmp_path / "val.txt"

    assert not validation_mode_enabled(parser.parse_args(["--validation-split", str(split)]))
    assert not validation_mode_enabled(parser.parse_args(["--validation-every", "10"]))
    assert validation_mode_enabled(
        parser.parse_args(["--validation-split", str(split), "--validation-every", "10"])
    )


def test_main_rejects_a_validation_split_that_would_never_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --validation-every"):
        main(["--validation-split", str(tmp_path / "val.txt")])


def test_main_rejects_a_negative_validation_interval() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        main(["--validation-every", "-1"])


def test_main_rejects_an_empty_validation_split(tmp_path: Path) -> None:
    """A split file that selects nothing is an error, not an empty evaluation."""
    root = tmp_path / "pdbbind_v2020"
    complex_dir = root / PANEL_SUBDIR / "10gs"
    complex_dir.mkdir(parents=True)
    (complex_dir / "10gs_protein.pdb").write_text("", encoding="utf-8")
    (complex_dir / "10gs_ligand.sdf").write_text("", encoding="utf-8")
    (tmp_path / "train.txt").write_text("10gs\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("# nothing selected yet\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No validation complexes found"):
        main(
            [
                "--validation-split",
                str(tmp_path / "val.txt"),
                "--validation-every",
                "5",
                "--dataset-root",
                str(root),
                "--dataset-split",
                str(tmp_path / "train.txt"),
                "--batch-size",
                "1",
                "--device",
                "cpu",
            ]
        )


class _StubExample:
    def __init__(self, complex_id: str) -> None:
        self.complex_id = complex_id


def test_evaluate_validation_loss_means_each_term_over_the_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """One complex at a time, no gradients, and the reported terms are the mean of the rows."""
    per_complex = {
        "10gs": (2.0, 4.0, 0.0, 0.0, 0.0),
        "11gs": (4.0, 6.0, 1.0, 2.0, 3.0),
        "1a30": (6.0, 8.0, 2.0, 4.0, 6.0),
    }
    examples = [_StubExample(complex_id) for complex_id in per_complex]
    loaded: list[str] = []
    grad_flags: list[bool] = []

    def fake_load(example, args, device):
        loaded.append(example.complex_id)
        return (torch.zeros(1, 3),) * 6

    def fake_step(model, node_features, positions, edge_index, ligand_bond_index, beta_min, beta_max, **kwargs):
        grad_flags.append(torch.is_grad_enabled())
        total, score, bond, shape, clash = per_complex[loaded[-1]]
        return (
            torch.tensor(total),
            0.5,
            torch.tensor(score),
            torch.tensor(bond),
            torch.tensor(shape),
            torch.tensor(clash),
            torch.tensor(0.0),
        )

    monkeypatch.setattr(train_module, "load_dataset_example", fake_load)
    monkeypatch.setattr(train_module, "training_step_with_breakdown", fake_step)

    args = build_parser().parse_args(["--device", "cpu"])
    means = evaluate_validation_loss(
        model=torch.nn.Identity(),
        validation_examples=examples,
        args=args,
        device=torch.device("cpu"),
    )

    assert means == pytest.approx((4.0, 6.0, 1.0, 2.0, 3.0))
    assert loaded == ["10gs", "11gs", "1a30"]
    assert grad_flags == [False, False, False]


def test_evaluate_validation_loss_rejects_an_empty_split() -> None:
    args = build_parser().parse_args(["--device", "cpu"])

    with pytest.raises(ValueError, match="at least one row"):
        evaluate_validation_loss(
            model=torch.nn.Identity(),
            validation_examples=[],
            args=args,
            device=torch.device("cpu"),
        )


def test_build_validation_examples_ignores_the_training_limit() -> None:
    """--dataset-limit exists to shorten a training run; it must not shrink the held-out set."""
    root = next((candidate for candidate in DATA_ROOT_CANDIDATES if candidate.is_dir()), None)
    if root is None:
        pytest.skip(f"no PDBbind dataset at {DATA_ROOT_CANDIDATES}")

    split = Path("config/evaluation/dissertation_panel20.txt")
    if not split.is_file():
        pytest.skip(f"{split} is not available")
    expected = [
        line.split()[0]
        for line in split.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    args = build_parser().parse_args(
        [
            "--dataset-root",
            str(root),
            "--dataset-limit",
            "1",
            "--validation-split",
            str(split),
            "--validation-every",
            "10",
        ]
    )

    examples = build_validation_examples(args)

    assert [example.complex_id for example in examples] == expected
    assert all(example.protein_path.is_file() for example in examples)
    assert all(example.ligand_path.is_file() for example in examples)


def _write_toy_complex(complex_dir: Path, complex_id: str) -> None:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    complex_dir.mkdir(parents=True, exist_ok=True)
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert molecule is not None
    assert AllChem.EmbedMolecule(molecule, randomSeed=0) == 0
    writer = Chem.SDWriter(str(complex_dir / f"{complex_id}_ligand.sdf"))
    writer.write(molecule)
    writer.close()
    (complex_dir / f"{complex_id}_protein.pdb").write_text(
        "\n".join(TOY_PROTEIN_ATOMS) + "\n",
        encoding="utf-8",
    )


def test_main_writes_a_held_out_validation_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("rdkit")

    root = tmp_path / "pdbbind_v2020"
    _write_toy_complex(root / PANEL_SUBDIR / "10gs", "10gs")
    _write_toy_complex(root / PANEL_SUBDIR / "11gs", "11gs")
    (tmp_path / "train.txt").write_text("10gs\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("11gs\n", encoding="utf-8")
    loss_csv = tmp_path / "loss_trace.csv"

    exit_code = main(
        [
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--steps",
            "1",
            "--dataset-root",
            str(root),
            "--dataset-split",
            str(tmp_path / "train.txt"),
            "--dataset-cache-dir",
            str(tmp_path / "cache"),
            "--validation-split",
            str(tmp_path / "val.txt"),
            "--validation-every",
            "1",
            "--loss-csv",
            str(loss_csv),
            "--sample-steps",
            "1",
            "--skip-pose-artifacts",
            "--skip-plot",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "validation_size=1" in captured
    assert "validation step=1" in captured
    assert "complexes=1" in captured
    assert f"validation_loss_csv={loss_csv.with_name('validation_loss.csv')}" in captured

    rows = _read_rows(loss_csv.with_name("validation_loss.csv"))
    assert rows[0] == VALIDATION_HEADER
    assert len(rows) == 2
    assert rows[1][0] == "1"
    assert all(float(value) >= 0.0 for value in rows[1][1:])


def test_main_notes_that_validation_never_ran(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Enabled but unreached is reported, not left as a silently missing file."""
    pytest.importorskip("rdkit")

    root = tmp_path / "pdbbind_v2020"
    _write_toy_complex(root / PANEL_SUBDIR / "10gs", "10gs")
    _write_toy_complex(root / PANEL_SUBDIR / "11gs", "11gs")
    (tmp_path / "train.txt").write_text("10gs\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("11gs\n", encoding="utf-8")
    loss_csv = tmp_path / "loss_trace.csv"

    exit_code = main(
        [
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--steps",
            "1",
            "--dataset-root",
            str(root),
            "--dataset-split",
            str(tmp_path / "train.txt"),
            "--dataset-cache-dir",
            str(tmp_path / "cache"),
            "--validation-split",
            str(tmp_path / "val.txt"),
            "--validation-every",
            "50",
            "--loss-csv",
            str(loss_csv),
            "--sample-steps",
            "1",
            "--skip-pose-artifacts",
            "--skip-plot",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "validation_size=1" in captured
    assert "validation_loss=not_evaluated" in captured
    assert not loss_csv.with_name("validation_loss.csv").exists()
