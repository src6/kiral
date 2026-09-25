"""Pose validity through the reference PoseBusters implementation.

The repository previously hand-rolled an approximation of these checks in ``utils/chemistry.py``.
That approximation is kept for cheap in-loop diagnostics, but it is not a validity verdict: as
measured on 2026-09-25 it reported every *crystal* pose on the dissertation panel as invalid
(clash fractions 0.46-0.58) because hydrogens carried a carbon radius and the threshold rule took
the stricter of two bounds.

PoseBusters (Buttenschoen et al., Chem. Sci. 2024) is the reference implementation used as the
standard validity benchmark for ML docking. It runs ~22 checks across chemical, intramolecular,
intermolecular and volume-overlap families, its verdicts need no reference pose, and it passes the
crystal poses on every one of them. Its pass rate is also the figure the literature quotes, so
numbers taken from here are comparable to published baselines rather than meaningful only locally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = "dock"
_LIGAND_ONLY_CONFIG = "mol"
_NON_CHECK_COLUMNS = ("file", "molecule", "time")


@dataclass(frozen=True)
class PoseValidity:
    """Outcome of the reference validity checks for a single posed ligand."""

    passed: bool
    checks_run: int
    checks_passed: int
    failing_checks: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.passed:
            return f"valid ({self.checks_passed}/{self.checks_run} checks)"
        return (
            f"invalid ({self.checks_passed}/{self.checks_run} checks; "
            f"failing: {', '.join(self.failing_checks)})"
        )


def _buster(config: str):
    try:
        from posebusters import PoseBusters
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "posebusters is not installed; run `uv sync` to restore the validity dependency"
        ) from exc
    return PoseBusters(config=config)


def evaluate_pose(
    sampled_ligand: str | Path,
    true_ligand: str | Path | None = None,
    protein: str | Path | None = None,
    *,
    config: str = DEFAULT_CONFIG,
) -> PoseValidity:
    """Run the reference validity checks on one posed ligand.

    Arguments are file paths rather than tensors on purpose: the checks need real chemistry
    (bond orders, valences, aromaticity), which a node-feature matrix does not carry. The ligand
    may be sdf/mol2/pdb; ``protein`` is required for the docking configuration and omitted for the
    ligand-only one. Passing ``true_ligand`` enables the redocking checks that need a reference.
    """
    kwargs: dict[str, Path] = {"mol_pred": Path(sampled_ligand)}
    if true_ligand is not None:
        kwargs["mol_true"] = Path(true_ligand)
    if protein is not None:
        kwargs["mol_cond"] = Path(protein)

    frame = _buster(config).bust(**kwargs)
    checks = [column for column in frame.columns if column not in _NON_CHECK_COLUMNS]
    failing = tuple(column for column in checks if not bool(frame[column].all()))
    return PoseValidity(
        passed=not failing,
        checks_run=len(checks),
        checks_passed=len(checks) - len(failing),
        failing_checks=failing,
    )


def evaluate_pose_ligand_only(sampled_ligand: str | Path) -> PoseValidity:
    """Chemistry-only verdict, usable without a receptor or a reference pose."""
    return evaluate_pose(sampled_ligand, config=_LIGAND_ONLY_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check posed ligands against the PoseBusters reference criteria",
    )
    parser.add_argument("--sampled", type=Path, required=True, help="Posed ligand file")
    parser.add_argument(
        "--true",
        type=Path,
        default=None,
        help="Reference ligand file; omit for a ligand-only chemistry verdict",
    )
    parser.add_argument("--protein", type=Path, default=None, help="Receptor file")
    parser.add_argument("--config", default=None, help="PoseBusters configuration name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        config = args.config
    else:
        config = DEFAULT_CONFIG if args.protein is not None else _LIGAND_ONLY_CONFIG
    validity = evaluate_pose(args.sampled, args.true, args.protein, config=config)
    print(f"config={config}")
    print(f"posebusters={validity.describe()}")
    return 0 if validity.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
