from __future__ import annotations

import pytest

from equidock_diff.cli import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "kiral" in captured.out
    assert "dock" in captured.out
    assert "resample" in captured.out
    assert "diagnostics" in captured.out


def test_cli_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["diagnostics", "--device", "cpu"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Kiral Environment Diagnostics" in captured.out
    assert "PyTorch Version" in captured.out
    assert "Resolved Device:      cpu" in captured.out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "kiral 0.2.0" in captured.out
