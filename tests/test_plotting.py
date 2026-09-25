from __future__ import annotations

from pathlib import Path

from equidock_diff.utils.plotting import ensure_mplconfigdir


def test_ensure_mplconfigdir_sets_repo_local_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    path = ensure_mplconfigdir()

    assert path == Path(".mplconfig")
    assert (tmp_path / ".mplconfig").is_dir()
    assert path.as_posix() == ".mplconfig"


def test_ensure_mplconfigdir_respects_existing_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom = tmp_path / "mpl-cache"
    monkeypatch.setenv("MPLCONFIGDIR", str(custom))

    path = ensure_mplconfigdir()

    assert path == custom
    assert custom.is_dir()
