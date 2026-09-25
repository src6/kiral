"""Tests for the batched serving engine's configuration resolution."""

from types import SimpleNamespace

import pytest

from equidock_diff.serve import resolve_model_config


def _cli(**overrides):
    base = {"hidden_dim": 64, "num_layers": 3}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_checkpoint_config_overrides_cli_defaults():
    saved = {
        "hidden_dim": 128,
        "num_layers": 5,
        "hetero_edges": True,
        "frame_hetero_backbone": False,
    }

    config = resolve_model_config(_cli(), saved)

    assert config.hidden_dim == 128
    assert config.num_layers == 5
    assert config.hetero_edges is True
    assert config.frame_hetero_backbone is False


def test_cli_values_are_the_fallback_without_a_checkpoint():
    config = resolve_model_config(_cli(hidden_dim=32, num_layers=2), None)

    assert config.hidden_dim == 32
    assert config.num_layers == 2
    # the dissertation's frame arm is the serving default
    assert config.frame_hetero_backbone is True
    assert config.hetero_edges is False


def test_partial_saved_args_keep_the_rest_of_the_cli_config():
    config = resolve_model_config(_cli(), {"hidden_dim": 16})

    assert config.hidden_dim == 16
    assert config.num_layers == 3
    assert config.frame_hetero_backbone is True


def test_unrelated_saved_keys_are_ignored():
    config = resolve_model_config(_cli(), {"sample_steps": 12, "noise_schedule": "linear"})

    assert config.hidden_dim == 64
    assert config.num_layers == 3


def test_serve_rejects_a_checkpoint_without_saved_args(tmp_path):
    """Guessing an architecture silently is worse than refusing to serve."""
    torch = pytest.importorskip("torch")
    from equidock_diff.serve import DockingEngine, build_parser

    checkpoint = tmp_path / "no_saved_args.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {}, "completed_steps": 0}, checkpoint)
    args = build_parser().parse_args(["--checkpoint", str(checkpoint), "--allow-random-weights"])

    with pytest.raises(SystemExit, match="saved_args"):
        DockingEngine(args)
