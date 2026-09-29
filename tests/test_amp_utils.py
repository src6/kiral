from __future__ import annotations

import contextlib
import pytest
import torch
from torch import nn

from kiral.models.amp_utils import (
    get_autocast_context,
    get_device_benchmark_info,
    maybe_compile_model,
)


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def test_autocast_context_disabled_returns_nullcontext() -> None:
    device = torch.device("cpu")
    ctx = get_autocast_context(device, enabled=False)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_context_cpu_enabled() -> None:
    device = torch.device("cpu")
    ctx = get_autocast_context(device, enabled=True, dtype=torch.bfloat16)
    # Context manager enters and exits cleanly
    with ctx:
        x = torch.randn(2, 4)
        m = SimpleLinear()
        out = m(x)
        assert out.shape == (2, 4)


def test_maybe_compile_model_disabled_returns_identity() -> None:
    m = SimpleLinear()
    compiled = maybe_compile_model(m, enabled=False)
    assert compiled is m


def test_get_device_benchmark_info_keys() -> None:
    device = torch.device("cpu")
    info = get_device_benchmark_info(device)
    assert "device" in info
    assert "type" in info
    assert "cuda_available" in info
    assert "device_name" in info
    assert info["type"] == "cpu"
