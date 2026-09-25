"""Hardware acceleration and mixed-precision (AMP) utilities."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def get_autocast_context(
    device: torch.device,
    enabled: bool = False,
    dtype: torch.dtype | None = None,
) -> contextlib.AbstractContextManager[Any]:
    """Return an appropriate torch.autocast context manager for the given device.

    On CUDA (e.g. RTX 3080 Ti), uses BF16 by default when supported to saturate Tensor Cores.
    On CPU, uses BF16 if requested.
    On MPS or when disabled, returns a nullcontext.
    """
    if not enabled:
        return contextlib.nullcontext()

    dev_type = device.type
    if dev_type == "cuda":
        target_dtype = dtype
        if target_dtype is None:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=target_dtype)

    if dev_type == "cpu":
        target_dtype = dtype or torch.bfloat16
        return torch.autocast(device_type="cpu", dtype=target_dtype)

    # MPS / others fallback
    return contextlib.nullcontext()


def maybe_compile_model(
    model: nn.Module,
    enabled: bool = False,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Optionally apply torch.compile kernel fusion to the model."""
    if not enabled:
        return model

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile is not available in this PyTorch version.")
        return model

    try:
        compiled = torch.compile(model, mode=mode)
        return compiled
    except Exception as exc:
        logger.warning("torch.compile failed (%s); using uncompiled model.", exc)
        return model


def get_device_benchmark_info(device: torch.device) -> dict[str, Any]:
    """Return a diagnostic summary of device acceleration features."""
    info: dict[str, Any] = {
        "device": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": False,
        "compile_supported": hasattr(torch, "compile"),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(device)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["vram_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        info["vram_total_mb"] = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    elif device.type == "mps" and hasattr(torch.backends, "mps"):
        info["device_name"] = "Apple Silicon (MPS)"
        info["mps_available"] = torch.backends.mps.is_available()
    else:
        info["device_name"] = "Host CPU"

    return info
