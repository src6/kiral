"""SE(3) geometry helpers."""

from __future__ import annotations

import torch


def random_rotation_matrix(
    batch_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Sample random rotation matrices using QR decomposition.

    MPS fallback: QR is not implemented on MPS, so generate on CPU and move.
    """
    cpu_device = torch.device("cpu") if device.type == "mps" else device
    a = torch.randn(batch_size, 3, 3, device=cpu_device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * d.unsqueeze(-2)
    det = torch.linalg.det(q)
    correction = torch.tensor([-1.0, 1.0, 1.0], device=cpu_device, dtype=dtype)
    q = torch.where(det.view(-1, 1, 1) < 0, q * correction, q)
    return q.to(device)


def apply_rigid_transform(
    pos: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor
) -> torch.Tensor:
    """Apply batched SE(3) transform to positions.

    pos: [N, 3] or [B, N, 3]
    rot: [3, 3] or [B, 3, 3]
    trans: [3] or [B, 3]
    """
    if pos.dim() == 2:
        return pos @ rot.T + trans
    return torch.matmul(pos, rot.transpose(-1, -2)) + trans.unsqueeze(-2)


def relative_positions(pos: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Center positions to preserve translation equivariance."""
    return pos - center


def batched_centroid(
    pos: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute centroid for [B, N, 3] positions with optional mask."""
    if mask is None:
        return pos.mean(dim=-2)
    weights = mask.unsqueeze(-1).to(pos.dtype)
    denom = weights.sum(dim=-2).clamp_min(1.0)
    return (pos * weights).sum(dim=-2) / denom
