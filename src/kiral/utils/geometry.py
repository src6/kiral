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


def kabsch_align(
    mobile: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Align mobile onto target with Kabsch.

    MPS fallback: SVD is not implemented efficiently on MPS for this path, so
    perform the small alignment solve on CPU and move the aligned coordinates
    back to the original device.
    """
    if mobile.shape != target.shape or mobile.dim() != 2 or mobile.size(-1) != 3:
        raise ValueError("mobile and target must both have shape [N, 3].")

    solve_device = torch.device("cpu") if mobile.device.type == "mps" else mobile.device
    mobile_solve = mobile.to(solve_device)
    target_solve = target.to(solve_device)

    mobile_center = mobile_solve.mean(dim=0, keepdim=True)
    target_center = target_solve.mean(dim=0, keepdim=True)
    mobile_centered = mobile_solve - mobile_center
    target_centered = target_solve - target_center

    covariance = mobile_centered.transpose(0, 1) @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    reflection = torch.sign(torch.linalg.det(u @ vh)).item()
    correction = torch.diag(
        torch.tensor([1.0, 1.0, reflection], device=solve_device, dtype=mobile.dtype)
    )
    rotation = u @ correction @ vh
    aligned = mobile_centered @ rotation + target_center
    return aligned.to(mobile.device)


def aligned_rmsd(
    mobile: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    aligned = kabsch_align(mobile, target)
    return torch.sqrt(torch.mean((aligned - target) ** 2))
