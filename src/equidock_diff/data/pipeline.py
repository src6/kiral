"""Stubs for protein-ligand preprocessing.

These are intentionally minimal to keep core logic for the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from equidock_diff.utils.geometry import batched_centroid, relative_positions


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    mask: torch.Tensor | None = None


def center_on_ligand(
    positions: torch.Tensor, ligand_mask: torch.Tensor
) -> torch.Tensor:
    center = batched_centroid(positions, ligand_mask)
    return relative_positions(positions, center.unsqueeze(-2))


def crop_protein_by_distance(
    positions: torch.Tensor,
    ligand_mask: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    center = batched_centroid(positions, ligand_mask)
    dist = torch.norm(positions - center.unsqueeze(-2), dim=-1)
    return dist <= cutoff
