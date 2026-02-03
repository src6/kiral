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

def build_graph_batch(
    node_features: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor | None = None, # ligand mask
) -> GraphBatch:
    if mask is None:
        raise ValueError("build_graph_batch requires a ligand mask.")

    ligand_mask = mask

    # Translation invariance
    centered_pos = center_on_ligand(positions, ligand_mask)

    # Identity encoding
    indicator = ligand_mask.float().unsqueeze(-1)
    x = torch.cat([node_features, indicator], dim=-1)

    # Apply spatial filtering
    crop_mask = crop_protein_by_distance(centered_pos, ligand_mask, cutoff=10.0)
    # Make sure ligand isn't cropped out of its own batch
    final_mask = crop_mask | ligand_mask

    # Filter the edges
    src, dst = edge_index
    edge_keep_mask = final_mask[src] & final_mask[dst]
    filtered_edge_index = edge_index[:, edge_keep_mask]

    # Remap node indices to be contiguous after filtering
    num_nodes = final_mask.size(0)
    device = filtered_edge_index.device

    # Pre-allocate map on the correct device
    idx_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    # Find indices of kept nodes and map them to 0...N_new
    kept_indices = torch.where(final_mask)[0]
    idx_map[kept_indices] = torch.arange(kept_indices.size(0), device=device)

    # Remap edge indices
    new_edge_index = idx_map[filtered_edge_index]

    return GraphBatch(
        node_features=x[final_mask],
        positions=centered_pos[final_mask],
        edge_index=new_edge_index,
        mask=final_mask # New batch mask
    )
