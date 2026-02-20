from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from equidock_diff.utils.chemistry import (
    ATOM_FEATURE_DIM,
    ATOM_SYMBOL_TO_INDEX,
    FeaturizeOutcome,
    featurize_ligand,
)
from equidock_diff.utils.geometry import batched_centroid, relative_positions


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    mask: torch.Tensor | None = None


def _normalize_element(symbol: str) -> str:
    cleaned = "".join(ch for ch in symbol.strip() if ch.isalpha())
    if not cleaned:
        return "OTHER"
    if len(cleaned) >= 2:
        candidate = cleaned[0].upper() + cleaned[1].lower()
        if candidate in ATOM_SYMBOL_TO_INDEX:
            return candidate
    candidate = cleaned[0].upper()
    if candidate in ATOM_SYMBOL_TO_INDEX:
        return candidate
    return "OTHER"


def _infer_pdb_element(line: str) -> str:
    element = _normalize_element(line[76:78])
    if element != "OTHER":
        return element
    return _normalize_element(line[12:16])


def build_complete_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            edges.append((src, dst))
    if not edges:
        return torch.empty((2, 0), device=device, dtype=torch.long)
    return torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()


def build_radius_edge_index(
    positions: torch.Tensor,
    *,
    cutoff: float,
) -> torch.Tensor:
    if positions.dim() != 2 or positions.size(-1) != 3:
        raise ValueError("positions must have shape [N, 3].")
    if positions.size(0) == 0:
        return torch.empty((2, 0), device=positions.device, dtype=torch.long)

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)
    keep = (dist <= cutoff) & ~torch.eye(
        positions.size(0),
        device=positions.device,
        dtype=torch.bool,
    )
    edge_index = keep.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        return torch.empty((2, 0), device=positions.device, dtype=torch.long)
    return edge_index


def load_protein_graph(pdb_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(pdb_path)
    coords: list[list[float]] = []
    symbols: list[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            coords.append([x, y, z])
            symbols.append(_infer_pdb_element(line))

    if not coords:
        raise ValueError(f"No atom coordinates found in {path}.")

    node_features = torch.zeros((len(coords), ATOM_FEATURE_DIM), dtype=torch.float32)
    other_atom_index = len(ATOM_SYMBOL_TO_INDEX)
    for atom_idx, symbol in enumerate(symbols):
        feature_index = ATOM_SYMBOL_TO_INDEX.get(symbol, other_atom_index)
        node_features[atom_idx, feature_index] = 1.0

    positions = torch.tensor(coords, dtype=torch.float32)
    return node_features, positions


def _require_ligand_graph(outcome: FeaturizeOutcome, ligand_path: Path) -> torch.Tensor:
    if outcome.skipped or outcome.graph is None:
        reason = outcome.skip_reason or "unknown"
        raise ValueError(f"Unable to featurize ligand {ligand_path}: {reason}.")
    return outcome.graph


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
    cutoff: float = 10.0,
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
    crop_mask = crop_protein_by_distance(centered_pos, ligand_mask, cutoff=cutoff)
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


def load_protein_ligand_graph(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float = 10.0,
    edge_cutoff: float = 4.5,
) -> GraphBatch:
    ligand_path = Path(ligand_path)
    ligand_graph = _require_ligand_graph(featurize_ligand(ligand_path), ligand_path)
    protein_features, protein_positions = load_protein_graph(protein_path)

    node_features = torch.cat([ligand_graph.x, protein_features], dim=0)
    positions = torch.cat([ligand_graph.pos, protein_positions], dim=0)
    ligand_mask = torch.zeros(node_features.size(0), dtype=torch.bool)
    ligand_mask[: ligand_graph.x.size(0)] = True
    edge_index = build_radius_edge_index(positions, cutoff=edge_cutoff)

    return build_graph_batch(
        node_features=node_features,
        positions=positions,
        edge_index=edge_index,
        mask=ligand_mask,
        cutoff=cutoff,
    )
