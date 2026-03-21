from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    ligand_bond_index: torch.Tensor | None = None


GRAPH_CACHE_FORMAT_VERSION = 1


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


def move_graph_batch(batch: GraphBatch, device: torch.device) -> GraphBatch:
    return GraphBatch(
        node_features=batch.node_features.to(device),
        positions=batch.positions.to(device),
        edge_index=batch.edge_index.to(device),
        mask=None if batch.mask is None else batch.mask.to(device),
        ligand_bond_index=None
        if batch.ligand_bond_index is None
        else batch.ligand_bond_index.to(device),
    )


def graph_batch_to_cpu(batch: GraphBatch) -> GraphBatch:
    return move_graph_batch(batch, torch.device("cpu"))


def graph_cache_path(
    cache_dir: Path,
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float,
    edge_cutoff: float,
) -> Path:
    protein = Path(protein_path)
    ligand = Path(ligand_path)
    protein_stat = protein.stat()
    ligand_stat = ligand.stat()
    signature = "|".join(
        [
            str(GRAPH_CACHE_FORMAT_VERSION),
            str(protein.resolve()),
            str(protein_stat.st_mtime_ns),
            str(protein_stat.st_size),
            str(ligand.resolve()),
            str(ligand_stat.st_mtime_ns),
            str(ligand_stat.st_size),
            f"{cutoff:.4f}",
            f"{edge_cutoff:.4f}",
        ]
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    stem = protein.name.replace("_protein.pdb", "")
    return cache_dir / f"{stem}_{digest}.pt"


def _serialize_graph_batch(batch: GraphBatch) -> dict[str, object]:
    cpu_batch = graph_batch_to_cpu(batch)
    return {
        "format_version": GRAPH_CACHE_FORMAT_VERSION,
        "node_features": cpu_batch.node_features,
        "positions": cpu_batch.positions,
        "edge_index": cpu_batch.edge_index,
        "mask": cpu_batch.mask,
        "ligand_bond_index": cpu_batch.ligand_bond_index,
    }


def _deserialize_graph_batch(payload: dict[str, object]) -> GraphBatch:
    format_version = int(payload.get("format_version", -1))
    if format_version != GRAPH_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported graph cache format: {format_version} (expected {GRAPH_CACHE_FORMAT_VERSION})."
        )
    return GraphBatch(
        node_features=payload["node_features"],  # type: ignore[arg-type]
        positions=payload["positions"],  # type: ignore[arg-type]
        edge_index=payload["edge_index"],  # type: ignore[arg-type]
        mask=payload.get("mask"),  # type: ignore[arg-type]
        ligand_bond_index=payload.get("ligand_bond_index"),  # type: ignore[arg-type]
    )


def load_graph_batch_cache(path: Path) -> GraphBatch:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return _deserialize_graph_batch(payload)


def write_graph_batch_cache(path: Path, batch: GraphBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_serialize_graph_batch(batch), path)


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
    ligand_bond_index: torch.Tensor | None = None,
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

    remapped_ligand_bond_index = None
    if ligand_bond_index is not None:
        bond_src, bond_dst = ligand_bond_index
        bond_keep_mask = final_mask[bond_src] & final_mask[bond_dst]
        filtered_ligand_bond_index = ligand_bond_index[:, bond_keep_mask]
        remapped_ligand_bond_index = idx_map[filtered_ligand_bond_index]

    return GraphBatch(
        node_features=x[final_mask],
        positions=centered_pos[final_mask],
        edge_index=new_edge_index,
        mask=final_mask, # New batch mask
        ligand_bond_index=remapped_ligand_bond_index,
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
        ligand_bond_index=ligand_graph.edge_index,
        cutoff=cutoff,
    )


def load_protein_ligand_graph_cached(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float = 10.0,
    edge_cutoff: float = 4.5,
    cache_dir: Path | None = None,
) -> GraphBatch:
    if cache_dir is None:
        return load_protein_ligand_graph(
            protein_path,
            ligand_path,
            cutoff=cutoff,
            edge_cutoff=edge_cutoff,
        )

    cache_path = graph_cache_path(
        cache_dir,
        protein_path,
        ligand_path,
        cutoff=cutoff,
        edge_cutoff=edge_cutoff,
    )
    if cache_path.exists():
        return load_graph_batch_cache(cache_path)

    batch = load_protein_ligand_graph(
        protein_path,
        ligand_path,
        cutoff=cutoff,
        edge_cutoff=edge_cutoff,
    )
    write_graph_batch_cache(cache_path, batch)
    return batch
