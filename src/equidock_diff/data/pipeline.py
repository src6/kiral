from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

from equidock_diff.utils.chemistry import (
    ATOM_FEATURE_DIM,
    ATOM_SYMBOL_TO_INDEX,
    FeaturizeOutcome,
    LigandGraph,
    featurize_ligand,
)
from equidock_diff.utils.geometry import batched_centroid, relative_positions


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    crop_mask: torch.Tensor | None = None
    ligand_bond_index: torch.Tensor | None = None
    resolved_crop_cutoff: float | None = None
    retained_protein_nodes: int | None = None

    @property
    def mask(self) -> torch.Tensor | None:
        """Compatibility alias for the original crop-mask field name."""
        return self.crop_mask


GRAPH_CACHE_FORMAT_VERSION = 2


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
        crop_mask=None if batch.crop_mask is None else batch.crop_mask.to(device),
        ligand_bond_index=None
        if batch.ligand_bond_index is None
        else batch.ligand_bond_index.to(device),
        resolved_crop_cutoff=batch.resolved_crop_cutoff,
        retained_protein_nodes=batch.retained_protein_nodes,
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
    context_policy: str = "fixed",
    protein_node_budget: int = 256,
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
            context_policy,
            f"{cutoff:.4f}",
            f"{edge_cutoff:.4f}",
            str(protein_node_budget),
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
        "crop_mask": cpu_batch.crop_mask,
        "ligand_bond_index": cpu_batch.ligand_bond_index,
        "resolved_crop_cutoff": cpu_batch.resolved_crop_cutoff,
        "retained_protein_nodes": cpu_batch.retained_protein_nodes,
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
        crop_mask=payload.get("crop_mask", payload.get("mask")),  # type: ignore[arg-type]
        ligand_bond_index=payload.get("ligand_bond_index"),  # type: ignore[arg-type]
        resolved_crop_cutoff=payload.get("resolved_crop_cutoff"),  # type: ignore[arg-type]
        retained_protein_nodes=payload.get("retained_protein_nodes"),  # type: ignore[arg-type]
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


def _require_ligand_graph(
    outcome: FeaturizeOutcome,
    ligand_path: Path,
) -> LigandGraph:
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


def ligand_max_span(ligand_positions: torch.Tensor) -> float:
    if ligand_positions.dim() != 2 or ligand_positions.size(-1) != 3:
        raise ValueError("ligand_positions must have shape [N, 3].")
    if ligand_positions.size(0) < 2:
        return 0.0
    return float(torch.cdist(ligand_positions, ligand_positions).max().item())


def resolve_context_crop_cutoff(
    ligand_positions: torch.Tensor,
    *,
    context_policy: str,
    cutoff: float,
) -> float:
    if context_policy in {"fixed", "gated"}:
        return float(cutoff)
    if context_policy != "adaptive":
        raise ValueError(f"Unsupported context policy: {context_policy}")
    adaptive_crop = ligand_max_span(ligand_positions) * 0.9 + 2.0
    return float(min(10.0, max(6.0, adaptive_crop)))


def gate_protein_nodes(
    positions: torch.Tensor,
    ligand_mask: torch.Tensor,
    crop_mask: torch.Tensor,
    *,
    protein_node_budget: int,
) -> tuple[torch.Tensor, int]:
    if protein_node_budget <= 0:
        raise ValueError("protein_node_budget must be positive.")

    protein_mask = ~ligand_mask
    cropped_protein_mask = crop_mask & protein_mask
    cropped_protein_indices = torch.where(cropped_protein_mask)[0]
    cropped_count = int(cropped_protein_indices.numel())
    if cropped_count <= protein_node_budget:
        return crop_mask | ligand_mask, cropped_count

    ligand_positions = positions[ligand_mask]
    protein_positions = positions[cropped_protein_indices]
    if ligand_positions.numel() == 0 or protein_positions.numel() == 0:
        return crop_mask | ligand_mask, cropped_count

    min_distances = torch.cdist(protein_positions, ligand_positions).min(dim=1).values
    ranked_indices = torch.argsort(min_distances, stable=True)
    kept_protein_indices = cropped_protein_indices[ranked_indices[:protein_node_budget]]

    gated_mask = ligand_mask.clone()
    gated_mask[kept_protein_indices] = True
    return gated_mask, protein_node_budget

def build_graph_batch(
    node_features: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor | None = None,  # ligand mask
    ligand_bond_index: torch.Tensor | None = None,
    cutoff: float = 10.0,
    resolved_crop_cutoff: float | None = None,
    context_policy: str = "fixed",
    protein_node_budget: int = 256,
) -> GraphBatch:
    if mask is None:
        raise ValueError("build_graph_batch requires a ligand mask.")

    ligand_mask = mask

    # Center on the ligand so downstream geometry depends on relative structure.
    centered_pos = center_on_ligand(positions, ligand_mask)

    # Identity encoding
    indicator = ligand_mask.float().unsqueeze(-1)
    x = torch.cat([node_features, indicator], dim=-1)

    # Apply spatial filtering
    crop_mask = crop_protein_by_distance(centered_pos, ligand_mask, cutoff=cutoff)
    # Make sure ligand isn't cropped out of its own batch
    if context_policy == "gated":
        final_mask, retained_protein_nodes = gate_protein_nodes(
            centered_pos,
            ligand_mask,
            crop_mask,
            protein_node_budget=protein_node_budget,
        )
    else:
        final_mask = crop_mask | ligand_mask
        retained_protein_nodes = int((final_mask & ~ligand_mask).sum().item())

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
        crop_mask=final_mask,  # Original-node crop mask retained for debugging/tests.
        ligand_bond_index=remapped_ligand_bond_index,
        resolved_crop_cutoff=cutoff if resolved_crop_cutoff is None else resolved_crop_cutoff,
        retained_protein_nodes=retained_protein_nodes,
    )


def load_protein_ligand_graph(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float = 10.0,
    edge_cutoff: float = 4.5,
    context_policy: str = "fixed",
    protein_node_budget: int = 256,
) -> GraphBatch:
    ligand_path = Path(ligand_path)
    ligand_graph = _require_ligand_graph(featurize_ligand(ligand_path), ligand_path)
    protein_features, protein_positions = load_protein_graph(protein_path)
    resolved_cutoff = resolve_context_crop_cutoff(
        ligand_graph.pos,
        context_policy=context_policy,
        cutoff=cutoff,
    )

    node_features = torch.cat([ligand_graph.x, protein_features], dim=0)
    positions = torch.cat([ligand_graph.pos, protein_positions], dim=0)
    ligand_mask = torch.zeros(node_features.size(0), dtype=torch.bool)
    ligand_mask[: ligand_graph.x.size(0)] = True

    centered_pos = center_on_ligand(positions, ligand_mask)
    indicator = ligand_mask.float().unsqueeze(-1)
    x = torch.cat([node_features, indicator], dim=-1)

    crop_mask = crop_protein_by_distance(centered_pos, ligand_mask, cutoff=resolved_cutoff)
    if context_policy == "gated":
        final_mask, retained_protein_nodes = gate_protein_nodes(
            centered_pos,
            ligand_mask,
            crop_mask,
            protein_node_budget=protein_node_budget,
        )
    else:
        final_mask = crop_mask | ligand_mask
        retained_protein_nodes = int((final_mask & ~ligand_mask).sum().item())

    cropped_positions = centered_pos[final_mask]
    new_edge_index = build_radius_edge_index(cropped_positions, cutoff=edge_cutoff)
    remapped_ligand_bond_index = ligand_graph.edge_index

    return GraphBatch(
        node_features=x[final_mask],
        positions=cropped_positions,
        edge_index=new_edge_index,
        crop_mask=final_mask,
        ligand_bond_index=remapped_ligand_bond_index,
        resolved_crop_cutoff=resolved_cutoff,
        retained_protein_nodes=retained_protein_nodes,
    )


def load_protein_ligand_graph_cached(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    cutoff: float = 10.0,
    edge_cutoff: float = 4.5,
    cache_dir: Path | None = None,
    context_policy: str = "fixed",
    protein_node_budget: int = 256,
) -> GraphBatch:
    effective_protein_node_budget = protein_node_budget if context_policy == "gated" else 256
    if cache_dir is None:
        return load_protein_ligand_graph(
            protein_path,
            ligand_path,
            cutoff=cutoff,
            edge_cutoff=edge_cutoff,
            context_policy=context_policy,
            protein_node_budget=effective_protein_node_budget,
        )

    effective_cutoff = cutoff
    if context_policy == "adaptive":
        ligand_path = Path(ligand_path)
        ligand_graph = _require_ligand_graph(featurize_ligand(ligand_path), ligand_path)
        effective_cutoff = resolve_context_crop_cutoff(
            ligand_graph.pos,
            context_policy=context_policy,
            cutoff=cutoff,
        )

    cache_path = graph_cache_path(
        cache_dir,
        protein_path,
        ligand_path,
        cutoff=effective_cutoff,
        edge_cutoff=edge_cutoff,
        context_policy=context_policy,
        protein_node_budget=effective_protein_node_budget,
    )
    if cache_path.exists():
        cached_batch = load_graph_batch_cache(cache_path)
        if cached_batch.resolved_crop_cutoff is None or cached_batch.retained_protein_nodes is None:
            return GraphBatch(
                node_features=cached_batch.node_features,
                positions=cached_batch.positions,
                edge_index=cached_batch.edge_index,
                crop_mask=cached_batch.crop_mask,
                ligand_bond_index=cached_batch.ligand_bond_index,
                resolved_crop_cutoff=effective_cutoff,
                retained_protein_nodes=int((cached_batch.node_features[:, -1] <= 0.5).sum().item()),
            )
        return cached_batch

    batch = load_protein_ligand_graph(
        protein_path,
        ligand_path,
        cutoff=cutoff,
        edge_cutoff=edge_cutoff,
        context_policy=context_policy,
        protein_node_budget=effective_protein_node_budget,
    )
    write_graph_batch_cache(cache_path, batch)
    return batch
