from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EDGE_TYPE_LIGAND_LIGAND = 0
EDGE_TYPE_PROTEIN_PROTEIN = 1
EDGE_TYPE_LIGAND_PROTEIN = 2
NUM_EDGE_TYPES = 3


@dataclass(frozen=True)
class EGNNConfig:
    node_dim: int
    hidden_dim: int
    num_layers: int
    time_dim: int = 32
    use_hetero_edges: bool = False
    use_ligand_global_node: bool = False


def infer_ligand_mask(node_features: torch.Tensor) -> torch.Tensor:
    if node_features.dim() != 2:
        raise ValueError("node_features must have shape [N, F].")

    if node_features.size(-1) >= 2:
        first = node_features[:, 0]
        second = node_features[:, 1]
        first_binary = torch.all((first == 0.0) | (first == 1.0))
        second_binary = torch.all((second == 0.0) | (second == 1.0))
        if bool(first_binary and second_binary and torch.allclose(first + second, torch.ones_like(first))):
            return first > 0.5

    if node_features.size(-1) >= 1:
        indicator = node_features[:, -1]
        if bool(torch.all((indicator == 0.0) | (indicator == 1.0))):
            return indicator > 0.5

    return torch.zeros(node_features.size(0), device=node_features.device, dtype=torch.bool)


def infer_edge_types(edge_index: torch.Tensor, ligand_mask: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    src_is_ligand = ligand_mask[src]
    dst_is_ligand = ligand_mask[dst]

    edge_types = torch.full_like(src, EDGE_TYPE_LIGAND_PROTEIN)
    edge_types[src_is_ligand & dst_is_ligand] = EDGE_TYPE_LIGAND_LIGAND
    edge_types[~src_is_ligand & ~dst_is_ligand] = EDGE_TYPE_PROTEIN_PROTEIN
    return edge_types


class EGNNLayer(nn.Module):
    """Minimal EGNN-style message passing block."""

    def __init__(self, config: EGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        if config.use_hetero_edges:
            self.message_transforms = nn.ModuleList(
                [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(NUM_EDGE_TYPES)]
            )
            for transform in self.message_transforms:
                nn.init.eye_(transform.weight)
                nn.init.zeros_(transform.bias)
        else:
            self.message_transforms = None
        if config.use_ligand_global_node:
            self.ligand_global_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.ligand_global_mlp = None
        self.coord_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        ligand_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        diff = positions[src] - positions[dst]
        radial = diff.pow(2).sum(dim=-1, keepdim=True)

        edge_inputs = torch.cat([node_states[src], node_states[dst], radial], dim=-1)
        base_messages = self.edge_mlp(edge_inputs)
        if self.message_transforms is None:
            messages = base_messages
        else:
            messages = torch.zeros_like(base_messages)
            for edge_type in range(NUM_EDGE_TYPES):
                edge_mask = edge_types == edge_type
                if not bool(edge_mask.any()):
                    continue
                messages[edge_mask] = self.message_transforms[edge_type](base_messages[edge_mask])

        num_nodes = node_states.size(0)
        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, dst, messages)
        if self.ligand_global_mlp is not None and ligand_mask is not None and bool(ligand_mask.any()):
            ligand_context = node_states[ligand_mask].mean(dim=0, keepdim=True)
            ligand_context = self.ligand_global_mlp(ligand_context)
            aggregated = aggregated.clone()
            aggregated[ligand_mask] = aggregated[ligand_mask] + ligand_context.expand(
                int(ligand_mask.sum().item()),
                -1,
            )

        coord_weights = self.coord_mlp(messages)
        coord_messages = diff * coord_weights
        coord_updates = torch.zeros_like(positions)
        coord_updates.index_add_(0, dst, coord_messages)

        updated_states = node_states + self.node_mlp(
            torch.cat([node_states, aggregated], dim=-1)
        )
        updated_positions = positions + coord_updates
        return updated_states, updated_positions

class EGNNScoreNet(nn.Module):
    """Interface for an EGNN-based score network.

    Implementations should be SE(3)-equivariant and use L=0/L=1 features.
    """

    def __init__(self, config: EGNNConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embed = nn.Linear(config.node_dim, config.hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.hidden_dim),
        )
        self.layers = nn.ModuleList(
            EGNNLayer(config) for _ in range(max(config.num_layers, 1))
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.dim() != 2:
            raise ValueError("node_features must have shape [N, F].")
        if positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3].")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")

        node_states = self.node_embed(node_features)
        time = time.reshape(-1)
        if time.numel() == 1:
            time = time.expand(node_features.size(0))
        elif time.numel() != node_features.size(0):
            raise ValueError("time must be scalar or have one value per node.")

        node_states = node_states + self.time_embed(time.unsqueeze(-1))

        centered_positions = positions - positions.mean(dim=0, keepdim=True)
        ligand_mask = None
        if self.config.use_hetero_edges or self.config.use_ligand_global_node:
            ligand_mask = infer_ligand_mask(node_features)
        if self.config.use_hetero_edges:
            edge_types = infer_edge_types(edge_index, ligand_mask)
        else:
            edge_types = torch.zeros(
                edge_index.size(1),
                device=edge_index.device,
                dtype=edge_index.dtype,
            )
        hidden_positions = centered_positions
        for layer in self.layers:
            node_states, hidden_positions = layer(
                node_states,
                hidden_positions,
                edge_index,
                edge_types,
                ligand_mask,
            )

        score_scale = self.score_head(node_states)
        score = centered_positions * score_scale + (hidden_positions - centered_positions)
        score = score - score.mean(dim=0, keepdim=True)
        return score
