from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EGNNConfig:
    node_dim: int
    hidden_dim: int
    num_layers: int
    time_dim: int = 32


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        diff = positions[src] - positions[dst]
        radial = diff.pow(2).sum(dim=-1, keepdim=True)

        edge_inputs = torch.cat([node_states[src], node_states[dst], radial], dim=-1)
        messages = self.edge_mlp(edge_inputs)

        num_nodes = node_states.size(0)
        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, dst, messages)

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
        hidden_positions = centered_positions
        for layer in self.layers:
            node_states, hidden_positions = layer(node_states, hidden_positions, edge_index)

        score_scale = self.score_head(node_states)
        score = centered_positions * score_scale + (hidden_positions - centered_positions)
        score = score - score.mean(dim=0, keepdim=True)
        return score
