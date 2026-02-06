"""EGNN model interfaces (core logic intentionally left to user)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EGNNConfig:
    node_dim: int
    hidden_dim: int
    num_layers: int


class EGNNScoreNet(nn.Module):
    """Interface for an EGNN-based score network.

    Implementations should be SE(3)-equivariant and use L=0/L=1 features.
    """

    def __init__(self, config: EGNNConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement EGNN score model forward pass")
