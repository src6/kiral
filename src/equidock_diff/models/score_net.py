"""Score network wrapper interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from equidock_diff.models.egnn import EGNNConfig, EGNNScoreNet


@dataclass(frozen=True)
class ScoreNetConfig:
    egnn: EGNNConfig


class ScoreNet(nn.Module):
    """Wrapper for score-based diffusion modelling."""

    def __init__(self, config: ScoreNetConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = EGNNScoreNet(config.egnn)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(node_features, positions, edge_index, time)
