"""FluxMLP: flat MLP for gyrokinetic flux prediction."""

from typing import List

import torch.nn as nn


class FluxMLP(nn.Module):
    """Flat MLP mapping 378 TGLF features to 2 flux targets.

    Architecture:
        BatchNorm1d -> [Linear -> ReLU -> Dropout] x N_layers -> Linear(n_outputs)
    """

    def __init__(
        self,
        input_dim: int = 378,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_bn(x)
        return self.network(x)
