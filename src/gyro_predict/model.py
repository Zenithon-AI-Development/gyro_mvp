"""FluxMLP: flat MLP for gyrokinetic flux prediction."""

from typing import List

import torch.nn as nn


class FluxMLP(nn.Module):
    """Flat MLP mapping TGLF features to 2 flux targets.

    Architecture:
        BatchNorm1d -> [Linear -> ReLU -> Dropout] x N_layers -> Linear(n_outputs) -> [Softplus]

    The optional Softplus activation ensures outputs are always positive (Q > 0),
    which is physically correct for energy fluxes.
    """

    def __init__(
        self,
        input_dim: int = 378,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
        use_softplus: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (378 for TGLF-only, 378+N with global params)
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            n_outputs: Number of outputs (2 for Q_electron, Q_ion)
            use_softplus: If True, apply Softplus to output to ensure Q > 0
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.use_softplus = use_softplus
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

        if use_softplus:
            self.output_activation = nn.Softplus()

    def forward(self, x):
        x = self.input_bn(x)
        x = self.network(x)
        if self.use_softplus:
            x = self.output_activation(x)
        return x
