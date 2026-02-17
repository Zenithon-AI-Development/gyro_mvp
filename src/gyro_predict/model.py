"""Model architectures for gyrokinetic flux prediction.

Models:
  FluxMLP          — flat MLP, single concatenated input (V5a/V5b)
  FiLMFluxMLP      — FiLM conditioning: params modulate TGLF hidden layers (V5c)
  HadamardFluxMLP  — dual-head with element-wise fusion (V5d)
  BilinearFluxMLP  — dual-head with bilinear fusion (V5e)
"""

import torch
import torch.nn as nn
from typing import List


class FluxMLP(nn.Module):
    """Flat MLP mapping TGLF features to 2 flux targets.

    Architecture:
        BatchNorm1d -> [Linear -> ReLU -> Dropout] x N_layers -> Linear(n_outputs) -> [Softplus]
    """
    dual_input = False

    def __init__(
        self,
        input_dim: int = 378,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
        use_softplus: bool = True,
    ):
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


class FiLMFluxMLP(nn.Module):
    """FiLM-conditioned MLP: global params modulate TGLF hidden layers.

    Architecture:
        Params → shared encoder → per-layer (gamma, beta)
        TGLF → BatchNorm → [Linear → FiLM → ReLU → Dropout] x N → Linear → Softplus

    FiLM: h = gamma * h + beta (element-wise per hidden layer)
    gamma initialized near 1.0, beta near 0.0 so model starts close to TGLF-only.
    """
    dual_input = True

    def __init__(
        self,
        tglf_dim: int = 378,
        param_dim: int = 13,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
        use_softplus: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.use_softplus = use_softplus
        self.hidden_dims = hidden_dims

        # TGLF input normalization
        self.tglf_bn = nn.BatchNorm1d(tglf_dim)

        # Param shared encoder: 13 → 64
        self.param_bn = nn.BatchNorm1d(param_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
        )

        # Per-layer FiLM generators: 64 → 2*H_i (gamma + beta)
        self.film_generators = nn.ModuleList()
        for h_dim in hidden_dims:
            gen = nn.Linear(64, 2 * h_dim)
            # Initialize gamma near 1, beta near 0
            nn.init.ones_(gen.weight[:h_dim])
            nn.init.zeros_(gen.weight[h_dim:])
            nn.init.zeros_(gen.bias[:h_dim])
            nn.init.zeros_(gen.bias[h_dim:])
            self.film_generators.append(gen)

        # TGLF trunk layers (without Sequential so we can apply FiLM between)
        self.trunk_linears = nn.ModuleList()
        self.trunk_dropouts = nn.ModuleList()
        in_dim = tglf_dim
        for h_dim in hidden_dims:
            self.trunk_linears.append(nn.Linear(in_dim, h_dim))
            self.trunk_dropouts.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output head
        self.output_linear = nn.Linear(hidden_dims[-1], n_outputs)
        if use_softplus:
            self.output_activation = nn.Softplus()

    def forward(self, x_tglf, x_params):
        # Encode params once
        p = self.param_bn(x_params)
        p = self.param_encoder(p)  # (B, 64)

        # TGLF trunk with FiLM modulation
        h = self.tglf_bn(x_tglf)
        for linear, dropout, film_gen in zip(
            self.trunk_linears, self.trunk_dropouts, self.film_generators
        ):
            h = linear(h)
            # FiLM: split into gamma and beta
            film = film_gen(p)  # (B, 2*H)
            gamma, beta = film.chunk(2, dim=-1)
            gamma = gamma + 1.0  # center around 1 (so default is identity)
            h = gamma * h + beta
            h = torch.relu(h)
            h = dropout(h)

        h = self.output_linear(h)
        if self.use_softplus:
            h = self.output_activation(h)
        return h


class HadamardFluxMLP(nn.Module):
    """Dual-head MLP with Hadamard (element-wise) fusion.

    Architecture:
        TGLF → BatchNorm → [Linear→ReLU→Dropout] x N → h_tglf
        Params → BatchNorm → Linear→ReLU→Linear → h_params
        Fusion: h = h_tglf ⊙ h_params
        Head: Linear→ReLU→Dropout→Linear→Softplus
    """
    dual_input = True

    def __init__(
        self,
        tglf_dim: int = 378,
        param_dim: int = 13,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
        use_softplus: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.use_softplus = use_softplus
        fusion_dim = hidden_dims[-1]  # e.g. 128 for [512,256,128]

        # TGLF encoder
        self.tglf_bn = nn.BatchNorm1d(tglf_dim)
        tglf_layers = []
        in_dim = tglf_dim
        for h_dim in hidden_dims:
            tglf_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.tglf_encoder = nn.Sequential(*tglf_layers)

        # Param encoder: 13 → 64 → fusion_dim
        self.param_bn = nn.BatchNorm1d(param_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Linear(64, fusion_dim),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

        if use_softplus:
            self.output_activation = nn.Softplus()

    def forward(self, x_tglf, x_params):
        h_tglf = self.tglf_encoder(self.tglf_bn(x_tglf))
        h_params = self.param_encoder(self.param_bn(x_params))

        # Hadamard fusion
        h = h_tglf * h_params

        h = self.head(h)
        if self.use_softplus:
            h = self.output_activation(h)
        return h


class BilinearFluxMLP(nn.Module):
    """Dual-head MLP with bilinear fusion.

    Architecture:
        TGLF → BatchNorm → [Linear→ReLU→Dropout] x N → h_tglf (last_H)
        Params → BatchNorm → Linear→ReLU→Linear → h_params (32)
        Fusion: h_k = h_tglf^T W_k h_params + b_k, k=1..64
        Head: Linear(64, n_outputs) → Softplus
    """
    dual_input = True

    def __init__(
        self,
        tglf_dim: int = 378,
        param_dim: int = 13,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_outputs: int = 2,
        use_softplus: bool = True,
        param_head_dim: int = 32,
        bilinear_out_dim: int = 64,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.use_softplus = use_softplus
        tglf_out_dim = hidden_dims[-1]

        # TGLF encoder
        self.tglf_bn = nn.BatchNorm1d(tglf_dim)
        tglf_layers = []
        in_dim = tglf_dim
        for h_dim in hidden_dims:
            tglf_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.tglf_encoder = nn.Sequential(*tglf_layers)

        # Param encoder: 13 → 64 → param_head_dim
        self.param_bn = nn.BatchNorm1d(param_dim)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Linear(64, param_head_dim),
        )

        # Bilinear fusion: (tglf_out_dim, param_head_dim) → bilinear_out_dim
        self.bilinear = nn.Bilinear(tglf_out_dim, param_head_dim, bilinear_out_dim)

        # Output head
        self.output_linear = nn.Linear(bilinear_out_dim, n_outputs)

        if use_softplus:
            self.output_activation = nn.Softplus()

    def forward(self, x_tglf, x_params):
        h_tglf = self.tglf_encoder(self.tglf_bn(x_tglf))
        h_params = self.param_encoder(self.param_bn(x_params))

        # Bilinear fusion
        h = self.bilinear(h_tglf, h_params)
        h = torch.relu(h)

        h = self.output_linear(h)
        if self.use_softplus:
            h = self.output_activation(h)
        return h


def create_model(
    model_type: str,
    tglf_dim: int,
    param_dim: int,
    hidden_dims: List[int],
    dropout: float,
    n_outputs: int = 2,
    use_softplus: bool = True,
) -> nn.Module:
    """Factory function to create model by type.

    Args:
        model_type: One of 'mlp', 'film', 'hadamard', 'bilinear'.
        tglf_dim: TGLF feature dimension (378).
        param_dim: Global param dimension (0 for TGLF-only, 13 for V5b+).
        hidden_dims: Hidden layer sizes for TGLF trunk.
        dropout: Dropout probability.
        n_outputs: Number of outputs.
        use_softplus: Whether to apply Softplus output activation.
    """
    if model_type == "mlp":
        return FluxMLP(
            input_dim=tglf_dim + param_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            n_outputs=n_outputs,
            use_softplus=use_softplus,
        )
    elif model_type == "film":
        return FiLMFluxMLP(
            tglf_dim=tglf_dim,
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            n_outputs=n_outputs,
            use_softplus=use_softplus,
        )
    elif model_type == "hadamard":
        return HadamardFluxMLP(
            tglf_dim=tglf_dim,
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            n_outputs=n_outputs,
            use_softplus=use_softplus,
        )
    elif model_type == "bilinear":
        return BilinearFluxMLP(
            tglf_dim=tglf_dim,
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            n_outputs=n_outputs,
            use_softplus=use_softplus,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
