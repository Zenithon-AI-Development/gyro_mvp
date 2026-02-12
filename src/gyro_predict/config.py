"""Central configuration for gyrokinetic flux prediction MVP."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Paths:
    data_dir: str = "/data"
    csv_dir: str = str(Path(__file__).resolve().parent.parent / "csvs")
    checkpoint_dir: str = "/checkpoints/gyro_predict"
    validated_csv: str = "validated_parameters.csv"
    flux_comparison_csv: str = "flux_comparison.csv"


@dataclass
class FeatureConfig:
    n_ky: int = 21
    n_modes: int = 4
    n_species: int = 2
    n_fields: int = 2
    energy_flux_idx: int = 1  # index into flux_type dimension
    agg_funcs: List[str] = field(default_factory=lambda: ["max", "sum", "mean", "std"])
    global_param_columns: List[str] = field(default_factory=list)

    @property
    def total_features(self) -> int:
        """Compute total features: 378 TGLF + len(global_param_columns)."""
        return 378 + len(self.global_param_columns)


@dataclass
class ModelConfig:
    input_dim: int = 378
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    n_outputs: int = 2
    use_softplus: bool = True  # Apply Softplus to output to ensure Q > 0


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-3
    epochs: int = 2000
    patience: int = 200
    n_folds: int = 5
    ensemble_size: int = 5
    batch_size: int = 32
    target_transform: str = "raw"  # "raw", "log", "power"
    seed: int = 42
    epsilon: float = 1e-6


@dataclass
class SearchConfig:
    dropout: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])
    weight_decay: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    lr: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3])
    hidden_dims: List[List[int]] = field(
        default_factory=lambda: [[128, 64, 32], [256, 128, 64], [512, 256, 128]]
    )
    target_transform: List[str] = field(
        default_factory=lambda: ["raw", "log", "power"]
    )
    ensemble_size: List[int] = field(default_factory=lambda: [5, 10])


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: str = "PLACEHOLDER_ENTITY"
    project: str = "PLACEHOLDER_PROJECT"
    group: str = "v1-mlp"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
