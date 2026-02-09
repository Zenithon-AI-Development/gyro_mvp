"""Loss functions and evaluation metrics for flux prediction."""

import numpy as np
import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """Relative L2 loss: mean(((pred - target) / |target|)^2).

    Used when target_transform="raw" to ensure equal weighting across
    different flux magnitude scales.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(torch.abs(target), min=self.epsilon)
        return torch.mean(((pred - target) / denom) ** 2)


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-6
) -> dict:
    """Compute evaluation metrics on original-scale Q values.

    Args:
        pred: Predictions, shape (N, 2) = [Q_electron, Q_ion].
        target: Ground truth, shape (N, 2).
        epsilon: Floor for relative error denominators.

    Returns:
        Dict of metric name -> value.
    """
    species_names = ["electron", "ion"]
    metrics = {}

    for s, name in enumerate(species_names):
        p = pred[:, s]
        t = target[:, s]
        denom = np.maximum(np.abs(t), epsilon)

        # Relative L2 per species
        rel_errors = ((p - t) / denom) ** 2
        metrics[f"rel_l2_{name}"] = float(np.mean(rel_errors))

        # MAE
        metrics[f"mae_{name}"] = float(np.mean(np.abs(p - t)))

        # RMSE
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean((p - t) ** 2)))

        # MAPE (%)
        metrics[f"mape_{name}"] = float(np.mean(np.abs(p - t) / denom) * 100)

    # Average relative L2 across species
    metrics["rel_l2_mean"] = (metrics["rel_l2_electron"] + metrics["rel_l2_ion"]) / 2

    return metrics
