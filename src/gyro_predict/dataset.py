"""PyTorch Dataset, target transforms, and fold DataLoader construction."""

from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class FluxDataset(Dataset):
    """Simple dataset wrapping feature and target numpy arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def apply_target_transform(
    targets: np.ndarray, transform: str, epsilon: float = 1e-6
) -> np.ndarray:
    """Apply target transform before training.

    Args:
        targets: Array of shape (N, 2).
        transform: One of "raw", "log", "power".
        epsilon: Small constant for numerical stability.

    Returns:
        Transformed targets, same shape.
    """
    if transform == "raw":
        return targets.copy()
    elif transform == "log":
        # Q values are always positive; log compresses dynamic range
        return np.log(np.abs(targets) + epsilon)
    elif transform == "power":
        return np.sign(targets) * np.abs(targets) ** (1.0 / 3.0)
    else:
        raise ValueError(f"Unknown target transform: {transform}")


def invert_target_transform(
    predictions: np.ndarray, transform: str, epsilon: float = 1e-6
) -> np.ndarray:
    """Invert target transform to recover original-scale Q.

    Args:
        predictions: Transformed predictions, shape (N, 2).
        transform: One of "raw", "log", "power".
        epsilon: Must match the epsilon used in apply_target_transform.

    Returns:
        Original-scale predictions.
    """
    if transform == "raw":
        return predictions.copy()
    elif transform == "log":
        # Inverse of log(|x| + eps): exp(y) - eps
        return np.exp(predictions) - epsilon
    elif transform == "power":
        return np.sign(predictions) * np.abs(predictions) ** 3.0
    else:
        raise ValueError(f"Unknown target transform: {transform}")


def make_fold_loaders(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    target_transform: str,
    batch_size: int,
    epsilon: float = 1e-6,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Create train/val DataLoaders for one CV fold.

    Fits StandardScaler on training features only. Applies target transform.

    Returns:
        (train_loader, val_loader, scaler)
    """
    # Fit scaler on training features only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_val = scaler.transform(features[val_idx])

    # Apply target transform
    y_train = apply_target_transform(targets[train_idx], target_transform, epsilon)
    y_val = apply_target_transform(targets[val_idx], target_transform, epsilon)

    train_ds = FluxDataset(X_train, y_train)
    val_ds = FluxDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler
