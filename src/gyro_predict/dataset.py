"""PyTorch Dataset, target transforms, and fold DataLoader construction."""

from typing import Optional, Tuple

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


class DualInputFluxDataset(Dataset):
    """Dataset returning (tglf_features, param_features, targets) triples."""

    def __init__(self, tglf_features: np.ndarray, param_features: np.ndarray,
                 targets: np.ndarray):
        self.tglf = torch.tensor(tglf_features, dtype=torch.float32)
        self.params = torch.tensor(param_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tglf)

    def __getitem__(self, idx):
        return self.tglf[idx], self.params[idx], self.targets[idx]


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
    tglf_dim: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Create train/val DataLoaders for one CV fold.

    Fits StandardScaler on training features only. Applies target transform.

    Args:
        features: Full feature array (N, D) where D = tglf_dim + param_dim.
        targets: Target array (N, 2).
        train_idx, val_idx: Fold indices.
        target_transform: Target transform name.
        batch_size: Batch size.
        epsilon: Numerical stability constant.
        tglf_dim: If set and < features.shape[1], split into (tglf, params)
                   and return DualInputFluxDataset. None = single-input.

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

    # Choose dataset type
    if tglf_dim is not None and tglf_dim < features.shape[1]:
        train_ds = DualInputFluxDataset(
            X_train[:, :tglf_dim], X_train[:, tglf_dim:], y_train)
        val_ds = DualInputFluxDataset(
            X_val[:, :tglf_dim], X_val[:, tglf_dim:], y_val)
    else:
        train_ds = FluxDataset(X_train, y_train)
        val_ds = FluxDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler
