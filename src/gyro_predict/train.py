"""Training orchestration: single model, K-fold CV, grid search, ensemble."""

import copy
import json
import os
import random
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, ModelConfig, TrainConfig
from .dataset import (
    DualInputFluxDataset,
    FluxDataset,
    apply_target_transform,
    invert_target_transform,
    make_fold_loaders,
)
from .loss import RelativeL2Loss, compute_metrics
from .model import FluxMLP, create_model


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_loss_fn(target_transform: str, epsilon: float) -> nn.Module:
    """Select loss function based on target transform."""
    if target_transform == "raw":
        return RelativeL2Loss(epsilon=epsilon)
    else:
        # For log/power transforms, relative errors are implicit
        return nn.MSELoss()


def _model_forward(model, batch, device):
    """Dispatch forward call based on model type (single vs dual input).

    Returns (predictions, targets_on_device).
    """
    if getattr(model, 'dual_input', False):
        x_tglf, x_params, y = batch
        x_tglf, x_params, y = x_tglf.to(device), x_params.to(device), y.to(device)
        pred = model(x_tglf, x_params)
    else:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
    return pred, y


def _model_predict(model, batch, device):
    """Forward pass for prediction only (no targets needed, for val/test).

    Returns predictions tensor on CPU.
    """
    if getattr(model, 'dual_input', False):
        x_tglf, x_params = batch[0], batch[1]
        x_tglf, x_params = x_tglf.to(device), x_params.to(device)
        return model(x_tglf, x_params).cpu()
    else:
        x = batch[0]
        x = x.to(device)
        return model(x).cpu()


def train_one_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_targets_original: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    wandb_run=None,
) -> Tuple[nn.Module, dict]:
    """Train a single model with early stopping.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        val_targets_original: Original-scale val targets for metric computation.
        model_config: Model architecture config.
        train_config: Training hyperparameters.
        device: torch device.
        wandb_run: Optional wandb run for logging.

    Returns:
        (best_model, metrics_dict)
    """
    model = create_model(
        model_type=model_config.model_type,
        tglf_dim=model_config.input_dim - model_config.param_dim,
        param_dim=model_config.param_dim,
        hidden_dims=model_config.hidden_dims,
        dropout=model_config.dropout,
        n_outputs=model_config.n_outputs,
        use_softplus=model_config.use_softplus,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.epochs)
    loss_fn = _get_loss_fn(train_config.target_transform, train_config.epsilon)

    best_val_metric = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(train_config.epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred, y = _model_forward(model, batch, device)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_preds = []
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pred, y = _model_forward(model, batch, device)
                val_loss_sum += loss_fn(pred, y).item()
                n_val_batches += 1
                val_preds.append(pred.cpu().numpy())

        val_loss = val_loss_sum / max(n_val_batches, 1)
        val_preds = np.concatenate(val_preds, axis=0)

        # Invert transform and compute original-scale metrics
        val_preds_original = invert_target_transform(
            val_preds, train_config.target_transform, train_config.epsilon
        )
        metrics = compute_metrics(
            val_preds_original, val_targets_original, train_config.epsilon
        )
        val_rel_l2 = metrics["rel_l2_mean"]

        # Logging
        if wandb_run is not None and epoch % 50 == 0:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rel_l2_mean": val_rel_l2,
            })

        # Early stopping
        if val_rel_l2 < best_val_metric:
            best_val_metric = val_rel_l2
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final metrics on best model
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            val_preds.append(_model_predict(model, batch, device).numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_preds_original = invert_target_transform(
        val_preds, train_config.target_transform, train_config.epsilon
    )
    final_metrics = compute_metrics(
        val_preds_original, val_targets_original, train_config.epsilon
    )
    final_metrics["best_epoch"] = epoch - patience_counter
    final_metrics["total_epochs"] = epoch + 1

    return model, final_metrics


def run_kfold(
    features: np.ndarray,
    targets: np.ndarray,
    experiment_groups: np.ndarray,
    metadata: pd.DataFrame,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    wandb_run=None,
) -> dict:
    """Run K-fold cross-validation.

    Returns dict with per-fold metrics, aggregate metrics, and out-of-fold predictions.
    """
    n_folds = train_config.n_folds
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=train_config.seed
    )

    # Determine if dual-input splitting is needed
    tglf_dim = None
    if model_config.model_type != "mlp" and model_config.param_dim > 0:
        tglf_dim = model_config.input_dim - model_config.param_dim

    fold_metrics = []
    fold_details = []
    oof_predictions = np.zeros_like(targets)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(features, experiment_groups)
    ):
        set_seed(train_config.seed + fold_idx)

        train_loader, val_loader, scaler = make_fold_loaders(
            features,
            targets,
            train_idx,
            val_idx,
            train_config.target_transform,
            train_config.batch_size,
            train_config.epsilon,
            tglf_dim=tglf_dim,
        )

        val_targets_original = targets[val_idx]

        model, metrics = train_one_model(
            train_loader,
            val_loader,
            val_targets_original,
            model_config,
            train_config,
            device,
            wandb_run=wandb_run,
        )

        fold_metrics.append(metrics)

        # Collect out-of-fold predictions
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                val_preds.append(_model_predict(model, batch, device).numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        oof_predictions[val_idx] = invert_target_transform(
            val_preds, train_config.target_transform, train_config.epsilon
        )

        # Store fold details with run names
        fold_details.append({
            "fold": fold_idx,
            "val_run_names": metadata.iloc[val_idx]["folder_name"].tolist(),
            "val_predictions": oof_predictions[val_idx].tolist(),
            "val_targets": targets[val_idx].tolist(),
        })

        print(
            f"  Fold {fold_idx + 1}/{n_folds}: "
            f"rel_l2_mean={metrics['rel_l2_mean']:.4f}, "
            f"mape_elec={metrics['mape_electron']:.1f}%, "
            f"mape_ion={metrics['mape_ion']:.1f}%"
        )

    # Aggregate metrics
    rel_l2_values = [m["rel_l2_mean"] for m in fold_metrics]
    oof_metrics = compute_metrics(oof_predictions, targets, train_config.epsilon)

    result = {
        "fold_metrics": fold_metrics,
        "mean_rel_l2": float(np.mean(rel_l2_values)),
        "std_rel_l2": float(np.std(rel_l2_values)),
        "oof_metrics": oof_metrics,
        "oof_predictions": oof_predictions,
        "fold_details": fold_details,
    }
    return result


def run_hyperparameter_search(
    features: np.ndarray,
    targets: np.ndarray,
    experiment_groups: np.ndarray,
    metadata: pd.DataFrame,
    config: Config,
    device: torch.device,
) -> Tuple[dict, list]:
    """Grid search over hyperparameters with K-fold CV.

    Returns:
        (best_config_dict, all_results_list)
    """
    import wandb

    search = config.search
    all_results = []
    best_score = float("inf")
    best_config_dict = None

    combos = list(product(
        search.dropout,
        search.weight_decay,
        search.lr,
        search.hidden_dims,
        search.target_transform,
    ))
    print(f"Starting grid search: {len(combos)} combinations")

    for i, (dropout, wd, lr, hidden_dims, tt) in enumerate(tqdm(combos, desc="HP search")):
        mc = ModelConfig(
            input_dim=config.features.total_features,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
        )
        tc = TrainConfig(
            lr=lr,
            weight_decay=wd,
            target_transform=tt,
            epochs=config.train.epochs,
            patience=config.train.patience,
            n_folds=config.train.n_folds,
            batch_size=config.train.batch_size,
            seed=config.train.seed,
            epsilon=config.train.epsilon,
        )

        hp_dict = {
            "dropout": dropout,
            "weight_decay": wd,
            "lr": lr,
            "hidden_dims": hidden_dims,
            "target_transform": tt,
        }

        # Optional wandb logging per combo
        wandb_run = None
        if config.wandb.enabled:
            wandb_run = wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                group="v1-mlp-search",
                name=f"hp_{i:03d}",
                config=hp_dict,
                reinit=True,
            )

        print(f"\n[{i + 1}/{len(combos)}] dropout={dropout}, wd={wd}, lr={lr}, "
              f"arch={hidden_dims}, transform={tt}")

        result = run_kfold(
            features, targets, experiment_groups, metadata, mc, tc, device, wandb_run
        )

        hp_dict["mean_rel_l2"] = result["mean_rel_l2"]
        hp_dict["std_rel_l2"] = result["std_rel_l2"]
        hp_dict["oof_mape_electron"] = result["oof_metrics"]["mape_electron"]
        hp_dict["oof_mape_ion"] = result["oof_metrics"]["mape_ion"]
        all_results.append(hp_dict)

        if wandb_run is not None:
            wandb_run.summary["mean_rel_l2"] = result["mean_rel_l2"]
            wandb_run.summary["std_rel_l2"] = result["std_rel_l2"]
            wandb_run.finish()

        if result["mean_rel_l2"] < best_score:
            best_score = result["mean_rel_l2"]
            best_config_dict = hp_dict.copy()
            print(f"  *** New best: mean_rel_l2 = {best_score:.4f}")

    print(f"\nBest config: {best_config_dict}")
    return best_config_dict, all_results


def train_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    ensemble_size: int,
    device: torch.device,
) -> Tuple[List[nn.Module], StandardScaler]:
    """Train an ensemble of models on all data.

    Returns:
        (list_of_models, fitted_scaler)
    """
    # Determine if dual-input splitting is needed
    tglf_dim = None
    if model_config.model_type != "mlp" and model_config.param_dim > 0:
        tglf_dim = model_config.input_dim - model_config.param_dim

    # Fit scaler on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y_transformed = apply_target_transform(
        targets, train_config.target_transform, train_config.epsilon
    )

    # Create dataset
    if tglf_dim is not None:
        ds = DualInputFluxDataset(
            X_scaled[:, :tglf_dim], X_scaled[:, tglf_dim:], y_transformed)
    else:
        ds = FluxDataset(X_scaled, y_transformed)
    loader = DataLoader(ds, batch_size=train_config.batch_size, shuffle=True)

    models = []
    for member_idx in range(ensemble_size):
        set_seed(train_config.seed + member_idx * 100)
        print(f"  Training ensemble member {member_idx + 1}/{ensemble_size}")

        model = create_model(
            model_type=model_config.model_type,
            tglf_dim=model_config.input_dim - model_config.param_dim,
            param_dim=model_config.param_dim,
            hidden_dims=model_config.hidden_dims,
            dropout=model_config.dropout,
            n_outputs=model_config.n_outputs,
            use_softplus=model_config.use_softplus,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config.epochs)
        loss_fn = _get_loss_fn(train_config.target_transform, train_config.epsilon)

        # Train for fixed epochs (no val set for stopping)
        for epoch in range(train_config.epochs):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                pred, y = _model_forward(model, batch, device)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        models.append(model)

    return models, scaler


def predict_ensemble(
    models: List[nn.Module],
    features: np.ndarray,
    scaler: StandardScaler,
    target_transform: str,
    epsilon: float,
    device: torch.device,
    tglf_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ensemble predictions with uncertainty.

    Args:
        models: List of trained models.
        features: Raw feature array (N, D).
        scaler: Fitted StandardScaler.
        target_transform: Target transform name.
        epsilon: Numerical stability constant.
        device: torch device.
        tglf_dim: If set and models are dual-input, split features at this index.

    Returns:
        (mean_predictions(N,2), std_predictions(N,2)) in original scale.
    """
    X_scaled = scaler.transform(features)

    is_dual = getattr(models[0], 'dual_input', False) if models else False

    if is_dual and tglf_dim is not None:
        X_tglf = torch.tensor(X_scaled[:, :tglf_dim], dtype=torch.float32).to(device)
        X_params = torch.tensor(X_scaled[:, tglf_dim:], dtype=torch.float32).to(device)
    else:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            if is_dual and tglf_dim is not None:
                pred = model(X_tglf, X_params).cpu().numpy()
            else:
                pred = model(X_tensor).cpu().numpy()
        pred_original = invert_target_transform(pred, target_transform, epsilon)
        all_preds.append(pred_original)

    all_preds = np.stack(all_preds, axis=0)  # (n_members, N, 2)
    mean_pred = np.mean(all_preds, axis=0)   # (N, 2)
    std_pred = np.std(all_preds, axis=0)     # (N, 2)

    return mean_pred, std_pred


def save_ensemble(
    models: List[nn.Module],
    scaler: StandardScaler,
    config_dict: dict,
    save_dir: str,
):
    """Save ensemble models, scaler, and config to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Save models
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{i}.pt"))

    # Save scaler
    import pickle
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved {len(models)} models + scaler + config to {save_dir}")
