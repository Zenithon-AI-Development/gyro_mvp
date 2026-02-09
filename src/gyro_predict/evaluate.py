"""Evaluation: plots, TGLF baseline comparison, result summaries."""

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loss import compute_metrics


def parity_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    experiment_groups: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Scatter plot of predicted vs true Q for each species."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    species_names = ["Electron Q", "Ion Q"]

    for s, (ax, name) in enumerate(zip(axes, species_names)):
        p = predictions[:, s]
        t = targets[:, s]

        if experiment_groups is not None:
            unique_groups = np.unique(experiment_groups)
            for group in unique_groups:
                mask = experiment_groups == group
                # Shorten label for legend
                short = group.split("-", 1)[-1] if "-" in group else group
                ax.scatter(t[mask], p[mask], alpha=0.7, s=40, label=short)
            ax.legend(fontsize=7, loc="upper left")
        else:
            ax.scatter(t, p, alpha=0.7, s=40)

        # Perfect prediction line
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Metrics annotation
        rel_errors = np.abs(p - t) / np.maximum(np.abs(t), 1e-6) * 100
        ax.set_xlabel("True Q")
        ax.set_ylabel("Predicted Q")
        ax.set_title(f"{name} (MAPE: {np.mean(rel_errors):.1f}%)")

    fig.suptitle("Parity Plot: Model Predictions vs CGYRO Truth", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved parity plot to {save_path}")
    plt.close(fig)


def error_distribution_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
):
    """Histogram of per-sample relative errors for each species."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    species_names = ["Electron", "Ion"]

    for s, (ax, name) in enumerate(zip(axes, species_names)):
        rel_errors = np.abs(predictions[:, s] - targets[:, s]) / np.maximum(
            np.abs(targets[:, s]), 1e-6
        ) * 100
        ax.hist(rel_errors, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(rel_errors), color="red", linestyle="--",
                    label=f"Mean: {np.mean(rel_errors):.1f}%")
        ax.axvline(np.median(rel_errors), color="blue", linestyle="--",
                    label=f"Median: {np.median(rel_errors):.1f}%")
        ax.set_xlabel("Relative Error (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Flux Relative Error Distribution")
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_by_regime(
    predictions: np.ndarray,
    targets: np.ndarray,
    experiment_groups: np.ndarray,
    save_path: Optional[str] = None,
):
    """Box plots of relative errors grouped by experiment type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    species_names = ["Electron", "Ion"]
    unique_groups = sorted(np.unique(experiment_groups))

    for s, (ax, name) in enumerate(zip(axes, species_names)):
        rel_errors = np.abs(predictions[:, s] - targets[:, s]) / np.maximum(
            np.abs(targets[:, s]), 1e-6
        ) * 100

        data_by_group = []
        labels = []
        for group in unique_groups:
            mask = experiment_groups == group
            data_by_group.append(rel_errors[mask])
            short = group.split("-", 1)[-1] if "-" in group else group
            labels.append(short)

        ax.boxplot(data_by_group, labels=labels)
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(f"{name} Error by Experiment Regime")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensemble_uncertainty_plot(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
):
    """Error bar plot: do ensemble disagreements correlate with actual errors?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    species_names = ["Electron", "Ion"]

    for s, (ax, name) in enumerate(zip(axes, species_names)):
        actual_error = np.abs(mean_pred[:, s] - targets[:, s])
        uncertainty = std_pred[:, s]

        ax.scatter(uncertainty, actual_error, alpha=0.6, s=40)
        ax.set_xlabel("Ensemble Std (Uncertainty)")
        ax.set_ylabel("Actual Absolute Error")
        ax.set_title(f"{name}: Uncertainty vs Error")

        # Correlation
        if len(uncertainty) > 2:
            corr = np.corrcoef(uncertainty, actual_error)[0, 1]
            ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.95),
                        xycoords="axes fraction", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_to_tglf_baseline(
    model_predictions: np.ndarray,
    targets: np.ndarray,
    flux_comparison_csv: str,
    epsilon: float = 1e-6,
) -> dict:
    """Compare model errors to TGLF baseline errors.

    Returns dict with side-by-side comparison.
    """
    df = pd.read_csv(flux_comparison_csv)

    tglf_mape_elec = float(df["electron_pct_diff"].abs().mean())
    tglf_mape_ion = float(df["ion_pct_diff"].abs().mean())

    model_mape_elec = float(
        np.mean(np.abs(model_predictions[:, 0] - targets[:, 0])
                / np.maximum(np.abs(targets[:, 0]), epsilon)) * 100
    )
    model_mape_ion = float(
        np.mean(np.abs(model_predictions[:, 1] - targets[:, 1])
                / np.maximum(np.abs(targets[:, 1]), epsilon)) * 100
    )

    comparison = {
        "tglf_mape_electron": tglf_mape_elec,
        "tglf_mape_ion": tglf_mape_ion,
        "model_mape_electron": model_mape_elec,
        "model_mape_ion": model_mape_ion,
        "improvement_electron_pct": (tglf_mape_elec - model_mape_elec) / tglf_mape_elec * 100,
        "improvement_ion_pct": (tglf_mape_ion - model_mape_ion) / tglf_mape_ion * 100,
    }

    print("\n=== TGLF Baseline Comparison ===")
    print(f"  Electron MAPE: TGLF={tglf_mape_elec:.1f}%  Model={model_mape_elec:.1f}%  "
          f"Improvement={comparison['improvement_electron_pct']:.1f}%")
    print(f"  Ion MAPE:      TGLF={tglf_mape_ion:.1f}%  Model={model_mape_ion:.1f}%  "
          f"Improvement={comparison['improvement_ion_pct']:.1f}%")

    return comparison


def summarize_results(
    kfold_results: dict,
    best_config: dict,
    tglf_comparison: dict,
    save_path: str,
) -> dict:
    """Create and save a JSON summary of all results."""
    summary = {
        "best_hyperparameters": best_config,
        "kfold_mean_rel_l2": kfold_results["mean_rel_l2"],
        "kfold_std_rel_l2": kfold_results["std_rel_l2"],
        "oof_metrics": kfold_results["oof_metrics"],
        "tglf_comparison": tglf_comparison,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results summary to {save_path}")

    return summary
