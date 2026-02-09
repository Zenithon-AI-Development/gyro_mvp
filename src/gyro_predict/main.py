"""CLI entry point for gyrokinetic flux prediction MVP."""

import argparse
import os

import numpy as np
import torch

from .config import Config, ModelConfig, TrainConfig
from .data_loading import load_all_data, load_run_list
from .evaluate import (
    compare_to_tglf_baseline,
    ensemble_uncertainty_plot,
    error_by_regime,
    error_distribution_plot,
    parity_plot,
    summarize_results,
)
from .train import (
    predict_ensemble,
    run_hyperparameter_search,
    run_kfold,
    save_ensemble,
    set_seed,
    train_ensemble,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gyrokinetic flux prediction: TGLF -> CGYRO Q"
    )
    parser.add_argument(
        "--mode",
        choices=["search", "train", "eval"],
        default="search",
        help="search: grid search HP | train: train with given config | eval: evaluate saved models",
    )
    parser.add_argument("--data_dir", default="/data")
    parser.add_argument("--checkpoint_dir", default="/checkpoints/gyro_predict")
    parser.add_argument("--csv_dir", default=None,
                        help="Directory containing validated_parameters.csv")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--target_transform", default="raw",
                        choices=["raw", "log", "power"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dims", default="256,128,64",
                        help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # WandB
    parser.add_argument("--wandb_entity", default="PLACEHOLDER_ENTITY")
    parser.add_argument("--wandb_project", default="PLACEHOLDER_PROJECT")
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = Config()
    config.paths.data_dir = args.data_dir
    config.paths.checkpoint_dir = args.checkpoint_dir
    if args.csv_dir:
        config.paths.csv_dir = args.csv_dir

    config.train.n_folds = args.n_folds
    config.train.seed = args.seed
    config.train.epochs = args.epochs
    config.train.patience = args.patience
    config.train.batch_size = args.batch_size

    config.wandb.entity = args.wandb_entity
    config.wandb.project = args.wandb_project
    config.wandb.enabled = not args.no_wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    csv_path = os.path.join(config.paths.csv_dir, config.paths.validated_csv)
    run_list = load_run_list(csv_path)
    print(f"Run list: {len(run_list)} entries")

    features, targets, metadata = load_all_data(config.paths.data_dir, run_list)
    experiment_groups = metadata["experiment"].values
    print(f"Features: {features.shape}, Targets: {targets.shape}")
    print(f"Experiment groups: {np.unique(experiment_groups, return_counts=True)}")

    # --- Init WandB ---
    if config.wandb.enabled:
        import wandb
        print(f"WandB: entity={config.wandb.entity}, project={config.wandb.project}")

    # === MODE: search ===
    if args.mode == "search":
        print("\n=== HYPERPARAMETER SEARCH ===")
        best_config_dict, all_results = run_hyperparameter_search(
            features, targets, experiment_groups, config, device
        )

        # Save search results
        import pandas as pd
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(config.paths.checkpoint_dir, "search_results.csv")
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nSearch results saved to {results_path}")
        print(f"Best config: {best_config_dict}")

        # Run final K-fold with best config to get detailed evaluation
        print("\n=== FINAL EVALUATION WITH BEST CONFIG ===")
        hidden_dims = best_config_dict["hidden_dims"]
        mc = ModelConfig(
            input_dim=config.features.total_features,
            hidden_dims=list(hidden_dims),
            dropout=best_config_dict["dropout"],
        )
        tc = TrainConfig(
            lr=best_config_dict["lr"],
            weight_decay=best_config_dict["weight_decay"],
            target_transform=best_config_dict["target_transform"],
            epochs=config.train.epochs,
            patience=config.train.patience,
            n_folds=config.train.n_folds,
            batch_size=config.train.batch_size,
            seed=config.train.seed,
        )

        kfold_result = run_kfold(
            features, targets, experiment_groups, mc, tc, device
        )

        # Generate evaluation plots
        plot_dir = os.path.join(config.paths.checkpoint_dir, "plots")
        oof_preds = kfold_result["oof_predictions"]

        parity_plot(oof_preds, targets, experiment_groups,
                    os.path.join(plot_dir, "parity_oof.png"))
        error_distribution_plot(oof_preds, targets,
                                os.path.join(plot_dir, "error_dist_oof.png"))
        error_by_regime(oof_preds, targets, experiment_groups,
                        os.path.join(plot_dir, "error_by_regime.png"))

        # TGLF comparison
        flux_csv = os.path.join(config.paths.csv_dir, config.paths.flux_comparison_csv)
        tglf_comp = compare_to_tglf_baseline(oof_preds, targets, flux_csv)

        # Save summary
        summarize_results(
            kfold_result, best_config_dict, tglf_comp,
            os.path.join(config.paths.checkpoint_dir, "results_summary.json"),
        )

        # Train final ensemble
        print("\n=== TRAINING FINAL ENSEMBLE ===")
        for ens_size in config.search.ensemble_size:
            print(f"\nEnsemble size: {ens_size}")
            models, scaler = train_ensemble(
                features, targets, mc, tc, ens_size, device
            )

            save_dir = os.path.join(
                config.paths.checkpoint_dir, f"ensemble_{ens_size}"
            )
            config_to_save = {**best_config_dict, "ensemble_size": ens_size}
            save_ensemble(models, scaler, config_to_save, save_dir)

    # === MODE: train ===
    elif args.mode == "train":
        print("\n=== TRAINING WITH SPECIFIED CONFIG ===")
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
        mc = ModelConfig(
            input_dim=config.features.total_features,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
        tc = TrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            target_transform=args.target_transform,
            epochs=config.train.epochs,
            patience=config.train.patience,
            n_folds=config.train.n_folds,
            batch_size=config.train.batch_size,
            seed=config.train.seed,
        )

        # K-fold evaluation
        wandb_run = None
        if config.wandb.enabled:
            import wandb
            wandb_run = wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                group="v1-mlp-train",
                config={
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                    "hidden_dims": hidden_dims,
                    "target_transform": args.target_transform,
                    "ensemble_size": args.ensemble_size,
                },
            )

        kfold_result = run_kfold(
            features, targets, experiment_groups, mc, tc, device, wandb_run
        )

        print(f"\nK-fold mean_rel_l2: {kfold_result['mean_rel_l2']:.4f} "
              f"+/- {kfold_result['std_rel_l2']:.4f}")

        # Plots
        plot_dir = os.path.join(config.paths.checkpoint_dir, "plots")
        oof_preds = kfold_result["oof_predictions"]
        parity_plot(oof_preds, targets, experiment_groups,
                    os.path.join(plot_dir, "parity_oof.png"))
        error_distribution_plot(oof_preds, targets,
                                os.path.join(plot_dir, "error_dist_oof.png"))
        error_by_regime(oof_preds, targets, experiment_groups,
                        os.path.join(plot_dir, "error_by_regime.png"))

        # TGLF comparison
        flux_csv = os.path.join(config.paths.csv_dir, config.paths.flux_comparison_csv)
        tglf_comp = compare_to_tglf_baseline(oof_preds, targets, flux_csv)

        # Train ensemble
        print(f"\n=== TRAINING ENSEMBLE (size={args.ensemble_size}) ===")
        models, scaler = train_ensemble(
            features, targets, mc, tc, args.ensemble_size, device
        )

        save_dir = os.path.join(config.paths.checkpoint_dir, "ensemble")
        config_dict = {
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "hidden_dims": hidden_dims,
            "target_transform": args.target_transform,
            "ensemble_size": args.ensemble_size,
        }
        save_ensemble(models, scaler, config_dict, save_dir)

        # Ensemble predictions on training data (sanity check)
        mean_pred, std_pred = predict_ensemble(
            models, features, scaler, args.target_transform, tc.epsilon, device
        )
        ensemble_uncertainty_plot(mean_pred, std_pred, targets,
                                  os.path.join(plot_dir, "ensemble_uncertainty.png"))

        # Summary
        summarize_results(
            kfold_result, config_dict, tglf_comp,
            os.path.join(config.paths.checkpoint_dir, "results_summary.json"),
        )

        if wandb_run is not None:
            wandb_run.finish()

    # === MODE: eval ===
    elif args.mode == "eval":
        print("\n=== EVALUATION MODE ===")
        # Load ensemble from checkpoint_dir/ensemble/
        import json
        import pickle
        from .model import FluxMLP

        ensemble_dir = os.path.join(config.paths.checkpoint_dir, "ensemble")
        with open(os.path.join(ensemble_dir, "config.json")) as f:
            saved_config = json.load(f)
        with open(os.path.join(ensemble_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        hidden_dims = saved_config["hidden_dims"]
        models = []
        i = 0
        while os.path.exists(os.path.join(ensemble_dir, f"model_{i}.pt")):
            model = FluxMLP(
                input_dim=config.features.total_features,
                hidden_dims=hidden_dims,
                dropout=saved_config["dropout"],
            ).to(device)
            model.load_state_dict(
                torch.load(os.path.join(ensemble_dir, f"model_{i}.pt"),
                            map_location=device)
            )
            model.eval()
            models.append(model)
            i += 1

        print(f"Loaded {len(models)} ensemble members")

        tt = saved_config["target_transform"]
        eps = config.train.epsilon
        mean_pred, std_pred = predict_ensemble(
            models, features, scaler, tt, eps, device
        )

        from .loss import compute_metrics as _compute_metrics
        metrics = _compute_metrics(mean_pred, targets, eps)
        print(f"Ensemble metrics: {metrics}")

        # All plots
        plot_dir = os.path.join(config.paths.checkpoint_dir, "plots")
        parity_plot(mean_pred, targets, experiment_groups,
                    os.path.join(plot_dir, "parity_ensemble.png"))
        error_distribution_plot(mean_pred, targets,
                                os.path.join(plot_dir, "error_dist_ensemble.png"))
        error_by_regime(mean_pred, targets, experiment_groups,
                        os.path.join(plot_dir, "error_by_regime_ensemble.png"))
        ensemble_uncertainty_plot(mean_pred, std_pred, targets,
                                  os.path.join(plot_dir, "ensemble_uncertainty.png"))

        flux_csv = os.path.join(config.paths.csv_dir, config.paths.flux_comparison_csv)
        compare_to_tglf_baseline(mean_pred, targets, flux_csv)


if __name__ == "__main__":
    main()
