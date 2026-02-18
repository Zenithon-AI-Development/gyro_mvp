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
from .model import create_model
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

    # CSV and feature options
    parser.add_argument("--csv_file", type=str, default="validated_parameters_cleaned.csv",
                        help="CSV file to load (in csvs/ dir)")
    parser.add_argument("--global_params", type=str, default="",
                        help="Comma-separated global param column names (empty=TGLF-only)")
    parser.add_argument("--use_softplus", action="store_true", default=True,
                        help="Apply Softplus to output to ensure Q > 0 (default: True)")
    parser.add_argument("--no_softplus", dest="use_softplus", action="store_false",
                        help="Disable Softplus output activation")
    parser.add_argument("--split_column", type=str, default="",
                        help="Column in CSV for train/ood split. "
                             "If set, trains on 'train' rows, evaluates on 'ood' rows.")

    # Model architecture
    parser.add_argument("--model_type", type=str, default="mlp",
                        choices=["mlp", "film", "hadamard", "bilinear",
                                 "structured_mlp", "structured_hadamard"],
                        help="Model architecture: mlp (flat), film (FiLM), hadamard (dual-head), "
                             "bilinear (dual-head), structured_mlp (ky-level), "
                             "structured_hadamard (ky-level + params)")
    parser.add_argument("--feature_type", type=str, default="flat",
                        choices=["flat", "structured"],
                        help="Feature representation: flat (378-dim aggregated) or "
                             "structured (21x18 ky-channel, stored flat)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "silu"],
                        help="Activation function for hidden layers")
    parser.add_argument("--no_kfold", action="store_true",
                        help="Skip K-fold CV, train ensemble directly on train split")
    parser.add_argument("--target_from_csv", action="store_true",
                        help="Load targets from Q_electron/Q_ion CSV columns instead of .h5 files")

    # WandB
    parser.add_argument("--wandb_entity", default="PLACEHOLDER_ENTITY")
    parser.add_argument("--wandb_project", default="PLACEHOLDER_PROJECT")
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def _count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    # --- Parse global params ---
    global_param_columns = [x.strip() for x in args.global_params.split(",") if x.strip()] if args.global_params else []
    config.features.global_param_columns = global_param_columns
    param_dim = len(global_param_columns)
    tglf_dim = 378
    input_dim = tglf_dim + param_dim
    print(f"Feature config: {tglf_dim} TGLF + {param_dim} global params = {input_dim} dims")
    print(f"Model type: {args.model_type}")
    if global_param_columns:
        print(f"Global params: {global_param_columns}")

    # --- Load data ---
    csv_path = os.path.join(config.paths.csv_dir, args.csv_file)
    run_list = load_run_list(csv_path)
    print(f"Run list: {len(run_list)} entries from {args.csv_file}")

    features, targets, metadata = load_all_data(
        config.paths.data_dir, run_list, global_param_columns,
        feature_builder=args.feature_type,
        use_csv_targets=args.target_from_csv,
    )
    experiment_groups = metadata["experiment"].values
    print(f"Features: {features.shape}, Targets: {targets.shape}")
    print(f"Experiment groups: {np.unique(experiment_groups, return_counts=True)}")

    # --- Split data if split_column is specified ---
    split_column = args.split_column.strip() if args.split_column else ""
    train_mask = None
    ood_mask = None
    ood_features = ood_targets = ood_metadata = ood_experiment_groups = None

    if split_column and split_column in metadata.columns:
        train_mask = metadata[split_column].values == "train"
        ood_mask = metadata[split_column].values == "ood"
        print(f"\nSplit column '{split_column}': {train_mask.sum()} train, {ood_mask.sum()} OOD")

        # Store OOD subset before overwriting
        ood_features = features[ood_mask]
        ood_targets = targets[ood_mask]
        ood_metadata = metadata[ood_mask].reset_index(drop=True)
        ood_experiment_groups = ood_metadata["experiment"].values

        # Overwrite with train-only (all downstream code uses these unmodified)
        features = features[train_mask]
        targets = targets[train_mask]
        metadata = metadata[train_mask].reset_index(drop=True)
        experiment_groups = metadata["experiment"].values
        print(f"Training subset: {features.shape}, OOD subset: {ood_features.shape}")
    elif split_column:
        print(f"WARNING: split_column '{split_column}' not found in metadata")

    # --- Init WandB ---
    if config.wandb.enabled:
        import wandb
        print(f"WandB: entity={config.wandb.entity}, project={config.wandb.project}")

    # === MODE: search ===
    if args.mode == "search":
        print("\n=== HYPERPARAMETER SEARCH ===")
        best_config_dict, all_results = run_hyperparameter_search(
            features, targets, experiment_groups, metadata, config, device
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
            features, targets, experiment_groups, metadata, mc, tc, device
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
            kfold_result, best_config_dict, tglf_comp, metadata,
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
        print(f"\n=== TRAINING WITH {input_dim}-dim inputs ({args.model_type}) ===")
        print(f"Hyperparameters: {args.hidden_dims}, dropout={args.dropout}, lr={args.lr}, "
              f"weight_decay={args.weight_decay}, {args.target_transform} transform")

        # Parse hidden dims from CLI
        hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

        mc = ModelConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            use_softplus=args.use_softplus,
            model_type=args.model_type,
            param_dim=param_dim,
            feature_type=args.feature_type,
            activation=args.activation,
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

        # Count parameters for a dummy model
        dummy_model = create_model(
            model_type=args.model_type,
            tglf_dim=tglf_dim,
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            use_softplus=args.use_softplus,
            activation=args.activation,
        )
        num_params = _count_parameters(dummy_model)
        print(f"Model parameters: {num_params:,}")
        del dummy_model

        # WandB init
        wandb_run = None
        if config.wandb.enabled:
            import wandb
            wandb_run = wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                group=f"{args.model_type}-{input_dim}dims",
                name=f"train_{args.model_type}_{input_dim}dims",
                config={
                    "model_type": args.model_type,
                    "input_dim": input_dim,
                    "tglf_dim": tglf_dim,
                    "param_dim": param_dim,
                    "global_params": global_param_columns,
                    "hidden_dims": hidden_dims,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "target_transform": args.target_transform,
                    "ensemble_size": args.ensemble_size,
                    "num_params": num_params,
                    "feature_type": args.feature_type,
                    "activation": args.activation,
                },
            )

        # K-fold evaluation (optional)
        kfold_result = None
        tglf_comp = None
        plot_dir = os.path.join(config.paths.checkpoint_dir, "plots")

        if not args.no_kfold:
            kfold_result = run_kfold(
                features, targets, experiment_groups, metadata, mc, tc, device, wandb_run
            )

            print(f"\nK-fold mean_rel_l2: {kfold_result['mean_rel_l2']:.4f} "
                  f"+/- {kfold_result['std_rel_l2']:.4f}")

            # Log k-fold metrics to wandb
            if wandb_run is not None:
                oof_m = kfold_result["oof_metrics"]
                wandb_run.summary.update({
                    "kfold/rel_l2_mean": kfold_result["mean_rel_l2"],
                    "kfold/rel_l2_std": kfold_result["std_rel_l2"],
                    "kfold/mape_electron": oof_m["mape_electron"],
                    "kfold/mape_ion": oof_m["mape_ion"],
                    "kfold/mae_electron": oof_m["mae_electron"],
                    "kfold/mae_ion": oof_m["mae_ion"],
                    "kfold/rmse_electron": oof_m["rmse_electron"],
                    "kfold/rmse_ion": oof_m["rmse_ion"],
                })

            # Plots
            oof_preds = kfold_result["oof_predictions"]
            parity_plot(oof_preds, targets, experiment_groups,
                        os.path.join(plot_dir, "parity_oof.png"))
            error_distribution_plot(oof_preds, targets,
                                    os.path.join(plot_dir, "error_dist_oof.png"))
            error_by_regime(oof_preds, targets, experiment_groups,
                            os.path.join(plot_dir, "error_by_regime.png"))

            # TGLF comparison
            flux_csv = os.path.join(config.paths.csv_dir, config.paths.flux_comparison_csv)
            if os.path.exists(flux_csv):
                tglf_comp = compare_to_tglf_baseline(oof_preds, targets, flux_csv)
            else:
                print(f"Skipping TGLF baseline comparison ({flux_csv} not found)")
        else:
            print("Skipping K-fold (--no_kfold)")

        # Train ensemble
        print(f"\n=== TRAINING ENSEMBLE (size={args.ensemble_size}) ===")
        models, scaler = train_ensemble(
            features, targets, mc, tc, args.ensemble_size, device,
            wandb_run=wandb_run,
        )

        save_dir = os.path.join(config.paths.checkpoint_dir, "ensemble")
        config_dict = {
            "model_type": args.model_type,
            "input_dim": input_dim,
            "tglf_dim": tglf_dim,
            "param_dim": param_dim,
            "global_params": global_param_columns,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "use_softplus": mc.use_softplus,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "target_transform": args.target_transform,
            "ensemble_size": args.ensemble_size,
            "num_params": num_params,
            "feature_type": args.feature_type,
            "activation": args.activation,
        }
        save_ensemble(models, scaler, config_dict, save_dir)

        # Determine tglf_dim for predict_ensemble
        from .train import DUAL_INPUT_TYPES
        ens_tglf_dim = tglf_dim if args.model_type in DUAL_INPUT_TYPES and param_dim > 0 else None

        # Ensemble predictions on training data (sanity check)
        mean_pred, std_pred = predict_ensemble(
            models, features, scaler, tc.target_transform, tc.epsilon, device,
            tglf_dim=ens_tglf_dim,
        )
        ensemble_uncertainty_plot(mean_pred, std_pred, targets,
                                  os.path.join(plot_dir, "ensemble_uncertainty.png"))

        # Summary (only if kfold was run)
        if kfold_result is not None:
            summarize_results(
                kfold_result, config_dict, tglf_comp or {}, metadata,
                os.path.join(config.paths.checkpoint_dir, "results_summary.json"),
            )

        if wandb_run is not None:
            wandb_run.finish()

        # === OOD EVALUATION (if split_column was used) ===
        if split_column and ood_features is not None and len(ood_features) > 0:
            print(f"\n=== OOD EVALUATION ({len(ood_features)} runs) ===")

            ood_mean_pred, ood_std_pred = predict_ensemble(
                models, ood_features, scaler, tc.target_transform, tc.epsilon, device,
                tglf_dim=ens_tglf_dim,
            )

            from .loss import compute_metrics as _compute_metrics
            ood_metrics = _compute_metrics(ood_mean_pred, ood_targets, tc.epsilon)
            print(f"OOD metrics: {ood_metrics}")

            # Compute per-run MAPE for outlier detection
            per_run_mape = []
            for i in range(len(ood_targets)):
                run_metrics = _compute_metrics(
                    ood_mean_pred[i:i+1], ood_targets[i:i+1], tc.epsilon)
                avg_mape = (run_metrics["mape_electron"] + run_metrics["mape_ion"]) / 2
                per_run_mape.append(avg_mape)
            max_mape = max(per_run_mape) if per_run_mape else 0.0
            print(f"OOD max single-run MAPE: {max_mape:.1f}%")

            # Log OOD metrics to wandb
            if config.wandb.enabled:
                import wandb
                ood_wandb_run = wandb.init(
                    entity=config.wandb.entity,
                    project=config.wandb.project,
                    group=f"{args.model_type}-{input_dim}dims",
                    name=f"ood_{args.model_type}_{input_dim}dims",
                    config=config_dict,
                )
                ood_wandb_run.summary.update({
                    "ood/rel_l2_mean": ood_metrics["rel_l2_mean"],
                    "ood/rel_l2_electron": ood_metrics["rel_l2_electron"],
                    "ood/rel_l2_ion": ood_metrics["rel_l2_ion"],
                    "ood/mape_electron": ood_metrics["mape_electron"],
                    "ood/mape_ion": ood_metrics["mape_ion"],
                    "ood/mae_electron": ood_metrics["mae_electron"],
                    "ood/mae_ion": ood_metrics["mae_ion"],
                    "ood/rmse_electron": ood_metrics["rmse_electron"],
                    "ood/rmse_ion": ood_metrics["rmse_ion"],
                    "ood/n_runs": len(ood_features),
                    "ood/max_mape": max_mape,
                })
                ood_wandb_run.finish()

            # OOD plots (separate directory)
            ood_plot_dir = os.path.join(config.paths.checkpoint_dir, "plots_ood")
            parity_plot(ood_mean_pred, ood_targets, ood_experiment_groups,
                        os.path.join(ood_plot_dir, "parity_ood.png"))
            error_distribution_plot(ood_mean_pred, ood_targets,
                                    os.path.join(ood_plot_dir, "error_dist_ood.png"))
            error_by_regime(ood_mean_pred, ood_targets, ood_experiment_groups,
                            os.path.join(ood_plot_dir, "error_by_regime_ood.png"))
            ensemble_uncertainty_plot(ood_mean_pred, ood_std_pred, ood_targets,
                                      os.path.join(ood_plot_dir, "ensemble_uncertainty_ood.png"))

            # Save OOD summary JSON
            import json as _json
            ood_summary = {
                "split": "ood",
                "n_runs": len(ood_features),
                "metrics": ood_metrics,
                "max_mape": max_mape,
                "per_run_mape": per_run_mape,
                "run_names": ood_metadata["folder_name"].tolist(),
                "config": config_dict,
            }
            ood_path = os.path.join(config.paths.checkpoint_dir, "ood_results_summary.json")
            os.makedirs(os.path.dirname(ood_path), exist_ok=True)
            with open(ood_path, "w") as f:
                _json.dump(ood_summary, f, indent=2)
            print(f"OOD results saved to {ood_path}")

    # === MODE: eval ===
    elif args.mode == "eval":
        print("\n=== EVALUATION MODE ===")
        # Load ensemble from checkpoint_dir/ensemble/ or directly from checkpoint_dir
        import json
        import pickle
        from .model import FluxMLP, create_model as _create_model

        # Try checkpoint_dir/ensemble first (new structure), then checkpoint_dir directly (V1 structure)
        ensemble_dir = os.path.join(config.paths.checkpoint_dir, "ensemble")
        if not os.path.exists(os.path.join(ensemble_dir, "model_0.pt")):
            # Fall back to checkpoint_dir directly (V1 model structure)
            ensemble_dir = config.paths.checkpoint_dir
            print(f"Using checkpoint_dir directly: {ensemble_dir}")

        config_json_path = os.path.join(ensemble_dir, "config.json")

        # Load saved model config
        if os.path.exists(config_json_path):
            with open(config_json_path) as f:
                saved_config = json.load(f)
            model_input_dim = saved_config.get("input_dim", 378)
            saved_model_type = saved_config.get("model_type", "mlp")
            saved_param_dim = saved_config.get("param_dim", 0)
            saved_tglf_dim = saved_config.get("tglf_dim", model_input_dim)
            print(f"Loaded model config: model_type={saved_model_type}, input_dim={model_input_dim}")
        else:
            # Fallback: infer from first model weight matrix
            model_path = os.path.join(ensemble_dir, "model_0.pt")
            checkpoint = torch.load(model_path, map_location=device)
            model_input_dim = checkpoint['fc1.weight'].shape[1]
            print(f"Inferred input_dim={model_input_dim} from model weights")
            saved_config = {"input_dim": model_input_dim, "hidden_dims": [512, 256, 128], "dropout": 0.2}
            saved_model_type = "mlp"
            saved_param_dim = 0
            saved_tglf_dim = model_input_dim

        # Verify data features match model input_dim
        if features.shape[1] != model_input_dim:
            print(f"ERROR: Feature dimension mismatch!")
            print(f"  Data features: {features.shape[1]} dims")
            print(f"  Model expects: {model_input_dim} dims")
            print(f"  Loaded with global_params={global_param_columns}")
            raise ValueError(f"Feature dimension mismatch: {features.shape[1]} != {model_input_dim}")

        with open(os.path.join(ensemble_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        hidden_dims = saved_config["hidden_dims"]
        models = []
        i = 0
        while os.path.exists(os.path.join(ensemble_dir, f"model_{i}.pt")):
            model = _create_model(
                model_type=saved_model_type,
                tglf_dim=saved_tglf_dim,
                param_dim=saved_param_dim,
                hidden_dims=hidden_dims,
                dropout=saved_config["dropout"],
                use_softplus=saved_config.get("use_softplus", False),
                activation=saved_config.get("activation", "relu"),
            ).to(device)
            model.load_state_dict(
                torch.load(os.path.join(ensemble_dir, f"model_{i}.pt"),
                            map_location=device)
            )
            model.eval()
            models.append(model)
            i += 1

        print(f"Loaded {len(models)} ensemble members, model_type={saved_model_type}, input_dim={model_input_dim}")

        tt = saved_config["target_transform"]
        eps = config.train.epsilon
        _dual_types = {"film", "hadamard", "bilinear", "structured_hadamard"}
        eval_tglf_dim = saved_tglf_dim if saved_model_type in _dual_types and saved_param_dim > 0 else None

        mean_pred, std_pred = predict_ensemble(
            models, features, scaler, tt, eps, device, tglf_dim=eval_tglf_dim,
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

        # Optional TGLF baseline comparison (skip if flux_comparison.csv doesn't exist)
        flux_csv = os.path.join(config.paths.csv_dir, config.paths.flux_comparison_csv)
        if os.path.exists(flux_csv):
            print("\n=== TGLF Baseline Comparison ===")
            compare_to_tglf_baseline(mean_pred, targets, flux_csv)
        else:
            print(f"\nSkipping TGLF baseline comparison ({config.paths.flux_comparison_csv} not found)")


if __name__ == "__main__":
    main()
