"""
Command-line entry point exposing the modular pipelines.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from .data_generation import available_data_generators
from .data_pipeline import (
    DataPipelineConfig,
    available_split_strategies,
    run_data_pipeline,
)
from .evaluation import EvaluationConfig, run_evaluation
from .io_utils import ensure_dir, load_split, load_model_bundle, compute_ensemble_predictions
from .lambda_functions import available_lambda_functions
from .exporters import save_band_csv, save_report_csv, save_summary_csv, save_mu_band_csv
from .empirical_risk import coverage_rate
from .pipelines.conditional import (
    ConditionalConfig,
    SingleModelConfig,
    run_empirical_cqr,
    run_ltt_certified,
    run_single_empirical,
    run_single_ltt,
    run_single_empirical_multi,
    run_single_ltt_multi,
)
from .pipelines.marginal import MarginalConfig, run_marginal, run_marginal_ltt
from .plotting import (
    plot_cqr_results,
    plot_cqr_results_multi,
    plot_ltt_results,
    plot_ltt_results_multi,
    plot_marginal_results,
    plot_marginal_results_multi,
    plot_single_empirical_results,
    plot_single_empirical_results_multi,
)
from .reporting import (
    print_cqr_report,
    print_data_report,
    print_ltt_report,
    print_marginal_report,
    print_training_report,
    print_evaluation_report,
    print_single_empirical_report,
)
from .models import ModelBundle
from .training import TrainingConfig, run_training


def _prepare_output_dir(root: str, pipeline: str) -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"{pipeline}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ensure_dir(run_dir)
    return run_dir


def _parse_params(pairs: list[str] | None) -> Dict[str, object]:
    """
    Convert CLI inputs of the form KEY=VALUE into dictionaries, attempting to
    cast values to bool/int/float when appropriate.
    """
    params: Dict[str, object] = {}
    if not pairs:
        return params
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid parameter '{pair}'. Expected KEY=VALUE.")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        value: object
        lowered = raw_value.strip().lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip()
        params[key] = value
    return params


def _build_plot_options(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    """
    Extract multi-dimensional plotting controls from CLI arguments, ensuring
    validity of the provided range/max settings.
    """
    index_range: Optional[Tuple[int, int]] = None
    if args.plot_index_range:
        start, end = args.plot_index_range
        if end <= start:
            raise ValueError("--plot-index-range requires END > START.")
        if start < 0:
            raise ValueError("--plot-index-range START must be non-negative.")
        index_range = (start, end)

    max_points = args.plot_max_points
    if max_points is not None and max_points <= 0:
        raise ValueError("--plot-max-points must be a positive integer.")

    if index_range is None and max_points is None:
        return None
    return {"index_range": index_range, "max_points": max_points}


# Use shared model loading function from io_utils
_load_model_bundle = load_model_bundle


def _save_mu_bands_if_ensemble(
    quantile_low_path: str,
    quantile_high_path: str,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """
    Detect if models are ensembles and save mu-band CSV files.
    
    Mu-band is computed as mean ± std of ensemble predictions.
    """
    import json
    
    # Check if models are ensembles by examining the paths
    low_path_obj = Path(quantile_low_path)
    high_path_obj = Path(quantile_high_path)
    
    # Check for ensemble subdirectory or training_summary.json
    ensemble_dir_low = None
    ensemble_dir_high = None
    
    if low_path_obj.is_dir():
        ensemble_subdir = low_path_obj / "ensemble"
        if ensemble_subdir.exists() and ensemble_subdir.is_dir():
            ensemble_dir_low = ensemble_subdir
        else:
            # Check training_summary.json
            summary_path = low_path_obj / "training_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                artifacts = summary.get("artifacts", {})
                if "quantile_low_ensemble" in artifacts:
                    first_path = Path(artifacts["quantile_low_ensemble"][0])
                    ensemble_dir_low = first_path.parent
    
    if high_path_obj.is_dir():
        ensemble_subdir = high_path_obj / "ensemble"
        if ensemble_subdir.exists() and ensemble_subdir.is_dir():
            ensemble_dir_high = ensemble_subdir
        else:
            # Check training_summary.json
            summary_path = high_path_obj / "training_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                artifacts = summary.get("artifacts", {})
                if "quantile_high_ensemble" in artifacts:
                    first_path = Path(artifacts["quantile_high_ensemble"][0])
                    ensemble_dir_high = first_path.parent
    
    # If both are ensembles, compute mu-band statistics
    if ensemble_dir_low is not None and ensemble_dir_high is not None:
        # Compute ensemble predictions for low and high quantiles
        low_preds_cal = compute_ensemble_predictions(ensemble_dir_low, "quantile_low", X_cal)
        high_preds_cal = compute_ensemble_predictions(ensemble_dir_high, "quantile_high", X_cal)
        low_preds_test = compute_ensemble_predictions(ensemble_dir_low, "quantile_low", X_test)
        high_preds_test = compute_ensemble_predictions(ensemble_dir_high, "quantile_high", X_test)
        
        # Compute mean predictions (point estimate)
        pred_cal = 0.5 * (low_preds_cal.mean(axis=0) + high_preds_cal.mean(axis=0))
        pred_test = 0.5 * (low_preds_test.mean(axis=0) + high_preds_test.mean(axis=0))
        
        # Compute mu-band: mean ± std across ensemble members
        # For quantile models, we compute std of the midpoint predictions
        midpoint_preds_cal = 0.5 * (low_preds_cal + high_preds_cal)
        midpoint_preds_test = 0.5 * (low_preds_test + high_preds_test)
        
        std_cal = midpoint_preds_cal.std(axis=0)
        std_test = midpoint_preds_test.std(axis=0)
        
        upper_band_cal = pred_cal + std_cal
        lower_band_cal = pred_cal - std_cal
        upper_band_test = pred_test + std_test
        lower_band_test = pred_test - std_test
        
        # Save mu-band CSV files at the same level as the cp_... folder (parent of output_dir)
        mu_dir = ensure_dir(output_dir.parent / "mu")
        save_mu_band_csv(
            mu_dir / "mu_band_calibration.csv",
            model_name,
            Y_cal,
            pred_cal,
            upper_band_cal,
            lower_band_cal,
        )
        save_mu_band_csv(
            mu_dir / "mu_band_test.csv",
            model_name,
            Y_test,
            pred_test,
            upper_band_test,
            lower_band_test,
        )


def _save_mu_bands_if_regressor_ensemble(
    regressor_path: str,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """
    Detect if regressor model is an ensemble and save mu-band CSV files.
    
    Mu-band is computed as mean ± std of ensemble predictions.
    """
    import json
    
    regressor_path_obj = Path(regressor_path)
    ensemble_dir = None
    
    # Check for ensemble subdirectory
    if regressor_path_obj.is_dir():
        ensemble_subdir = regressor_path_obj / "ensemble"
        if ensemble_subdir.exists() and ensemble_subdir.is_dir():
            ensemble_dir = ensemble_subdir
        else:
            # Check training_summary.json
            summary_path = regressor_path_obj / "training_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                artifacts = summary.get("artifacts", {})
                if "regressor_ensemble" in artifacts:
                    first_path = Path(artifacts["regressor_ensemble"][0])
                    ensemble_dir = first_path.parent
    
    # If it's an ensemble, compute mu-band statistics
    if ensemble_dir is not None:
        # Compute ensemble predictions
        preds_cal = compute_ensemble_predictions(ensemble_dir, "regressor", X_cal)
        preds_test = compute_ensemble_predictions(ensemble_dir, "regressor", X_test)
        
        # Compute mean predictions (point estimate)
        mean_preds_cal = preds_cal.mean(axis=0)
        mean_preds_test = preds_test.mean(axis=0)
        
        # Compute mu-band: mean ± std across ensemble members
        std_cal = preds_cal.std(axis=0)
        std_test = preds_test.std(axis=0)
        
        upper_band_cal = mean_preds_cal + std_cal
        lower_band_cal = mean_preds_cal - std_cal
        upper_band_test = mean_preds_test + std_test
        lower_band_test = mean_preds_test - std_test
        
        # Save mu-band CSV files at the same level as the cp_... folder (parent of output_dir)
        mu_dir = ensure_dir(output_dir.parent / "mu")
        save_mu_band_csv(
            mu_dir / "mu_band_calibration.csv",
            model_name,
            Y_cal,
            mean_preds_cal,
            upper_band_cal,
            lower_band_cal,
        )
        save_mu_band_csv(
            mu_dir / "mu_band_test.csv",
            model_name,
            Y_test,
            mean_preds_test,
            upper_band_test,
            lower_band_test,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple conformal prediction runner.")
    parser.add_argument(
        "--pipeline",
        choices=["data", "train", "eval-model", "marginal", "conditional"],
        default="conditional",
        help="Which pipeline to execute.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Target non-coverage level.")
    parser.add_argument("--delta", type=float, default=0.05, help="Failure probability for LTT.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data generation/shuffling.")
    parser.add_argument("--n-samples", type=int, default=5000, help="Total synthetic samples.")
    parser.add_argument("--n-train", type=int, default=2000, help="Training size.")
    parser.add_argument("--n-cal", type=int, default=2000, help="Calibration size.")
    parser.add_argument("--n-val", type=int, default=0, help="Validation size for data/training workflows.")
    parser.add_argument(
        "--data-npz",
        type=str,
        default=None,
        help="Optional path to a real dataset (.npz with X and Y).",
    )
    parser.add_argument(
        "--data-fn",
        choices=available_data_generators(),
        default="linear_sine",
        help="Synthetic data generator for --pipeline data.",
    )
    parser.add_argument(
        "--data-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override generator parameters (repeatable).",
    )
    parser.add_argument(
        "--split-fn",
        choices=available_split_strategies(),
        default="sequential",
        help="How to split the dataset in the data pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root directory for saved figures/results.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save notebook-style plots under the output directory.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (may require GUI backend).",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=None,
        help="For multi-dimensional plots, downsample to at most this many sorted points.",
    )
    parser.add_argument(
        "--plot-index-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="For multi-dimensional plots, only visualize sorted indices in [START, END).",
    )
    parser.add_argument(
        "--grouping",
        choices=["quantile", "adaptive", "tree", "kmeans"],
        default="quantile",
        help="Grouping strategy for conditional pipelines.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=4,
        help="Number of bins for quantile grouping.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for kmeans grouping.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=5,
        help="Maximum leaf nodes for tree-based grouping.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=100,
        help="Minimum samples per leaf for tree-based grouping.",
    )
    parser.add_argument(
        "--lambda-fn",
        choices=list(available_lambda_functions()),
        default="grid_from_errors",
        help="Which lambda_functions builder to use for λ grids (used in LTT-certified modes).",
    )
    parser.add_argument(
        "--lambda-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override parameters for the selected lambda function (repeatable).",
    )
    parser.add_argument(
        "--mode",
        choices=["empirical", "ltt"],
        default="empirical",
        help="Guarantee style for conformal pipelines (empirical vs LTT-certified).",
    )
    parser.add_argument(
        "--conditional-family",
        choices=["normal", "cqr"],
        default="normal",
        help="Modeling family for the conditional pipeline (single regressor vs CQR).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="Minimum per-group coverage target for single-model empirical mode.",
    )
    parser.add_argument(
        "--coverage-tol",
        type=float,
        default=0.02,
        help="Tolerance for per-group coverage when merging bins.",
    )
    parser.add_argument(
        "--train-target",
        choices=["regressor", "quantile"],
        default="regressor",
        help="What to train when --pipeline train is selected.",
    )
    parser.add_argument(
        "--regressor-model",
        choices=["linear", "gbr", "xgb"],
        default="linear",
        help="Regressor family for the training pipeline.",
    )
    parser.add_argument(
        "--quantile-model",
        choices=["gbr", "xgb"],
        default="gbr",
        help="Quantile estimator for the training pipeline.",
    )
    parser.add_argument(
        "--regressor-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra parameters for the chosen regressor (repeatable).",
    )
    parser.add_argument(
        "--quantile-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra parameters for the chosen quantile model (repeatable).",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to train split (.npz) produced by the data pipeline.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation split (.npz) produced by the data pipeline.",
    )
    parser.add_argument(
        "--tl-data",
        type=str,
        default=None,
        help="Path to transfer learning split (.npz) for fine-tuning models.",
    )
    parser.add_argument(
        "--tl-add-trees",
        type=int,
        default=200,
        help="Number of additional trees to add during fine-tuning (XGBoost/GBR only).",
    )
    parser.add_argument(
        "--tl-eta",
        type=float,
        default=0.05,
        help="Learning rate (eta) for fine-tuning phase (XGBoost/GBR only).",
    )
    parser.add_argument(
        "--mu-band",
        action="store_true",
        help="Enable model uncertainty band by training ensemble with different seeds.",
    )
    parser.add_argument(
        "--mu-n-seeds",
        type=int,
        default=5,
        help="Number of ensemble members (seeds) for mu-band (default: 5).",
    )
    parser.add_argument(
        "--mu-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of seeds for ensemble (overrides --mu-n-seeds).",
    )
    parser.add_argument(
        "--mu-bootstrap",
        action="store_true",
        help="Bootstrap TL data for each ensemble member (only used with --tl-data).",
    )
    parser.add_argument(
        "--mu-bootstrap-frac",
        type=float,
        default=1.0,
        help="Fraction of TL data to use in bootstrap (default: 1.0).",
    )
    parser.add_argument(
        "--cal-data",
        type=str,
        default=None,
        help="Path to calibration split (.npz) for conformal pipelines.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test split (.npz) for evaluation/conformal pipelines.",
    )
    parser.add_argument(
        "--regressor-path",
        type=str,
        default=None,
        help="Optional path to a pretrained point regressor (.joblib) for marginal pipeline.",
    )
    parser.add_argument(
        "--quantile-low-path",
        type=str,
        default=None,
        help="Optional path to a pretrained lower quantile model (.joblib).",
    )
    parser.add_argument(
        "--quantile-high-path",
        type=str,
        default=None,
        help="Optional path to a pretrained upper quantile model (.joblib).",
    )
    args = parser.parse_args()
    lambda_params = _parse_params(args.lambda_param)
    regressor_params = _parse_params(args.regressor_param)
    quantile_params = _parse_params(args.quantile_param)
    plot_options = _build_plot_options(args)
    data_params = _parse_params(args.data_param)

    run_dir = _prepare_output_dir(args.output_dir, args.pipeline)
    
    # Extract dataset name: "calib" if calibration data, "test" if test data
    def _extract_dataset_name(
        cal_data: Optional[str] = None,
        test_data: Optional[str] = None,
    ) -> str:
        # Check calibration data path first
        if cal_data:
            cal_path_lower = cal_data.lower()
            if "calib" in cal_path_lower or "cal" in cal_path_lower:
                return "calib"
        
        # Check test data path
        if test_data:
            test_path_lower = test_data.lower()
            if "test" in test_path_lower:
                return "test"
        
        # Default to "calib" if we have cal_data, otherwise "test"
        if cal_data:
            return "calib"
        elif test_data:
            return "test"
        return "unknown"
    
    dataset_name = _extract_dataset_name(args.cal_data, args.test_data)
    
    # Helper to compute test statistics per group from group_test_bands
    def _compute_test_stats_from_bands(group_test_bands: list, group_stats: list) -> list:
        """Compute test statistics per group from group_test_bands."""
        test_stats = []
        # Create a map from group_id to test bands
        bands_map = {band.get("group", i): band for i, band in enumerate(group_test_bands)}
        
        for g_stat in group_stats:
            group_id = g_stat.get("group") or g_stat.get("bin", 0)
            test_band = bands_map.get(group_id)
            
            if test_band:
                Y_test_g = np.asarray(test_band["Y"]).ravel()
                lower_test_g = np.asarray(test_band["lower"]).ravel()
                upper_test_g = np.asarray(test_band["upper"]).ravel()
                in_band_test_g = np.asarray(test_band["in_band"]).ravel()
                
                coverage_test = float(in_band_test_g.mean()) if len(in_band_test_g) > 0 else float("nan")
                avg_len_test = float((upper_test_g - lower_test_g).mean()) if len(upper_test_g) > 0 else float("nan")
            else:
                coverage_test = float("nan")
                avg_len_test = float("nan")
            
            test_stats.append({
                "group": group_id,
                "coverage": coverage_test,
                "avg_length": avg_len_test,
            })
        
        return test_stats

    if args.pipeline == "data":
        cfg = DataPipelineConfig(
            output_dir=run_dir,
            n_samples=args.n_samples,
            n_train=args.n_train,
            n_val=args.n_val,
            n_cal=args.n_cal,
            seed=args.seed,
            data_fn=args.data_fn if args.data_npz is None else args.data_fn,
            data_params=data_params,
            data_npz=args.data_npz,
            split_fn=args.split_fn,
        )
        summary = run_data_pipeline(cfg)
        print_data_report(summary)
        return

    if args.pipeline == "train":
        if not args.train_data:
            raise ValueError("--train-data is required for the training pipeline.")
        
        # Generate seeds for mu-band if enabled
        mu_seeds = None
        if args.mu_band:
            if args.mu_seeds:
                mu_seeds = args.mu_seeds
            else:
                # Generate n_seeds starting from base seed
                mu_seeds = [args.seed + i * 11 for i in range(args.mu_n_seeds)]
        
        cfg = TrainingConfig(
            train_data=args.train_data,
            val_data=args.val_data,
            tl_data=args.tl_data,
            alpha=args.alpha,
            target=args.train_target,
            regressor_model=args.regressor_model,
            regressor_params=regressor_params,
            quantile_model=args.quantile_model,
            quantile_params=quantile_params,
            output_dir=run_dir,
            seed=args.seed,
            tl_add_trees=args.tl_add_trees,
            tl_eta=args.tl_eta,
            mu_band=args.mu_band,
            mu_seeds=mu_seeds,
            mu_bootstrap=args.mu_bootstrap,
            mu_bootstrap_frac=args.mu_bootstrap_frac,
        )
        summary = run_training(cfg)
        print_training_report(summary)
        return

    if args.pipeline == "eval-model":
        if not args.test_data:
            raise ValueError("--test-data is required for model evaluation.")
        eval_cfg = EvaluationConfig(
            target=args.train_target,
            test_data=args.test_data,
            regressor_path=args.regressor_path,
            quantile_low_path=args.quantile_low_path,
            quantile_high_path=args.quantile_high_path,
            output_dir=run_dir,
        )
        summary = run_evaluation(eval_cfg)
        print_evaluation_report(summary)
        return

    # Conformal pipelines
    if not args.cal_data or not args.test_data:
        raise ValueError("--cal-data and --test-data are required for conformal pipelines.")
    X_cal, Y_cal = load_split(args.cal_data)
    X_test, Y_test = load_split(args.test_data)

    if args.pipeline == "marginal":
        if not args.regressor_path:
            raise ValueError("--regressor-path is required for the marginal pipeline.")
        model_bundle = _load_model_bundle(args.regressor_path, "regressor")
        conformal_dir = ensure_dir(run_dir / "conformal")
        cfg = MarginalConfig(
            X_cal=X_cal,
            Y_cal=Y_cal,
            X_test=X_test,
            Y_test=Y_test,
            alpha=args.alpha,
            model_bundle=model_bundle,
            delta=args.delta,
            lambda_fn=args.lambda_fn,
            lambda_params=lambda_params,
        )
        is_multidim = X_test.ndim == 2 and X_test.shape[1] > 1
        if args.mode == "empirical":
            summary = run_marginal(cfg)
            print_marginal_report(summary)
            model_name = model_bundle.name
            avg_len_cal = float((summary["upper_cal"] - summary["lower_cal"]).mean())
            save_summary_csv(
                conformal_dir / "marginal_summary.csv",
                model_name,
                summary["alpha"],
                None,  # delta not used in empirical
                [],  # Empty group_stats for marginal (single global group)
                is_empirical=True,
                global_empirical_coverage_cal=summary["coverage_cal"],
                global_empirical_coverage_test=summary["coverage_test"],
                global_lambda=summary["lambda"],
                global_avg_length_cal=avg_len_cal,
                global_avg_length_test=summary["avg_length"],
            )
            save_band_csv(
                conformal_dir / "marginal_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                summary["Y_hat_cal"],
                summary["lower_cal"],
                summary["upper_cal"],
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "marginal_test_bands.csv",
                model_name,
                summary["Y_test"],
                summary["Y_hat_test"],
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if model is an ensemble
            _save_mu_bands_if_regressor_ensemble(
                args.regressor_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
        else:
            summary = run_marginal_ltt(cfg)
            print_ltt_report(summary)
            model_name = model_bundle.name
            # Calculate average interval length for marginal LTT
            avg_len_cal = float((summary["upper_cal"] - summary["lower_cal"]).mean())
            avg_len_test = float((summary["upper_test"] - summary["lower_test"]).mean())
            save_summary_csv(
                conformal_dir / "marginal_ltt_summary.csv",
                model_name,
                summary["alpha"],
                summary["delta"],
                [],  # Empty group_stats for marginal (single global group)
                is_empirical=False,
                global_empirical_coverage_cal=summary["coverage_cal"],
                global_empirical_coverage_test=summary["coverage_test"],
                global_ltt_coverage_cal=summary["coverage_certified"],
                global_ltt_coverage_test=summary["coverage_certified"],  # Same certified coverage for test
                global_lambda=summary["lambda_best"],
                global_num_safe_lambdas=summary["num_safe_lambdas"],
                global_avg_length_cal=avg_len_cal,
                global_avg_length_test=avg_len_test,
            )
            save_band_csv(
                conformal_dir / "marginal_ltt_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                summary["y_hat_cal"],
                summary["lower_cal"],
                summary["upper_cal"],
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "marginal_ltt_test_bands.csv",
                model_name,
                summary["Y_test"],
                summary["y_hat_test"],
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if model is an ensemble
            _save_mu_bands_if_regressor_ensemble(
                args.regressor_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
            if args.save_plots or args.show_plots:
                if is_multidim:
                    plot_ltt_results_multi(summary, conformal_dir, args.show_plots, plot_options)
                else:
                    plot_ltt_results(summary, conformal_dir, args.show_plots)
        return

    # Conditional pipeline
    conformal_dir = ensure_dir(run_dir / "conformal")
    if args.conditional_family == "normal":
        if not args.regressor_path:
            raise ValueError("--regressor-path is required for conditional normal mode.")
        model_bundle = _load_model_bundle(args.regressor_path, "regressor")
        single_cfg = SingleModelConfig(
            X_cal=X_cal,
            Y_cal=Y_cal,
            X_test=X_test,
            Y_test=Y_test,
            model=model_bundle,
            alpha=args.alpha,
            delta=args.delta,
            grouping=args.grouping,
            n_bins=args.n_bins,
            n_clusters=args.n_clusters,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            min_coverage=args.min_coverage,
            coverage_tolerance=args.coverage_tol,
            lambda_fn=args.lambda_fn,
            lambda_params=lambda_params,
        )
        is_multidim = X_test.ndim == 2 and X_test.shape[1] > 1
        if args.mode == "empirical":
            summary = run_single_empirical_multi(single_cfg) if is_multidim else run_single_empirical(single_cfg)
            print_single_empirical_report(summary)
            model_name = model_bundle.name
            group_stats = summary.get("group_stats", [])
            group_test_stats = _compute_test_stats_from_bands(
                summary.get("group_test_bands", []),
                group_stats
            )
            save_summary_csv(
                conformal_dir / "conditional_normal_empirical_summary.csv",
                model_name,
                summary["alpha"],
                None,  # delta not used in empirical
                group_stats,
                group_test_stats=group_test_stats,
                is_empirical=True,
                min_coverage=args.min_coverage,
                coverage_tolerance=args.coverage_tol,
            )
            lam = summary["lambda"]
            lower_cal = summary["y_hat_cal"] - lam
            upper_cal = summary["y_hat_cal"] + lam
            save_band_csv(
                conformal_dir / "conditional_normal_empirical_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                summary["y_hat_cal"],
                lower_cal,
                upper_cal,
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "conditional_normal_empirical_test_bands.csv",
                model_name,
                summary["Y_test"],
                summary["y_hat_test"],
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if model is an ensemble
            _save_mu_bands_if_regressor_ensemble(
                args.regressor_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
        else:
            summary = run_single_ltt_multi(single_cfg) if is_multidim else run_single_ltt(single_cfg)
            print_ltt_report(summary)
            model_name = model_bundle.name
            group_stats = summary.get("group_stats", [])
            group_test_stats = _compute_test_stats_from_bands(
                summary.get("group_test_bands", []),
                group_stats
            )
            # Pass global num_safe_lambdas to be used for all groups
            save_summary_csv(
                conformal_dir / "conditional_normal_ltt_summary.csv",
                model_name,
                summary["alpha"],
                summary["delta"],
                group_stats,
                group_test_stats=group_test_stats,
                is_empirical=False,
                global_num_safe_lambdas=summary.get("num_safe_lambdas"),
                min_coverage=args.min_coverage,
                coverage_tolerance=args.coverage_tol,
            )
            save_band_csv(
                conformal_dir / "conditional_normal_ltt_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                summary["y_hat_cal"],
                summary["lower_cal"],
                summary["upper_cal"],
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "conditional_normal_ltt_test_bands.csv",
                model_name,
                summary["Y_test"],
                summary["y_hat_test"],
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if model is an ensemble
            _save_mu_bands_if_regressor_ensemble(
                args.regressor_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
            if args.save_plots or args.show_plots:
                if is_multidim:
                    plot_ltt_results_multi(summary, conformal_dir, args.show_plots, plot_options)
                else:
                    plot_ltt_results(summary, conformal_dir, args.show_plots)
    else:
        if not (args.quantile_low_path and args.quantile_high_path):
            raise ValueError("Provide both --quantile-low-path and --quantile-high-path for conditional CQR mode.")
        quantile_models = (
            _load_model_bundle(args.quantile_low_path, "quantile_low"),
            _load_model_bundle(args.quantile_high_path, "quantile_high"),
        )
        cond_cfg = ConditionalConfig(
            X_cal=X_cal,
            Y_cal=Y_cal,
            X_test=X_test,
            Y_test=Y_test,
            alpha=args.alpha,
            delta=args.delta,
            grouping=args.grouping,
            n_bins=args.n_bins,
            n_clusters=args.n_clusters,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            lambda_fn=args.lambda_fn,
            lambda_params=lambda_params,
            quantile_models=quantile_models,
            min_coverage=args.min_coverage,
            coverage_tolerance=args.coverage_tol,
            seed=args.seed,
        )
        is_multidim = X_test.ndim == 2 and X_test.shape[1] > 1
        if args.mode == "empirical":
            summary = run_empirical_cqr(cond_cfg)
            print_cqr_report(summary)
            model_name = f"{quantile_models[0].name}|{quantile_models[1].name}"
            bin_stats = summary.get("bin_stats", [])
            # For CQR, bin_stats contains both cal and test stats, but we need to separate them
            # Check if there's group_test_stats or compute from test data
            group_test_stats = None
            if "group_test_stats" in summary:
                group_test_stats = summary["group_test_stats"]
            else:
                # Compute from test data using group labels
                group_labels_test = summary.get("group_labels_test")
                if group_labels_test is not None:
                    group_test_stats = []
                    for bin_stat in bin_stats:
                        group_id = bin_stat.get("group") or bin_stat.get("bin", 0)
                        mask = (group_labels_test == group_id)
                        if np.any(mask):
                            lower_test_g = summary["lower_test"][mask]
                            upper_test_g = summary["upper_test"][mask]
                            Y_test_g = summary["Y_test"][mask]
                            coverage_test = coverage_rate(lower_test_g, upper_test_g, Y_test_g)
                            avg_len_test = float((upper_test_g - lower_test_g).mean())
                            group_test_stats.append({
                                "group": group_id,
                                "coverage": float(coverage_test),
                                "avg_length": avg_len_test,
                            })
            save_summary_csv(
                conformal_dir / "conditional_cqr_empirical_summary.csv",
                model_name,
                summary["alpha"],
                None,  # delta not used in empirical
                bin_stats,  # Use bin_stats which has group info
                group_test_stats=group_test_stats,
                is_empirical=True,
                min_coverage=args.min_coverage,
                coverage_tolerance=args.coverage_tol,
            )
            pred_cal = 0.5 * (summary["q_low_cal"] + summary["q_high_cal"])
            pred_test = 0.5 * (summary["q_low_test"] + summary["q_high_test"])
            save_band_csv(
                conformal_dir / "conditional_cqr_empirical_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                pred_cal,
                summary["lower_cal"],
                summary["upper_cal"],
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "conditional_cqr_empirical_test_bands.csv",
                model_name,
                summary["Y_test"],
                pred_test,
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if models are ensembles
            _save_mu_bands_if_ensemble(
                args.quantile_low_path,
                args.quantile_high_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
        else:
            summary = run_ltt_certified(cond_cfg)
            print_ltt_report(summary)
            model_name = f"{quantile_models[0].name}|{quantile_models[1].name}"
            group_stats = summary.get("group_stats", [])
            group_test_stats = _compute_test_stats_from_bands(
                summary.get("group_test_bands", []),
                group_stats
            )
            # Pass global num_safe_lambdas to be used for all groups
            save_summary_csv(
                conformal_dir / "conditional_cqr_ltt_summary.csv",
                model_name,
                summary["alpha"],
                summary["delta"],
                group_stats,
                group_test_stats=group_test_stats,
                is_empirical=False,
                global_num_safe_lambdas=summary.get("num_safe_lambdas"),
                min_coverage=args.min_coverage,
                coverage_tolerance=args.coverage_tol,
            )
            save_band_csv(
                conformal_dir / "conditional_cqr_ltt_calibration_bands.csv",
                model_name,
                summary["Y_cal"],
                summary["y_hat_cal"],
                summary["lower_cal"],
                summary["upper_cal"],
                summary.get("group_labels_cal"),
            )
            save_band_csv(
                conformal_dir / "conditional_cqr_ltt_test_bands.csv",
                model_name,
                summary["Y_test"],
                summary["y_hat_test"],
                summary["lower_test"],
                summary["upper_test"],
                summary.get("group_labels_test"),
            )
            # Save mu-band CSV files if models are ensembles
            _save_mu_bands_if_ensemble(
                args.quantile_low_path,
                args.quantile_high_path,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
                model_name,
                run_dir,
            )
            if args.save_plots or args.show_plots:
                if is_multidim:
                    plot_ltt_results_multi(summary, conformal_dir, args.show_plots, plot_options)
                else:
                    plot_ltt_results(summary, conformal_dir, args.show_plots)


if __name__ == "__main__":
    main()
