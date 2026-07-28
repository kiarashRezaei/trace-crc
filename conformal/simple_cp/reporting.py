"""
Formatted prints matching the notebook outputs for each scenario.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _print_header(title: str) -> None:
    bar = "=" * 60
    print("\n" + bar)
    print(f" {title} ")
    print(bar)


def print_marginal_report(result: Dict[str, object]) -> None:
    _print_header("MARGINAL CONFORMAL PREDICTION")
    target = (1 - float(result["alpha"])) * 100
    print(f"Target coverage (1 - α)        = {target:.2f}%")
    print(f"Empirical calibration coverage = {float(result['coverage_cal'])*100:.2f}%")
    print(f"Empirical test coverage        = {float(result['coverage_test'])*100:.2f}%")
    print(f"λ (half-width)                 = {float(result['lambda']):.4f}")
    print(f"Average interval length        = {float(result['avg_length']):.4f}")


def print_cqr_report(result: Dict[str, object]) -> None:
    _print_header("ROUTE A: CONFORMALIZED QUANTILE REGRESSION (CQR)")
    target = (1 - float(result["alpha"])) * 100
    print(f"Target coverage (1 - α)        = {target:.2f}%")
    print(f"Empirical marginal coverage     = {float(result['coverage_test'])*100:.2f}%")
    print(f"Average interval length (CQR)   = {float(result['avg_length']):.4f}")

    bin_source = result.get("bin_stats_source", "quantiles")
    if bin_source == "groups":
        print("\nApproximate conditional coverage by merged groups:")
    else:
        print("\nApproximate conditional coverage by X-bin:")
        K = int(result.get("K", 0))
        if K:
            print(f"(Using K = {K} equally populated bins)")
        else:
            print("(No bins available)")

    bin_stats: List[dict] = result["bin_stats"]
    if not bin_stats:
        label = "groups" if bin_source == "groups" else "bins"
        print(f"  No non-empty {label} to report.")
    else:
        for stats in bin_stats:
            if bin_source == "groups" and "group" in stats:
                label = f"Group {stats['group']}"
            else:
                label = f"Bin {stats.get('bin', stats.get('group', '?'))}"

            x_min = stats.get("x_min")
            x_max = stats.get("x_max")
            avg_width = stats.get("avg_width", stats.get("avg_length", float("nan")))
            count = stats.get("count", stats.get("group_size", 0))
            coverage_pct = float(stats.get("coverage", 0.0)) * 100

            if x_min is None or x_max is None or np.any(np.isnan([x_min, x_max])):
                print(
                    f"  {label}: coverage ≈ {coverage_pct:.2f}% | avg width = {avg_width:.4f} "
                    f"| n = {count}"
                )
            else:
                print(
                    f"  {label}: X in [{x_min:.3f}, {x_max:.3f}], "
                    f"coverage ≈ {coverage_pct:.2f}% | avg width = {avg_width:.4f} "
                    f"| n = {count}"
                )

        if result["avg_cov_bins"] is not None:
            header = "MERGED GROUPS" if bin_source == "groups" else "X-BINS"
            print("\n" + "-" * 60)
            print(f" AVERAGE OVER {header} ")
            print("-" * 60)
            print(f"Average empirical coverage       = {float(result['avg_cov_bins'])*100:.2f}%")
            print(f"Average interval length          = {float(result['avg_len_bins']):.4f}")


def print_ltt_report(result: Dict[str, object]) -> None:
    lambda_best = float(result["lambda_best"])
    alpha = float(result["alpha"])
    delta = float(result["delta"])
    lambda_grid = np.asarray(result["lambda_grid"])
    R_hat = np.asarray(result["R_hat"])
    R_upper = np.asarray(result["R_upper"])

    idx_best = int(np.abs(lambda_grid - lambda_best).argmin())
    empirical_noncoverage = float(R_hat[idx_best])
    empirical_coverage = 1 - empirical_noncoverage
    certified_coverage = float(result["coverage_certified"])
    coverage_cal = float(result["coverage_cal"])

    print("Best (narrowest) λ with certified non-coverage ≤ α:")
    print(f"lambda_best = {lambda_best:.4f}")
    print(f"Empirical non-coverage at λ_best: {empirical_noncoverage:.4f}")
    print(f"Uniform upper bound at λ_best   : {float(R_upper[idx_best]):.4f}")
    print(f"Empirical coverage at λ_best     : {empirical_coverage*100:.2f}%")
    print(
        f"LTT-certified coverage ≥         : {certified_coverage*100:.2f}% "
        f"(with probability ≥ {1 - delta:.2f})"
    )

    print(
        f"\nEmpirical calibration coverage at λ_best : {coverage_cal*100:.2f}%"
    )

    _print_header("GLOBAL LTT MARGINAL COVERAGE SUMMARY")
    print(f"Target non-coverage α        = {alpha:.3f}")
    print(f"Target coverage              = {(1 - alpha)*100:.2f}%")
    print(f"LTT meta level δ             = {delta:.3f}")
    print(f"Guarantee holds with prob ≥  = {(1 - delta)*100:.2f}%")

    print("\nAt the chosen λ_best:")
    print(f"  λ_best                     = {lambda_best:.4f}")
    print(f"  Empirical calibration cov  = {coverage_cal*100:.2f}%")
    print(
        f"  LTT-certified coverage ≥   = {certified_coverage*100:.2f}% "
        f"(with probability ≥ {(1 - delta)*100:.2f}%)"
    )
    print(f"  Average interval length    = {2 * lambda_best:.4f}")
    print(f"  Number of safe λ values    = {int(result['num_safe_lambdas'])}")

    group_stats = result.get("group_stats", [])
    if group_stats:
        _print_header("GROUP-WISE COVERAGE AND INTERVAL STATISTICS")
        for stats in group_stats:
            print(f"\nGroup {stats.get('group', 0)} (n={stats['group_size']}):")
            if stats.get("lambda_best") is None:
                print("  No certified λ — cannot compute certified coverage.")
                continue
            print(f"  λ_best                     = {stats['lambda_best']:.4f}")
            print(f"  Empirical coverage         = {stats['empirical_coverage']*100:.2f}%")
            print(
                f"  LTT-certified coverage ≥   = {stats['certified_coverage']*100:.2f}% "
                f"(with prob ≥ {(1 - delta):.2f})"
            )
            print(f"  Avg interval length        = {stats['avg_length']:.4f}")

    print("=" * 60 + "\n")


def print_training_report(result: Dict[str, object]) -> None:
    _print_header("MODEL TRAINING SUMMARY")
    print(f"Target type                = {result['target']}")
    if "model" in result:
        print(f"Model family               = {result['model']}")
    print(f"Training file              = {result['train_data']}")
    print(f"Validation file            = {result['val_data']}")
    if result["target"] == "quantile":
        print(f"Alpha (target non-coverage)= {result['alpha']:.3f}")
    print(f"Artifacts saved under      = {result['output_dir']}")
    params = result.get("model_params") or {}
    if params:
        print("\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    train_losses = result.get("train_mse") or {}
    if train_losses:
        print("\nTraining MSE:")
        for name, value in train_losses.items():
            print(f"  {name:<25}= {value:.6f}")
    val_losses = result.get("val_mse")
    if isinstance(val_losses, dict) and val_losses:
        print("\nValidation MSE:")
        for name, value in val_losses.items():
            print(f"  {name:<25}= {value:.6f}")

    artifacts = result.get("artifacts", {})
    if artifacts:
        print("\nSaved model files:")
        for name, path in artifacts.items():
            print(f"  - {name}: {path}")
    if "summary_path" in result:
        print(f"\nSummary file: {result['summary_path']}")


def print_single_empirical_report(result: Dict[str, object]) -> None:
    _print_header("SINGLE-MODEL APPROXIMATE CONDITIONAL (EMPIRICAL)")
    lam = float(result["lambda"])
    print(f"Global half-width λ           = {lam:.4f}")
    print(f"Calibration coverage          = {result['coverage_cal']*100:.2f}%")
    print(f"Test coverage                 = {result['coverage_test']*100:.2f}%")
    print(f"Min per-group coverage target = {result['min_coverage']*100:.2f}% "
          f"(tolerance ±{result['coverage_tolerance']*100:.2f}%)")

    group_stats = result.get("group_stats", [])
    if group_stats:
        print("\nPer-group stats:")
        for stats in group_stats:
            print(
                f"  Group {stats['group']}: "
                f"coverage={stats['coverage']*100:.2f}% "
                f"avg_len={stats['avg_length']:.4f} "
                f"n={stats['group_size']}"
            )

    print("\n" + "=" * 60)


def print_data_report(summary: Dict[str, object]) -> None:
    _print_header("DATA PIPELINE SUMMARY")
    print(f"Source               = {summary['generator']}")
    print(f"Total samples        = {summary['n_samples']}")
    print(f"Split function       = {summary['split_fn']}")
    print(f"Train/Val/Cal/Test   = {summary['n_train']} / {summary['n_val']} / {summary['n_cal']} / {summary['n_test']}")
    split_paths = summary.get("split_paths", {})
    if split_paths:
        print("\nSaved splits:")
        for name, path in split_paths.items():
            print(f"  {name:<5} -> {path}")
    if "summary_path" in summary:
        print(f"\nSummary file: {summary['summary_path']}")


def print_evaluation_report(summary: Dict[str, object]) -> None:
    _print_header("MODEL EVALUATION SUMMARY")
    print(f"Target type     = {summary['target']}")
    print(f"Test data       = {summary['test_data']}")
    for metric, value in summary.get("metrics", {}).items():
        print(f"{metric.upper():<15}= {value:.6f}")
    if "summary_path" in summary:
        print(f"\nSummary file: {summary['summary_path']}")

