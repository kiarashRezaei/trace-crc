"""
Approximate conditional conformal pipelines: empirical (CQR-style) and
LTT-certified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from ..empirical_risk import cqr_scores, coverage_rate, empirical_risk_curve, residuals
from ..grouping import (
    adaptive_binning,
    kmeans_groups,
    kmeans_groups_multi,
    quantile_bins,
    tree_based_groups,
    tree_groups_multi,
)
from ..lambda_functions import build_lambda_grid
from ..models import ModelBundle, quantile_midpoint_predictor
from ...core.uniform_concentration import get_eta_star_upper, shat_upper_tail


GroupStrategy = Literal["quantile", "adaptive", "tree", "kmeans"]


@dataclass
class ConditionalConfig:
    X_cal: np.ndarray
    Y_cal: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    alpha: float = 0.1
    delta: float = 0.05
    grouping: GroupStrategy = "quantile"
    n_bins: int = 4
    n_clusters: int = 5
    lambda_fn: str = "grid_from_errors"
    lambda_params: Dict[str, object] = field(default_factory=dict)
    max_leaf_nodes: int = 5
    min_samples_leaf: int = 100
    quantile_models: Optional[Tuple[ModelBundle, ModelBundle]] = None
    min_coverage: float = 0.9
    coverage_tolerance: float = 0.02
    seed: int = 0


@dataclass
class SingleModelConfig:
    X_cal: np.ndarray
    Y_cal: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    model: ModelBundle
    alpha: float = 0.1
    delta: float = 0.05
    grouping: GroupStrategy = "quantile"
    n_bins: int = 4
    n_clusters: int = 5
    max_leaf_nodes: int = 5
    min_samples_leaf: int = 100
    min_coverage: float = 0.9
    coverage_tolerance: float = 0.02
    lambda_fn: str = "grid_from_errors"
    lambda_params: Dict[str, object] = field(default_factory=dict)


def _sort_groups(groups: List[np.ndarray], X_cal: np.ndarray) -> List[np.ndarray]:
    valid = [idx for idx in groups if len(idx) > 0]
    if X_cal.ndim == 1:
        coordinates = X_cal
    else:
        coordinates = X_cal[:, 0]
    return sorted(valid, key=lambda idx: float(np.mean(coordinates[idx])))


def _make_groups(config: ConditionalConfig, X_cal: np.ndarray, Y_cal: np.ndarray) -> List[np.ndarray]:
    is_multidim = X_cal.ndim == 2 and X_cal.shape[1] > 1

    # For multi-dimensional features, we only allow tree- or kmeans-based
    # grouping. Quantile/adaptive binning remain strictly 1D.
    if is_multidim and config.grouping in {"quantile", "adaptive"}:
        raise ValueError(
            "Quantile/adaptive grouping is only supported for 1D features. "
            "For multi-dimensional X, use grouping='tree' or grouping='kmeans'."
        )

    if config.grouping == "quantile":
        groups = quantile_bins(X_cal, config.n_bins)
    elif config.grouping == "adaptive":
        groups = adaptive_binning(X_cal)
    elif config.grouping == "tree":
        groups = tree_based_groups(
            X_cal,
            Y_cal,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            seed=config.seed,
        )
    elif config.grouping == "kmeans":
        groups = kmeans_groups(X_cal, config.n_clusters, min_size=config.min_samples_leaf, seed=config.seed)
    else:
        raise ValueError(f"Unsupported grouping strategy: {config.grouping}")
    return _sort_groups(groups, X_cal)


def _make_groups_multi(
    config: ConditionalConfig, X_cal: np.ndarray, Y_cal: np.ndarray, X_test: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if config.grouping not in {"tree", "kmeans"}:
        raise ValueError("Multi-dimensional CQR mode supports grouping='tree' or 'kmeans' only.")

    if config.grouping == "tree":
        return tree_groups_multi(
            X_cal,
            Y_cal,
            X_test,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            seed=config.seed,
        )
    return kmeans_groups_multi(
        X_cal,
        X_test,
        n_clusters=config.n_clusters,
        min_size=config.min_samples_leaf,
        seed=config.seed,
    )


def run_empirical_cqr(config: ConditionalConfig) -> Dict[str, object]:
    """
    Approximate conditional coverage using Conformalized Quantile Regression.
    """
    if config.quantile_models is None:
        raise ValueError("Provide pretrained quantile models (--quantile-low-path/high-path).")
    low_model, high_model = config.quantile_models

    X_cal = config.X_cal
    if X_cal.ndim == 2 and X_cal.shape[1] > 1:
        return run_empirical_cqr_multi(config)

    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()

    q_low_cal = low_model.predict(X_cal)
    q_high_cal = high_model.predict(X_cal)
    scores = cqr_scores(q_low_cal, q_high_cal, Y_cal)

    n_cal = len(scores)
    quantile_level = np.ceil((n_cal + 1) * (1 - config.alpha)) / (n_cal + 1)
    q_cqr = np.quantile(scores, quantile_level)

    q_low_test = low_model.predict(X_test)
    q_high_test = high_model.predict(X_test)
    lower_test = q_low_test - q_cqr
    upper_test = q_high_test + q_cqr
    lower_cal = q_low_cal - q_cqr
    upper_cal = q_high_cal + q_cqr

    in_band = (Y_test.ravel() >= lower_test) & (Y_test.ravel() <= upper_test)
    coverage = float(in_band.mean())
    avg_length = float((upper_test - lower_test).mean())

    groups = _make_groups(config, X_cal, Y_cal.reshape(-1, 1))
    bins = [{"idx": idx} for idx in groups if len(idx) > 0]
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)

    def compute_bin_coverage(bin_info: Dict[str, object]) -> None:
        idx = bin_info["idx"]
        if len(idx) == 0:
            bin_info["coverage"] = 1.0
            bin_info["avg_length"] = 0.0
            bin_info["x_min"] = float("nan")
            bin_info["x_max"] = float("nan")
            return
        lower_bin = lower_cal[idx]
        upper_bin = upper_cal[idx]
        bin_info["coverage"] = coverage_rate(lower_bin, upper_bin, Y_cal[idx])
        bin_info["avg_length"] = float((upper_bin - lower_bin).mean())
        X_g = np.asarray(X_cal[idx]).ravel()
        bin_info["x_min"] = float(X_g.min())
        bin_info["x_max"] = float(X_g.max())

    if bins:
        while True:
            for b in bins:
                compute_bin_coverage(b)
            failing = [
                i
                for i, b in enumerate(bins)
                if b["coverage"] < config.min_coverage - config.coverage_tolerance
            ]
            if not failing or len(bins) == 1:
                break
            bins = _merge_failing_bins(bins, failing)

        for b in bins:
            compute_bin_coverage(b)

    group_stats = []
    for g, b in enumerate(bins):
        idx = b["idx"]
        coverage_g = b.get("coverage", float("nan"))
        avg_len_g = b.get("avg_length", float("nan"))
        x_min = b.get("x_min")
        x_max = b.get("x_max")
        group_stats.append(
            {
                "group": g,
                "group_size": int(len(idx)),
                "coverage": coverage_g,
                "coverage_cal": coverage_g,
                "avg_length": avg_len_g,
                "x_min": x_min,
                "x_max": x_max,
            }
        )

    group_test_stats = []
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    X_test_flat = np.asarray(X_test).ravel()
    Y_test_flat = Y_test
    for g, stats in enumerate(group_stats):
        x_min = stats.get("x_min")
        x_max = stats.get("x_max")
        if x_min is None or x_max is None or np.isnan(x_min) or np.isnan(x_max):
            continue
        mask = (X_test_flat >= x_min) & (X_test_flat <= x_max)
        if not np.any(mask):
            continue
        lower_group = lower_test[mask]
        upper_group = upper_test[mask]
        coverage_test_g = coverage_rate(lower_group, upper_group, Y_test_flat[mask])
        avg_len_test_g = float((upper_group - lower_group).mean())
        group_labels_test[mask] = g
        group_test_stats.append(
            {
                "group": g,
                "x_min": x_min,
                "x_max": x_max,
                "coverage": float(coverage_test_g),
                "avg_width": avg_len_test_g,
                "count": int(mask.sum()),
            }
        )

    if group_test_stats:
        bin_stats = group_test_stats
        avg_cov_bins = float(np.mean([b["coverage"] for b in bin_stats])) if bin_stats else None
        avg_len_bins = float(np.mean([b["avg_width"] for b in bin_stats])) if bin_stats else None
        bin_stats_source = "groups"
        K_value = len(bin_stats)
    else:
        X_flat = X_test.ravel()
        if X_flat.size > 0:
            K = max(2, min(20, len(X_cal) // 100))
            bin_edges = np.quantile(X_flat, np.linspace(0, 1, K + 1))
            bin_indices = np.digitize(X_flat, bin_edges[1:-1], right=True)
            bin_stats = []
            for g in range(K):
                idx_g = np.where(bin_indices == g)[0]
                if len(idx_g) == 0:
                    continue
                cov_g = in_band[idx_g].mean()
                avg_len_g = float((upper_test - lower_test)[idx_g].mean())
                bin_stats.append(
                    {
                        "bin": g,
                        "x_min": float(bin_edges[g]),
                        "x_max": float(bin_edges[g + 1]),
                        "coverage": float(cov_g),
                        "avg_width": avg_len_g,
                        "count": int(len(idx_g)),
                    }
                )
            avg_cov_bins = float(np.mean([b["coverage"] for b in bin_stats])) if bin_stats else None
            avg_len_bins = float(np.mean([b["avg_width"] for b in bin_stats])) if bin_stats else None
            bin_stats_source = "quantiles"
            K_value = K if bin_stats else 0
        else:
            bin_stats = []
            avg_cov_bins = None
            avg_len_bins = None
            bin_stats_source = "quantiles"
            K_value = 0

    if bins:
        for g, b in enumerate(bins):
            group_labels_cal[b["idx"]] = g

    return {
        "type": "CQR",
        "alpha": config.alpha,
        "min_coverage": config.min_coverage,
        "coverage_tolerance": config.coverage_tolerance,
        "coverage_test": coverage,
        "avg_length": avg_length,
        "q_cqr": float(q_cqr),
        "group_stats": group_stats,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "q_low_cal": q_low_cal,
        "q_high_cal": q_high_cal,
        "q_low_test": q_low_test,
        "q_high_test": q_high_test,
        "lower_cal": lower_cal,
        "upper_cal": upper_cal,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "scores": scores,
        "bin_stats": bin_stats,
        "avg_cov_bins": avg_cov_bins,
        "avg_len_bins": avg_len_bins,
        "K": K_value,
        "bin_stats_source": bin_stats_source,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
        "group_test_stats": group_test_stats,
    }


def run_empirical_cqr_multi(config: ConditionalConfig) -> Dict[str, object]:
    """
    Multi-dimensional variant of the CQR pipeline that keeps calibration/test
    grouping aligned via tree/k-means assignments.
    """
    if config.quantile_models is None:
        raise ValueError("Provide pretrained quantile models (--quantile-low-path/high-path).")
    if config.grouping not in {"tree", "kmeans"}:
        raise ValueError("Multi-dimensional CQR supports grouping='tree' or 'kmeans' only.")

    low_model, high_model = config.quantile_models
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()

    q_low_cal = low_model.predict(X_cal)
    q_high_cal = high_model.predict(X_cal)
    scores = cqr_scores(q_low_cal, q_high_cal, Y_cal)
    n_cal = len(scores)
    quantile_level = np.ceil((n_cal + 1) * (1 - config.alpha)) / (n_cal + 1)
    q_cqr = np.quantile(scores, quantile_level)

    q_low_test = low_model.predict(X_test)
    q_high_test = high_model.predict(X_test)

    lower_cal = q_low_cal - q_cqr
    upper_cal = q_high_cal + q_cqr
    lower_test = q_low_test - q_cqr
    upper_test = q_high_test + q_cqr

    in_band = (Y_test >= lower_test) & (Y_test <= upper_test)
    coverage = float(in_band.mean())
    avg_length = float((upper_test - lower_test).mean())

    cal_groups, test_groups = _make_groups_multi(config, X_cal, Y_cal, X_test)
    bins = [
        {"cal_idx": cal_idx, "test_idx": test_idx}
        for cal_idx, test_idx in zip(cal_groups, test_groups)
        if len(cal_idx) > 0
    ]

    def compute_bin_stats(bin_info: Dict[str, object]) -> None:
        idx = bin_info["cal_idx"]
        if len(idx) == 0:
            bin_info["coverage"] = 1.0
            bin_info["avg_length"] = 0.0
            bin_info["count"] = 0
            return
        lower_bin = lower_cal[idx]
        upper_bin = upper_cal[idx]
        bin_info["coverage"] = coverage_rate(lower_bin, upper_bin, Y_cal[idx])
        bin_info["avg_length"] = float((upper_bin - lower_bin).mean())
        bin_info["count"] = int(len(idx))

    while True:
        for b in bins:
            compute_bin_stats(b)
        failing = [
            i
            for i, b in enumerate(bins)
            if b["coverage"] < config.min_coverage - config.coverage_tolerance
        ]
        if not failing or len(bins) == 1:
            break
        new_bins = _merge_bins_multi(bins, failing)
        if len(new_bins) >= len(bins):
            break
        bins = new_bins

    for b in bins:
        compute_bin_stats(b)

    group_stats = []
    bin_stats = []
    group_test_stats = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)

    for g, b in enumerate(bins):
        cal_idx = b["cal_idx"]
        test_idx = b["test_idx"]
        if len(cal_idx) == 0:
            continue

        lower_bin = lower_cal[cal_idx]
        upper_bin = upper_cal[cal_idx]
        coverage_g = coverage_rate(lower_bin, upper_bin, Y_cal[cal_idx])
        avg_len_g = float((upper_bin - lower_bin).mean())

        stats_entry = {
            "group": g,
            "group_size": int(len(cal_idx)),
            "coverage": float(coverage_g),
            "avg_length": avg_len_g,
            "x_min": float("nan"),
            "x_max": float("nan"),
        }
        group_stats.append(stats_entry)
        bin_stats.append(
            {
                "group": g,
                "x_min": float("nan"),
                "x_max": float("nan"),
                "coverage": float(coverage_g),
                "avg_width": avg_len_g,
                "count": int(len(cal_idx)),
            }
        )
        group_labels_cal[cal_idx] = g

        if len(test_idx) > 0:
            lower_group = lower_test[test_idx]
            upper_group = upper_test[test_idx]
            coverage_test_g = coverage_rate(lower_group, upper_group, Y_test[test_idx])
            avg_len_test_g = float((upper_group - lower_group).mean())
            group_test_stats.append(
                {
                    "group": g,
                    "coverage": float(coverage_test_g),
                    "avg_length": avg_len_test_g,
                    "count": int(len(test_idx)),
                }
            )
            group_labels_test[test_idx] = g

    avg_cov_bins = float(np.mean([b["coverage"] for b in bin_stats])) if bin_stats else None
    avg_len_bins = float(np.mean([b["avg_width"] for b in bin_stats])) if bin_stats else None

    return {
        "type": "CQR",
        "alpha": config.alpha,
        "min_coverage": config.min_coverage,
        "coverage_tolerance": config.coverage_tolerance,
        "coverage_test": coverage,
        "avg_length": avg_length,
        "q_cqr": float(q_cqr),
        "group_stats": group_stats,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "q_low_cal": q_low_cal,
        "q_high_cal": q_high_cal,
        "q_low_test": q_low_test,
        "q_high_test": q_high_test,
        "lower_cal": lower_cal,
        "upper_cal": upper_cal,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "scores": scores,
        "bin_stats": bin_stats,
        "avg_cov_bins": avg_cov_bins,
        "avg_len_bins": avg_len_bins,
        "K": len(bin_stats),
        "bin_stats_source": "groups",
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
        "group_test_stats": group_test_stats,
    }


def _uniform_upper_bounds(R_hat: np.ndarray, n_cal: int, m: int, alpha: float, delta: float) -> Tuple[np.ndarray, float]:
    eta_star = get_eta_star_upper(n_cal, m, alpha, delta, 100_000)
    bounds = []
    for r_hat in R_hat:
        try:
            bounds.append(shat_upper_tail(r_hat, n_cal, m, delta, eta_star, 100_000))
        except ValueError:
            bounds.append(1.0)
    return np.asarray(bounds), eta_star


def run_ltt_certified(config: ConditionalConfig) -> Dict[str, object]:
    """
    LTT-certified approximate conditional coverage following the notebook flow.
    """
    if config.quantile_models is None:
        raise ValueError("Provide pretrained quantile models (--quantile-low-path/high-path).")
    if config.X_cal.ndim == 2 and config.X_cal.shape[1] > 1:
        return run_ltt_certified_multi(config)
    low_model, high_model = config.quantile_models
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()
    predictor = quantile_midpoint_predictor(low_model, high_model)
    y_hat_cal = predictor(X_cal)
    y_hat_test = predictor(X_test)

    abs_err = residuals(Y_cal, y_hat_cal)
    lambda_grid = build_lambda_grid(
        config.lambda_fn,
        abs_err,
        config.alpha,
        config.lambda_params,
    )
    R_hat = empirical_risk_curve(abs_err, lambda_grid)
    R_upper, eta_star = _uniform_upper_bounds(R_hat, len(abs_err), len(lambda_grid), config.alpha, config.delta)

    safe_mask = R_upper <= config.alpha
    if not np.any(safe_mask):
        raise RuntimeError("No λ satisfied the LTT constraint; consider expanding the grid.")
    lambda_best = float(lambda_grid[safe_mask].min())
    idx_best = int(np.abs(lambda_grid - lambda_best).argmin())
    certified_coverage = float(1 - R_upper[idx_best])

    lower = y_hat_test - lambda_best
    upper = y_hat_test + lambda_best
    coverage = coverage_rate(lower, upper, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_best, y_hat_cal + lambda_best, Y_cal)

    # Initialize bins from grouping strategy
    initial_groups = _make_groups(config, X_cal, Y_cal)
    bins = [{"idx": idx} for idx in initial_groups if len(idx) > 0]
    
    m = len(lambda_grid)
    maxiter = 100_000

    def compute_bin_stats(bin_info: Dict[str, object]) -> None:
        """Compute R_hat, R_upper, and lambda_best for a bin."""
        idx_g = bin_info["idx"]
        n_g = len(idx_g)

        Y_g = Y_cal[idx_g].ravel()
        Yhat_g = y_hat_cal[idx_g].ravel()
        abs_err_g = np.abs(Y_g - Yhat_g)

        # Empirical risk curve in this bin
        R_hat_g = empirical_risk_curve(abs_err_g, lambda_grid)

        # LTT uniform concentration within bin
        non_empty_bins = [b for b in bins if len(b["idx"]) > 0]
        delta_bin = config.delta / max(len(non_empty_bins), 1)
        eta_star_g = get_eta_star_upper(n_g, m, config.alpha, delta_bin, maxiter)

        R_upper_g = []
        for r_hat in R_hat_g:
            try:
                R_upper_g.append(
                    shat_upper_tail(r_hat, n_g, m, delta_bin, eta_star_g, maxiter)
                )
            except ValueError:
                R_upper_g.append(1.0)
        R_upper_g = np.asarray(R_upper_g)

        # Find λ with certified risk ≤ α
        safe_mask_g = R_upper_g <= config.alpha
        if np.any(safe_mask_g):
            lambda_best_g = float(lambda_grid[safe_mask_g].min())
        else:
            lambda_best_g = None

        bin_info.update({
            "n": n_g,
            "R_hat": R_hat_g,
            "R_upper": R_upper_g,
            "lambda_best": lambda_best_g,
        })

    # Adaptive merging loop (from notebook)
    iteration = 0
    while True:
        iteration += 1

        # Compute stats for all bins
        for b in bins:
            compute_bin_stats(b)

        # Check which bins fail
        failing_indices = [i for i, b in enumerate(bins) if b["lambda_best"] is None]

        if len(failing_indices) == 0:
            break

        if len(bins) == 1:
            break

        # Merge failing bins with their right neighbor when possible
        new_bins = []
        i = 0
        while i < len(bins):
            b = bins[i]
            if b["lambda_best"] is None:
                if i < len(bins) - 1:
                    # Merge with right neighbor
                    b_next = bins[i + 1]
                    merged_idx = np.unique(np.concatenate([b["idx"], b_next["idx"]]))
                    new_bins.append({"idx": merged_idx})
                    i += 2
                else:
                    # Last bin and failing: merge with previous in new_bins
                    if len(new_bins) > 0:
                        merged_idx = np.unique(np.concatenate([new_bins[-1]["idx"], b["idx"]]))
                        new_bins[-1] = {"idx": merged_idx}
                    else:
                        # Only one bin total and it fails → keep as is
                        new_bins.append({"idx": b["idx"]})
                    i += 1
            else:
                new_bins.append({"idx": b["idx"]})
                i += 1

        # If we didn't actually reduce the number of bins, stop to avoid infinite loop
        if len(new_bins) >= len(bins):
            break

        bins = new_bins

    # Final stats computation for all bins
    for b in bins:
        compute_bin_stats(b)

    # Build group_stats and group_bands from final bins
    group_stats = []
    group_bands = []
    group_test_bands = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    for g, b in enumerate(bins):
        idx_g = b["idx"]
        lambda_best_g = b.get("lambda_best", None)
        R_hat_g = b.get("R_hat", np.array([]))
        R_upper_g = b.get("R_upper", np.array([]))

        if lambda_best_g is None:
            group_stats.append(
                {
                    "group": g,
                    "group_size": int(len(idx_g)),
                    "lambda_best": None,
                    "empirical_coverage": float("nan"),
                    "certified_coverage": float("nan"),
                    "avg_length": float("nan"),
                }
            )
            continue

        idx_l = int(np.abs(lambda_grid - lambda_best_g).argmin())
        empirical_cov_g = 1 - float(R_hat_g[idx_l])
        certified_cov_g = 1 - float(R_upper_g[idx_l])
        Y_cal_g = np.asarray(Y_cal[idx_g]).ravel()
        y_hat_cal_g = np.asarray(y_hat_cal[idx_g]).ravel()
        lower_g = y_hat_cal_g - lambda_best_g
        upper_g = y_hat_cal_g + lambda_best_g
        avg_len_g = float((upper_g - lower_g).mean())
        in_band_g = (Y_cal_g >= lower_g) & (Y_cal_g <= upper_g)
        X_cal_g = np.asarray(X_cal[idx_g]).ravel()
        x_min, x_max = float(X_cal_g.min()), float(X_cal_g.max())

        group_bands.append(
            {
                "group": g,
                "X": X_cal_g,
                "Y": Y_cal_g,
                "Yhat": y_hat_cal_g,
                "lower": lower_g,
                "upper": upper_g,
                "in_band": in_band_g,
                "lambda": lambda_best_g,
            }
        )
        group_labels_cal[idx_g] = g

        # Assign test points to this group's X range
        X_test_all = np.asarray(X_test).ravel()
        Y_test_all = np.asarray(Y_test).ravel()
        y_hat_test_all = np.asarray(y_hat_test).ravel()
        test_mask = (X_test_all >= x_min) & (X_test_all <= x_max)
        if np.any(test_mask):
            lower_test_g = y_hat_test_all[test_mask] - lambda_best_g
            upper_test_g = y_hat_test_all[test_mask] + lambda_best_g
            in_band_test_g = (Y_test_all[test_mask] >= lower_test_g) & (
                Y_test_all[test_mask] <= upper_test_g
            )

            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test_all[test_mask],
                    "Y": Y_test_all[test_mask],
                    "Yhat": y_hat_test_all[test_mask],
                    "lower": lower_test_g,
                    "upper": upper_test_g,
                    "in_band": in_band_test_g,
                }
            )
            existing = group_labels_test[test_mask]
            replace_mask = existing == -1
            updated = np.where(replace_mask, g, existing)
            group_labels_test[test_mask] = updated

        group_stats.append(
            {
                "group": g,
                "group_size": int(len(idx_g)),
                "lambda_best": lambda_best_g,
                "empirical_coverage": empirical_cov_g,
                "certified_coverage": certified_cov_g,
                "avg_length": avg_len_g,
            }
        )

    return {
        "type": "LTT-certified",
        "alpha": config.alpha,
        "delta": config.delta,
        "lambda_best": lambda_best,
        "coverage_test": coverage,
        "num_safe_lambdas": int(safe_mask.sum()),
        "eta_star": float(eta_star),
        "group_stats": group_stats,
        "coverage_cal": float(coverage_cal),
        "coverage_certified": certified_coverage,
        "lambda_grid": lambda_grid,
        "R_hat": R_hat,
        "R_upper": R_upper,
        "safe_mask": safe_mask,
        "safe_lambdas": lambda_grid[safe_mask],
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "lower_cal": y_hat_cal - lambda_best,
        "upper_cal": y_hat_cal + lambda_best,
        "lower_test": lower,
        "upper_test": upper,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def run_ltt_certified_multi(config: ConditionalConfig) -> Dict[str, object]:
    """
    Multi-dimensional variant of the LTT-certified pipeline that keeps
    calibration/test groupings aligned.
    """
    if config.quantile_models is None:
        raise ValueError("Provide pretrained quantile models (--quantile-low-path/high-path).")
    if config.grouping not in {"tree", "kmeans"}:
        raise ValueError("Multi-dimensional LTT mode supports grouping='tree' or 'kmeans' only.")

    low_model, high_model = config.quantile_models
    predictor = quantile_midpoint_predictor(low_model, high_model)

    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()

    y_hat_cal = predictor(X_cal).ravel()
    y_hat_test = predictor(X_test).ravel()

    abs_err = residuals(Y_cal, y_hat_cal)
    lambda_grid = build_lambda_grid(
        config.lambda_fn,
        abs_err,
        config.alpha,
        config.lambda_params,
    )
    R_hat = empirical_risk_curve(abs_err, lambda_grid)
    R_upper, eta_star = _uniform_upper_bounds(R_hat, len(abs_err), len(lambda_grid), config.alpha, config.delta)

    safe_mask = R_upper <= config.alpha
    if not np.any(safe_mask):
        raise RuntimeError("No λ satisfied the LTT constraint; consider expanding the grid.")
    lambda_best = float(lambda_grid[safe_mask].min())
    idx_best = int(np.abs(lambda_grid - lambda_best).argmin())
    certified_coverage = float(1 - R_upper[idx_best])

    lower_test = y_hat_test - lambda_best
    upper_test = y_hat_test + lambda_best
    coverage_test = coverage_rate(lower_test, upper_test, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_best, y_hat_cal + lambda_best, Y_cal)

    cal_groups, test_groups = _make_groups_multi(config, X_cal, Y_cal, X_test)
    bins = [
        {"cal_idx": cal_idx, "test_idx": test_idx}
        for cal_idx, test_idx in zip(cal_groups, test_groups)
        if len(cal_idx) > 0
    ]

    m = len(lambda_grid)
    maxiter = 100_000

    def compute_bin_stats(bin_info: Dict[str, object]) -> None:
        cal_idx = bin_info["cal_idx"]
        n_g = len(cal_idx)
        if n_g == 0:
            bin_info.update(
                {
                    "lambda_best": None,
                    "R_hat": np.array([]),
                    "R_upper": np.array([]),
                }
            )
            return

        Y_g = Y_cal[cal_idx]
        y_hat_g = y_hat_cal[cal_idx]
        abs_err_g = np.abs(Y_g - y_hat_g)
        R_hat_g = empirical_risk_curve(abs_err_g, lambda_grid)

        non_empty = [b for b in bins if len(b["cal_idx"]) > 0]
        delta_bin = config.delta / max(len(non_empty), 1)
        eta_star_g = get_eta_star_upper(n_g, m, config.alpha, delta_bin, maxiter)

        R_upper_g = []
        for r_hat in R_hat_g:
            try:
                R_upper_g.append(
                    shat_upper_tail(r_hat, n_g, m, delta_bin, eta_star_g, maxiter)
                )
            except ValueError:
                R_upper_g.append(1.0)
        R_upper_g = np.asarray(R_upper_g)
        safe_mask_g = R_upper_g <= config.alpha
        lambda_best_g = float(lambda_grid[safe_mask_g].min()) if np.any(safe_mask_g) else None

        bin_info.update(
            {
                "lambda_best": lambda_best_g,
                "R_hat": R_hat_g,
                "R_upper": R_upper_g,
            }
        )

    while True:
        for b in bins:
            compute_bin_stats(b)
        failing = [i for i, b in enumerate(bins) if b["lambda_best"] is None]
        if not failing or len(bins) == 1:
            break
        new_bins = _merge_bins_multi(bins, failing)
        if len(new_bins) >= len(bins):
            break
        bins = new_bins

    for b in bins:
        compute_bin_stats(b)

    group_stats = []
    group_bands = []
    group_test_bands = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)

    for g, b in enumerate(bins):
        cal_idx = b["cal_idx"]
        test_idx = b["test_idx"]
        lambda_best_g = b.get("lambda_best")
        R_hat_g = b.get("R_hat", np.array([]))
        R_upper_g = b.get("R_upper", np.array([]))

        if lambda_best_g is None:
            group_stats.append(
                {
                    "group": g,
                    "group_size": int(len(cal_idx)),
                    "lambda_best": None,
                    "empirical_coverage": float("nan"),
                    "certified_coverage": float("nan"),
                    "avg_length": float("nan"),
                }
            )
            continue

        idx_lambda = int(np.abs(lambda_grid - lambda_best_g).argmin())
        empirical_cov_g = 1 - float(R_hat_g[idx_lambda]) if len(R_hat_g) else float("nan")
        certified_cov_g = 1 - float(R_upper_g[idx_lambda]) if len(R_upper_g) else float("nan")

        lower_cal = y_hat_cal[cal_idx] - lambda_best_g
        upper_cal = y_hat_cal[cal_idx] + lambda_best_g
        avg_len_g = float((upper_cal - lower_cal).mean())
        in_band_cal = (Y_cal[cal_idx] >= lower_cal) & (Y_cal[cal_idx] <= upper_cal)

        group_bands.append(
            {
                "group": g,
                "X": X_cal[cal_idx],
                "Y": Y_cal[cal_idx],
                "Yhat": y_hat_cal[cal_idx],
                "lower": lower_cal,
                "upper": upper_cal,
                "in_band": in_band_cal,
                "lambda": lambda_best_g,
            }
        )
        group_labels_cal[cal_idx] = g

        if len(test_idx) > 0:
            y_hat_test_g = y_hat_test[test_idx]
            lower_test_g = y_hat_test_g - lambda_best_g
            upper_test_g = y_hat_test_g + lambda_best_g
            in_band_test_g = (Y_test[test_idx] >= lower_test_g) & (Y_test[test_idx] <= upper_test_g)
            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test[test_idx],
                    "Y": Y_test[test_idx],
                    "Yhat": y_hat_test_g,
                    "lower": lower_test_g,
                    "upper": upper_test_g,
                    "in_band": in_band_test_g,
                }
            )
            group_labels_test[test_idx] = g

        group_stats.append(
            {
                "group": g,
                "group_size": int(len(cal_idx)),
                "lambda_best": lambda_best_g,
                "empirical_coverage": empirical_cov_g,
                "certified_coverage": certified_cov_g,
                "avg_length": avg_len_g,
            }
        )

    return {
        "type": "LTT-certified",
        "alpha": config.alpha,
        "delta": config.delta,
        "lambda_best": lambda_best,
        "coverage_test": coverage_test,
        "num_safe_lambdas": int(safe_mask.sum()),
        "eta_star": float(eta_star),
        "group_stats": group_stats,
        "coverage_cal": float(coverage_cal),
        "coverage_certified": certified_coverage,
        "lambda_grid": lambda_grid,
        "R_hat": R_hat,
        "R_upper": R_upper,
        "safe_mask": safe_mask,
        "safe_lambdas": lambda_grid[safe_mask],
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "lower_cal": y_hat_cal - lambda_best,
        "upper_cal": y_hat_cal + lambda_best,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def _merge_failing_bins(bins: List[Dict[str, np.ndarray]], failing: List[int]) -> List[Dict[str, np.ndarray]]:
    new_bins: List[Dict[str, np.ndarray]] = []
    i = 0
    while i < len(bins):
        if i in failing:
            if i < len(bins) - 1:
                merged_idx = np.unique(np.concatenate([bins[i]["idx"], bins[i + 1]["idx"]]))
                new_bins.append({"idx": merged_idx})
                i += 2
            else:
                if new_bins:
                    merged_idx = np.unique(np.concatenate([new_bins[-1]["idx"], bins[i]["idx"]]))
                    new_bins[-1] = {"idx": merged_idx}
                else:
                    new_bins.append({"idx": bins[i]["idx"]})
                i += 1
        else:
            new_bins.append({"idx": bins[i]["idx"]})
            i += 1
    return new_bins


def _merge_bins_multi(
    bins: List[Dict[str, np.ndarray]],
    failing: List[int],
    cal_key: str = "cal_idx",
    test_key: str = "test_idx",
) -> List[Dict[str, np.ndarray]]:
    """
    Merge failing bins for multi-dimensional grouping, keeping calibration and
    test assignments aligned.
    """
    new_bins: List[Dict[str, np.ndarray]] = []
    i = 0
    while i < len(bins):
        if i in failing:
            if i < len(bins) - 1:
                cal_merged = np.unique(np.concatenate([bins[i][cal_key], bins[i + 1][cal_key]]))
                test_merged = np.unique(np.concatenate([bins[i][test_key], bins[i + 1][test_key]]))
                new_bins.append({cal_key: cal_merged, test_key: test_merged})
                i += 2
            else:
                if new_bins:
                    cal_merged = np.unique(np.concatenate([new_bins[-1][cal_key], bins[i][cal_key]]))
                    test_merged = np.unique(np.concatenate([new_bins[-1][test_key], bins[i][test_key]]))
                    new_bins[-1] = {cal_key: cal_merged, test_key: test_merged}
                else:
                    new_bins.append(bins[i])
                i += 1
        else:
            new_bins.append(bins[i])
            i += 1
    return new_bins


def run_single_empirical(config: SingleModelConfig) -> Dict[str, object]:
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()
    y_hat_cal = config.model.predict(X_cal).ravel()
    y_hat_test = config.model.predict(X_test).ravel()

    abs_err = np.abs(Y_cal - y_hat_cal)
    lambda_global = float(np.quantile(abs_err, config.min_coverage))

    groups = _make_groups(config, X_cal, Y_cal.reshape(-1, 1))
    bins = [{"idx": idx} for idx in groups]

    def compute_bin_coverage(bin_info: Dict[str, object]) -> None:
        idx = bin_info["idx"]
        lower = y_hat_cal[idx] - lambda_global
        upper = y_hat_cal[idx] + lambda_global
        bin_info["coverage"] = coverage_rate(lower, upper, Y_cal[idx])

    while True:
        for b in bins:
            compute_bin_coverage(b)
        failing = [
            i
            for i, b in enumerate(bins)
            if b["coverage"] < config.min_coverage - config.coverage_tolerance
        ]
        if not failing or len(bins) == 1:
            break
        bins = _merge_failing_bins(bins, failing)

    group_stats = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    group_bands = []
    group_test_bands = []
    for g, b in enumerate(bins):
        idx = b["idx"]
        lower = y_hat_cal[idx] - lambda_global
        upper = y_hat_cal[idx] + lambda_global
        coverage_g = coverage_rate(lower, upper, Y_cal[idx])
        avg_len_g = float((upper - lower).mean())
        X_g = X_cal[idx].ravel()
        group_stats.append(
            {
                "group": g,
                "group_size": int(len(idx)),
                "coverage": coverage_g,
                "avg_length": avg_len_g,
                "x_min": float(X_g.min()),
                "x_max": float(X_g.max()),
            }
        )
        in_band = (Y_cal[idx] >= lower) & (Y_cal[idx] <= upper)
        group_bands.append(
            {
                "group": g,
                "X": X_g,
                "Y": Y_cal[idx],
                "Yhat": y_hat_cal[idx],
                "lower": lower,
                "upper": upper,
                "in_band": in_band,
            }
        )
        group_labels_cal[idx] = g

        X_test_all = X_test.ravel()
        Y_test_all = Y_test
        mask = (X_test_all >= float(X_g.min())) & (X_test_all <= float(X_g.max()))
        if np.any(mask):
            lower_test = y_hat_test[mask] - lambda_global
            upper_test = y_hat_test[mask] + lambda_global
            in_band_test = (Y_test_all[mask] >= lower_test) & (Y_test_all[mask] <= upper_test)
            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test_all[mask],
                    "Y": Y_test_all[mask],
                    "Yhat": y_hat_test[mask],
                    "lower": lower_test,
                    "upper": upper_test,
                    "in_band": in_band_test,
                }
            )
            unset = group_labels_test[mask] == -1
            group_labels_test[mask] = np.where(unset, g, group_labels_test[mask])

    lower_test = y_hat_test - lambda_global
    upper_test = y_hat_test + lambda_global
    coverage_test = coverage_rate(lower_test, upper_test, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_global, y_hat_cal + lambda_global, Y_cal)

    return {
        "type": "single-empirical",
        "lambda": lambda_global,
        "alpha": config.alpha,
        "min_coverage": config.min_coverage,
        "coverage_tolerance": config.coverage_tolerance,
        "coverage_test": coverage_test,
        "coverage_cal": coverage_cal,
        "group_stats": group_stats,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def run_single_empirical_multi(config: SingleModelConfig) -> Dict[str, object]:
    """
    Multi-dimensional variant of the single-model empirical pipeline.

    Uses tree- or kmeans-based grouping in the full feature space and assigns
    both calibration and test points to groups via the same model, without
    flattening X or relying on 1D ranges.
    """
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()
    y_hat_cal = config.model.predict(X_cal).ravel()
    y_hat_test = config.model.predict(X_test).ravel()

    if config.grouping not in {"tree", "kmeans"}:
        raise ValueError("Multi-dimensional single-model mode supports grouping='tree' or 'kmeans' only.")

    abs_err = np.abs(Y_cal - y_hat_cal)
    lambda_global = float(np.quantile(abs_err, config.min_coverage))

    if config.grouping == "tree":
        cal_groups, test_groups = tree_groups_multi(
            X_cal,
            Y_cal,
            X_test,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            seed=0,
        )
    else:
        cal_groups, test_groups = kmeans_groups_multi(
            X_cal,
            X_test,
            n_clusters=config.n_clusters,
            min_size=config.min_samples_leaf,
            seed=0,
        )

    bins = [
        {"cal_idx": cal_idx, "test_idx": test_idx}
        for cal_idx, test_idx in zip(cal_groups, test_groups)
        if len(cal_idx) > 0
    ]

    def compute_bin_coverage(bin_info: Dict[str, object]) -> None:
        idx = bin_info["cal_idx"]
        if len(idx) == 0:
            bin_info["coverage"] = 1.0
            return
        lower = y_hat_cal[idx] - lambda_global
        upper = y_hat_cal[idx] + lambda_global
        bin_info["coverage"] = coverage_rate(lower, upper, Y_cal[idx])

    # Adaptive merging on calibration coverage, merging both cal/test indices
    while True:
        for b in bins:
            compute_bin_coverage(b)
        failing = [
            i
            for i, b in enumerate(bins)
            if b["coverage"] < config.min_coverage - config.coverage_tolerance
        ]
        if not failing or len(bins) == 1:
            break

        new_bins: List[Dict[str, np.ndarray]] = []
        i = 0
        while i < len(bins):
            if i in failing:
                if i < len(bins) - 1:
                    cal_merged = np.unique(np.concatenate([bins[i]["cal_idx"], bins[i + 1]["cal_idx"]]))
                    test_merged = np.unique(np.concatenate([bins[i]["test_idx"], bins[i + 1]["test_idx"]]))
                    new_bins.append({"cal_idx": cal_merged, "test_idx": test_merged})
                    i += 2
                else:
                    if new_bins:
                        cal_merged = np.unique(np.concatenate([new_bins[-1]["cal_idx"], bins[i]["cal_idx"]]))
                        test_merged = np.unique(np.concatenate([new_bins[-1]["test_idx"], bins[i]["test_idx"]]))
                        new_bins[-1] = {"cal_idx": cal_merged, "test_idx": test_merged}
                    else:
                        new_bins.append(bins[i])
                    i += 1
            else:
                new_bins.append(bins[i])
                i += 1
        if len(new_bins) >= len(bins):
            break
        bins = new_bins

    # Final stats
    group_stats = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    group_bands = []
    group_test_bands = []
    for g, b in enumerate(bins):
        cal_idx = b["cal_idx"]
        test_idx = b["test_idx"]

        lower_cal = y_hat_cal[cal_idx] - lambda_global
        upper_cal = y_hat_cal[cal_idx] + lambda_global
        coverage_g = coverage_rate(lower_cal, upper_cal, Y_cal[cal_idx])
        avg_len_g = float((upper_cal - lower_cal).mean())

        group_stats.append(
            {
                "group": g,
                "group_size": int(len(cal_idx)),
                "coverage": coverage_g,
                "avg_length": avg_len_g,
            }
        )

        in_band_cal = (Y_cal[cal_idx] >= lower_cal) & (Y_cal[cal_idx] <= upper_cal)
        group_bands.append(
            {
                "group": g,
                "X": X_cal[cal_idx],
                "Y": Y_cal[cal_idx],
                "Yhat": y_hat_cal[cal_idx],
                "lower": lower_cal,
                "upper": upper_cal,
                "in_band": in_band_cal,
            }
        )
        group_labels_cal[cal_idx] = g

        if len(test_idx) > 0:
            lower_test_g = y_hat_test[test_idx] - lambda_global
            upper_test_g = y_hat_test[test_idx] + lambda_global
            in_band_test_g = (Y_test[test_idx] >= lower_test_g) & (Y_test[test_idx] <= upper_test_g)
            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test[test_idx],
                    "Y": Y_test[test_idx],
                    "Yhat": y_hat_test[test_idx],
                    "lower": lower_test_g,
                    "upper": upper_test_g,
                    "in_band": in_band_test_g,
                }
            )
            group_labels_test[test_idx] = g

    lower_test = y_hat_test - lambda_global
    upper_test = y_hat_test + lambda_global
    coverage_test = coverage_rate(lower_test, upper_test, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_global, y_hat_cal + lambda_global, Y_cal)

    return {
        "type": "single-empirical",
        "lambda": lambda_global,
        "alpha": config.alpha,
        "min_coverage": config.min_coverage,
        "coverage_tolerance": config.coverage_tolerance,
        "coverage_test": coverage_test,
        "coverage_cal": coverage_cal,
        "group_stats": group_stats,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def run_single_ltt(config: SingleModelConfig) -> Dict[str, object]:
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()
    y_hat_cal = config.model.predict(X_cal).ravel()
    y_hat_test = config.model.predict(X_test).ravel()

    abs_err = np.abs(Y_cal - y_hat_cal)
    lambda_grid = build_lambda_grid(
        config.lambda_fn,
        abs_err,
        config.alpha,
        config.lambda_params,
    )
    R_hat = empirical_risk_curve(abs_err, lambda_grid)
    R_upper, eta_star = _uniform_upper_bounds(R_hat, len(abs_err), len(lambda_grid), config.alpha, config.delta)

    safe_mask = R_upper <= config.alpha
    if not np.any(safe_mask):
        raise RuntimeError("No λ satisfied the LTT constraint; consider expanding the grid.")
    lambda_best = float(lambda_grid[safe_mask].min())
    idx_best = int(np.abs(lambda_grid - lambda_best).argmin())
    certified_coverage = float(1 - R_upper[idx_best])

    lower = y_hat_test - lambda_best
    upper = y_hat_test + lambda_best
    coverage = coverage_rate(lower, upper, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_best, y_hat_cal + lambda_best, Y_cal)

    initial_groups = _make_groups(config, X_cal, Y_cal.reshape(-1, 1))
    bins = [{"idx": idx} for idx in initial_groups if len(idx) > 0]

    m = len(lambda_grid)
    maxiter = 100_000

    def compute_bin_stats(bin_info: Dict[str, object]) -> None:
        idx_g = bin_info["idx"]
        n_g = len(idx_g)
        abs_err_g = np.abs(Y_cal[idx_g] - y_hat_cal[idx_g])
        R_hat_g = empirical_risk_curve(abs_err_g, lambda_grid)
        non_empty_bins = [b for b in bins if len(b["idx"]) > 0]
        delta_bin = config.delta / max(len(non_empty_bins), 1)
        eta_star_g = get_eta_star_upper(n_g, m, config.alpha, delta_bin, maxiter)
        R_upper_g = []
        for r_hat in R_hat_g:
            try:
                R_upper_g.append(shat_upper_tail(r_hat, n_g, m, delta_bin, eta_star_g, maxiter))
            except ValueError:
                R_upper_g.append(1.0)
        R_upper_g = np.asarray(R_upper_g)
        safe_mask_g = R_upper_g <= config.alpha
        lambda_best_g = float(lambda_grid[safe_mask_g].min()) if np.any(safe_mask_g) else None
        bin_info.update({"lambda_best": lambda_best_g, "R_hat": R_hat_g, "R_upper": R_upper_g})

    while True:
        for b in bins:
            compute_bin_stats(b)
        failing = [i for i, b in enumerate(bins) if b["lambda_best"] is None]
        if not failing or len(bins) == 1:
            break
        bins = _merge_failing_bins(bins, failing)

    for b in bins:
        compute_bin_stats(b)

    group_stats = []
    group_bands = []
    group_test_bands = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    for g, b in enumerate(bins):
        idx_g = b["idx"]
        lambda_best_g = b.get("lambda_best")
        R_hat_g = b.get("R_hat", np.array([]))
        R_upper_g = b.get("R_upper", np.array([]))
        if lambda_best_g is None:
            group_stats.append(
                {
                    "group": g,
                    "group_size": int(len(idx_g)),
                    "lambda_best": None,
                    "empirical_coverage": float("nan"),
                    "certified_coverage": float("nan"),
                    "avg_length": float("nan"),
                }
            )
            continue
        idx_l = int(np.abs(lambda_grid - lambda_best_g).argmin())
        empirical_cov_g = 1 - float(R_hat_g[idx_l])
        certified_cov_g = 1 - float(R_upper_g[idx_l])
        lower_g = y_hat_cal[idx_g] - lambda_best_g
        upper_g = y_hat_cal[idx_g] + lambda_best_g
        avg_len_g = float((upper_g - lower_g).mean())
        in_band_g = (Y_cal[idx_g] >= lower_g) & (Y_cal[idx_g] <= upper_g)
        group_bands.append(
            {
                "group": g,
                "X": X_cal[idx_g].ravel(),
                "Y": Y_cal[idx_g],
                "Yhat": y_hat_cal[idx_g],
                "lower": lower_g,
                "upper": upper_g,
                "in_band": in_band_g,
                "lambda": lambda_best_g,
            }
        )
        group_labels_cal[idx_g] = g
        X_test_all = X_test.ravel()
        Y_test_all = Y_test
        mask = (X_test_all >= float(X_cal[idx_g].min())) & (X_test_all <= float(X_cal[idx_g].max()))
        if np.any(mask):
            lower_test_g = y_hat_test[mask] - lambda_best_g
            upper_test_g = y_hat_test[mask] + lambda_best_g
            in_band_test_g = (Y_test_all[mask] >= lower_test_g) & (Y_test_all[mask] <= upper_test_g)
            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test_all[mask],
                    "Y": Y_test_all[mask],
                    "Yhat": y_hat_test[mask],
                    "lower": lower_test_g,
                    "upper": upper_test_g,
                    "in_band": in_band_test_g,
                }
            )
            current = group_labels_test[mask]
            missing = current == -1
            group_labels_test[mask] = np.where(missing, g, current)

        group_stats.append(
            {
                "group": g,
                "group_size": int(len(idx_g)),
                "lambda_best": lambda_best_g,
                "empirical_coverage": empirical_cov_g,
                "certified_coverage": certified_cov_g,
                "avg_length": avg_len_g,
            }
        )

    return {
        "type": "LTT-certified",
        "alpha": config.alpha,
        "delta": config.delta,
        "lambda_best": lambda_best,
        "coverage_test": coverage,
        "num_safe_lambdas": int(safe_mask.sum()),
        "eta_star": float(eta_star),
        "group_stats": group_stats,
        "coverage_cal": float(coverage_cal),
        "coverage_certified": certified_coverage,
        "lambda_grid": lambda_grid,
        "R_hat": R_hat,
        "R_upper": R_upper,
        "safe_mask": safe_mask,
        "safe_lambdas": lambda_grid[safe_mask],
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "lower_cal": y_hat_cal - lambda_best,
        "upper_cal": y_hat_cal + lambda_best,
        "lower_test": lower,
        "upper_test": upper,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def run_single_ltt_multi(config: SingleModelConfig) -> Dict[str, object]:
    """
    Multi-dimensional variant of the single-model LTT-certified pipeline.

    Uses tree- or kmeans-based grouping on full-dimensional X and assigns
    calibration and test points consistently via the same model.
    """
    X_cal = config.X_cal
    Y_cal = config.Y_cal.ravel()
    X_test = config.X_test
    Y_test = config.Y_test.ravel()
    y_hat_cal = config.model.predict(X_cal).ravel()
    y_hat_test = config.model.predict(X_test).ravel()

    if config.grouping not in {"tree", "kmeans"}:
        raise ValueError("Multi-dimensional single-model LTT mode supports grouping='tree' or 'kmeans' only.")

    abs_err = np.abs(Y_cal - y_hat_cal)
    lambda_grid = build_lambda_grid(
        config.lambda_fn,
        abs_err,
        config.alpha,
        config.lambda_params,
    )
    R_hat = empirical_risk_curve(abs_err, lambda_grid)
    R_upper, eta_star = _uniform_upper_bounds(R_hat, len(abs_err), len(lambda_grid), config.alpha, config.delta)

    safe_mask = R_upper <= config.alpha
    if not np.any(safe_mask):
        raise RuntimeError("No λ satisfied the LTT constraint; consider expanding the grid.")
    lambda_best = float(lambda_grid[safe_mask].min())
    idx_best = int(np.abs(lambda_grid - lambda_best).argmin())
    certified_coverage = float(1 - R_upper[idx_best])

    lower = y_hat_test - lambda_best
    upper = y_hat_test + lambda_best
    coverage = coverage_rate(lower, upper, Y_test)
    coverage_cal = coverage_rate(y_hat_cal - lambda_best, y_hat_cal + lambda_best, Y_cal)

    # Grouping in multi-D
    if config.grouping == "tree":
        cal_groups, test_groups = tree_groups_multi(
            X_cal,
            Y_cal,
            X_test,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            seed=0,
        )
    else:
        cal_groups, test_groups = kmeans_groups_multi(
            X_cal,
            X_test,
            n_clusters=config.n_clusters,
            min_size=config.min_samples_leaf,
            seed=0,
        )

    bins = [
        {"cal_idx": cal_idx, "test_idx": test_idx}
        for cal_idx, test_idx in zip(cal_groups, test_groups)
        if len(cal_idx) > 0
    ]

    m = len(lambda_grid)
    maxiter = 100_000

    def compute_bin_stats(bin_info: Dict[str, object]) -> None:
        idx_g = bin_info["cal_idx"]
        n_g = len(idx_g)
        abs_err_g = np.abs(Y_cal[idx_g] - y_hat_cal[idx_g])
        R_hat_g = empirical_risk_curve(abs_err_g, lambda_grid)
        non_empty_bins = [b for b in bins if len(b["cal_idx"]) > 0]
        delta_bin = config.delta / max(len(non_empty_bins), 1)
        eta_star_g = get_eta_star_upper(n_g, m, config.alpha, delta_bin, maxiter)
        R_upper_g = []
        for r_hat in R_hat_g:
            try:
                R_upper_g.append(shat_upper_tail(r_hat, n_g, m, delta_bin, eta_star_g, maxiter))
            except ValueError:
                R_upper_g.append(1.0)
        R_upper_g = np.asarray(R_upper_g)
        safe_mask_g = R_upper_g <= config.alpha
        lambda_best_g = float(lambda_grid[safe_mask_g].min()) if np.any(safe_mask_g) else None
        bin_info.update({"lambda_best": lambda_best_g, "R_hat": R_hat_g, "R_upper": R_upper_g})

    # Adaptive merging loop on calibration bins
    while True:
        for b in bins:
            compute_bin_stats(b)
        failing = [i for i, b in enumerate(bins) if b["lambda_best"] is None]
        if not failing or len(bins) == 1:
            break
        new_bins: List[Dict[str, np.ndarray]] = []
        i = 0
        while i < len(bins):
            if i in failing:
                if i < len(bins) - 1:
                    cal_merged = np.unique(np.concatenate([bins[i]["cal_idx"], bins[i + 1]["cal_idx"]]))
                    test_merged = np.unique(np.concatenate([bins[i]["test_idx"], bins[i + 1]["test_idx"]]))
                    new_bins.append({"cal_idx": cal_merged, "test_idx": test_merged})
                    i += 2
                else:
                    if new_bins:
                        cal_merged = np.unique(np.concatenate([new_bins[-1]["cal_idx"], bins[i]["cal_idx"]]))
                        test_merged = np.unique(np.concatenate([new_bins[-1]["test_idx"], bins[i]["test_idx"]]))
                        new_bins[-1] = {"cal_idx": cal_merged, "test_idx": test_merged}
                    else:
                        new_bins.append(bins[i])
                    i += 1
            else:
                new_bins.append(bins[i])
                i += 1
        if len(new_bins) >= len(bins):
            break
        bins = new_bins

    for b in bins:
        compute_bin_stats(b)

    group_stats = []
    group_bands = []
    group_test_bands = []
    group_labels_cal = np.full(len(Y_cal), -1, dtype=int)
    group_labels_test = np.full(len(Y_test), -1, dtype=int)
    for g, b in enumerate(bins):
        idx_g = b["cal_idx"]
        test_idx = b["test_idx"]
        lambda_best_g = b.get("lambda_best")
        R_hat_g = b.get("R_hat", np.array([]))
        R_upper_g = b.get("R_upper", np.array([]))
        if lambda_best_g is None:
            group_stats.append(
                {
                    "group": g,
                    "group_size": int(len(idx_g)),
                    "lambda_best": None,
                    "empirical_coverage": float("nan"),
                    "certified_coverage": float("nan"),
                    "avg_length": float("nan"),
                }
            )
            continue
        idx_l = int(np.abs(lambda_grid - lambda_best_g).argmin())
        empirical_cov_g = 1 - float(R_hat_g[idx_l])
        certified_cov_g = 1 - float(R_upper_g[idx_l])
        lower_g = y_hat_cal[idx_g] - lambda_best_g
        upper_g = y_hat_cal[idx_g] + lambda_best_g
        avg_len_g = float((upper_g - lower_g).mean())
        in_band_g = (Y_cal[idx_g] >= lower_g) & (Y_cal[idx_g] <= upper_g)
        group_bands.append(
            {
                "group": g,
                "X": X_cal[idx_g],
                "Y": Y_cal[idx_g],
                "Yhat": y_hat_cal[idx_g],
                "lower": lower_g,
                "upper": upper_g,
                "in_band": in_band_g,
                "lambda": lambda_best_g,
            }
        )
        group_labels_cal[idx_g] = g
        if len(test_idx) > 0:
            lower_test_g = y_hat_test[test_idx] - lambda_best_g
            upper_test_g = y_hat_test[test_idx] + lambda_best_g
            in_band_test_g = (Y_test[test_idx] >= lower_test_g) & (Y_test[test_idx] <= upper_test_g)
            group_test_bands.append(
                {
                    "group": g,
                    "X": X_test[test_idx],
                    "Y": Y_test[test_idx],
                    "Yhat": y_hat_test[test_idx],
                    "lower": lower_test_g,
                    "upper": upper_test_g,
                    "in_band": in_band_test_g,
                }
            )
            group_labels_test[test_idx] = g

        group_stats.append(
            {
                "group": g,
                "group_size": int(len(idx_g)),
                "lambda_best": lambda_best_g,
                "empirical_coverage": empirical_cov_g,
                "certified_coverage": certified_cov_g,
                "avg_length": avg_len_g,
            }
        )

    return {
        "type": "LTT-certified",
        "alpha": config.alpha,
        "delta": config.delta,
        "lambda_best": lambda_best,
        "coverage_test": coverage,
        "num_safe_lambdas": int(safe_mask.sum()),
        "eta_star": float(eta_star),
        "group_stats": group_stats,
        "coverage_cal": float(coverage_cal),
        "coverage_certified": certified_coverage,
        "lambda_grid": lambda_grid,
        "R_hat": R_hat,
        "R_upper": R_upper,
        "safe_mask": safe_mask,
        "safe_lambdas": lambda_grid[safe_mask],
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_hat_cal": y_hat_cal,
        "y_hat_test": y_hat_test,
        "lower_cal": y_hat_cal - lambda_best,
        "upper_cal": y_hat_cal + lambda_best,
        "lower_test": lower,
        "upper_test": upper,
        "group_bands": group_bands,
        "group_test_bands": group_test_bands,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }
