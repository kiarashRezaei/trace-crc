"""
Plain marginal conformal prediction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..empirical_risk import empirical_risk_curve, residuals, coverage_rate
from ..lambda_functions import build_lambda_grid, fixed_lambda
from ...core.uniform_concentration import get_eta_star_upper, shat_upper_tail
from ..models import ModelBundle


@dataclass
class MarginalConfig:
    X_cal: np.ndarray
    Y_cal: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    alpha: float = 0.1
    model_bundle: ModelBundle | None = None
    delta: float = 0.05
    lambda_fn: str = "grid_from_errors"
    lambda_params: Dict[str, object] = field(default_factory=dict)


def run_marginal(config: MarginalConfig) -> Dict[str, float]:
    """
    Standard split conformal approach using provided calibration/test splits.
    """
    if config.model_bundle is None:
        raise ValueError("Marginal conformal pipeline requires a pretrained regressor (--regressor-path).")
    model = config.model_bundle
    X_cal, Y_cal = config.X_cal, config.Y_cal
    X_test, Y_test = config.X_test, config.Y_test
    y_hat_cal = model.predict(X_cal)
    abs_err = residuals(Y_cal, y_hat_cal)

    lam = fixed_lambda(abs_err, config.alpha)

    def interval(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return preds - lam, preds + lam

    y_hat_test = model.predict(X_test)
    lower_test, upper_test = interval(y_hat_test)
    lower_cal, upper_cal = interval(y_hat_cal)

    coverage_test = coverage_rate(lower_test, upper_test, Y_test)
    coverage_cal = coverage_rate(lower_cal, upper_cal, Y_cal)

    group_labels_cal = np.full(len(Y_cal.ravel()), "global", dtype=object)
    group_labels_test = np.full(len(Y_test.ravel()), "global", dtype=object)

    return {
        "lambda": float(lam),
        "coverage_test": float(coverage_test),
        "coverage_cal": float(coverage_cal),
        "avg_length": float((upper_test - lower_test).mean()),
        "alpha": config.alpha,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_hat_cal": y_hat_cal,
        "Y_hat_test": y_hat_test,
        "lower_cal": lower_cal,
        "upper_cal": upper_cal,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }


def _uniform_upper_bounds(R_hat: np.ndarray, n_cal: int, m: int, alpha: float, delta: float) -> tuple[np.ndarray, float]:
    eta_star = get_eta_star_upper(n_cal, m, alpha, delta, 100_000)
    bounds = []
    for r_hat in R_hat:
        try:
            bounds.append(shat_upper_tail(r_hat, n_cal, m, delta, eta_star, 100_000))
        except ValueError:
            bounds.append(1.0)
    return np.asarray(bounds), eta_star


def run_marginal_ltt(config: MarginalConfig) -> Dict[str, object]:
    if config.model_bundle is None:
        raise ValueError("Marginal LTT pipeline requires a pretrained regressor (--regressor-path).")

    model = config.model_bundle
    X_cal, Y_cal = config.X_cal, config.Y_cal.ravel()
    X_test, Y_test = config.X_test, config.Y_test.ravel()

    y_hat_cal = model.predict(X_cal).ravel()
    y_hat_test = model.predict(X_test).ravel()
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
    lower_cal = y_hat_cal - lambda_best
    upper_cal = y_hat_cal + lambda_best

    coverage_test = coverage_rate(lower_test, upper_test, Y_test)
    coverage_cal = coverage_rate(lower_cal, upper_cal, Y_cal)

    group_labels_cal = np.full(len(Y_cal), "global", dtype=object)
    group_labels_test = np.full(len(Y_test), "global", dtype=object)

    return {
        "type": "LTT-certified",
        "alpha": config.alpha,
        "delta": config.delta,
        "lambda_best": lambda_best,
        "coverage_test": float(coverage_test),
        "coverage_cal": float(coverage_cal),
        "coverage_certified": certified_coverage,
        "num_safe_lambdas": int(safe_mask.sum()),
        "eta_star": float(eta_star),
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
        "lower_cal": lower_cal,
        "upper_cal": upper_cal,
        "lower_test": lower_test,
        "upper_test": upper_test,
        "group_stats": [],
        "group_bands": [],
        "group_test_bands": [],
        "group_labels_cal": group_labels_cal,
        "group_labels_test": group_labels_test,
    }
