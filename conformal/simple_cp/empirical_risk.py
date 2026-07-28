"""
Empirical risk and coverage utilities shared across marginal and conditional
pipelines.
"""

from __future__ import annotations

import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute residuals."""
    return np.abs(y_true.ravel() - y_pred.ravel())


def empirical_risk_curve(abs_errors: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    """
    R_hat(λ) = P(|Y - ŷ| > λ) computed empirically.
    """
    risks = []
    for lam in lambda_grid:
        risks.append(np.mean(abs_errors > lam))
    return np.asarray(risks)


def coverage_rate(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> float:
    """Share of samples whose targets lie inside [lower, upper]."""
    inside = (y_true.ravel() >= lower.ravel()) & (y_true.ravel() <= upper.ravel())
    return float(inside.mean())


def cqr_scores(q_low: np.ndarray, q_high: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Conformity scores used by Conformalized Quantile Regression."""
    scores = np.maximum(q_low - y, y - q_high)
    return np.maximum(scores, 0.0)

