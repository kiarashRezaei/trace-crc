"""
Ways to propose λ values (tolerances / half-widths) given absolute residuals.
"""

from __future__ import annotations

from inspect import signature
from typing import Callable, Dict, Iterable

import numpy as np


LambdaGridFn = Callable[..., np.ndarray]


def grid_from_errors(
    abs_errors: np.ndarray,
    quantile: float = 0.95,
    padding: float = 1.3,
    n_points: int = 60,
) -> np.ndarray:
    """
    Match the notebook recipe: build a grid up to q * padding, where q is a high
    quantile of the errors.
    """
    q = np.quantile(abs_errors, quantile)
    upper = float(q * padding)
    return np.linspace(0.0, upper, n_points)


def fixed_lambda(abs_errors: np.ndarray, alpha: float) -> float:
    """
    Classic conformal choice: (1-alpha)-quantile of calibration residuals (single lambda value).
    """
    n = len(abs_errors)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.partition(abs_errors, rank - 1)[rank - 1])


def adaptive_grid(
    abs_errors: np.ndarray,
    coarse_points: int = 20,
    fine_points: int = 40,
) -> np.ndarray:
    """
    Start with a coarse linspace to find the maximum residual, then refine near
    that region. Useful when errors are heavy-tailed.
    """
    max_err = float(abs_errors.max())
    coarse = np.linspace(0.0, max_err, coarse_points, endpoint=True)
    fine_start = 0.5 * max_err
    fine = np.linspace(fine_start, 1.2 * max_err, fine_points, endpoint=True)
    return np.unique(np.concatenate([coarse, fine]))


LAMBDA_GRID_FUNCTIONS: Dict[str, LambdaGridFn] = {
    "grid_from_errors": grid_from_errors,
    "adaptive_grid": adaptive_grid,
}


def available_lambda_functions() -> Iterable[str]:
    """Names exposed to the CLI/config when picking λ strategies."""
    return sorted(LAMBDA_GRID_FUNCTIONS.keys())


def build_lambda_grid(
    name: str,
    abs_errors: np.ndarray,
    alpha: float,
    params: Dict[str, object] | None = None,
) -> np.ndarray:
    """
    Evaluate a registered λ-grid function with optional parameters.

    If the target function accepts an `alpha` argument and the caller did not
    explicitly override it via `params`, we automatically pass the pipeline's
    alpha for convenience.
    """
    if name not in LAMBDA_GRID_FUNCTIONS:
        raise ValueError(
            f"Unknown lambda function '{name}'. Available: {', '.join(available_lambda_functions())}"
        )

    fn = LAMBDA_GRID_FUNCTIONS[name]
    fn_params = dict(params or {})
    sig = signature(fn)
    if "alpha" in sig.parameters and "alpha" not in fn_params:
        fn_params["alpha"] = alpha

    return fn(abs_errors=abs_errors, **fn_params)

