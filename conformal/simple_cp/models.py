"""
Lightweight regression models used by the simple conformal pipelines.

The goal is to expose a handful of functions that return trained predictors
without introducing custom classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression


Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass
class ModelBundle:
    """
    Container for the trained estimator plus an optional callable that maps
    inputs to predictions. Keeping it dataclass-based makes the returned object
    easy to inspect without adding behavior.
    """

    name: str
    model: object
    predict: Predictor


def fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Optional[Callable[[], Any]] = None,
    model_name: Optional[str] = None,
) -> ModelBundle:
    """
    Fit a user-specified regressor (defaults to Ordinary Least Squares).

    Provide `model_factory` to return any sklearn-style estimator exposing
    `fit(X, y)` and `predict(X_new)` if you want something other than OLS.
    """
    if model_factory is None:
        from sklearn.linear_model import LinearRegression

        model_factory = LinearRegression

    reg = model_factory()
    reg.fit(X, y.ravel())

    if model_name is None:
        model_name = getattr(reg, "__class__", type("obj", (), {})).__name__

    return ModelBundle(model_name, reg, lambda X_new: reg.predict(X_new))


def fit_quantile_regressors(
    X: np.ndarray,
    y: np.ndarray,
    alpha_target: float = 0.1,
    n_estimators: int = 300,
    max_depth: int = 3,
    random_state: int = 0,
    low_model_factory: Optional[Callable[[float], Any]] = None,
    high_model_factory: Optional[Callable[[float], Any]] = None,
) -> Tuple[ModelBundle, ModelBundle]:
    """
    Train lower/upper quantile regressors (defaults to GradientBoostingRegressor).

    Pass `low_model_factory` / `high_model_factory` if you want to supply your
    own estimators; each factory receives the desired quantile level and must
    return an object exposing `fit(X, y)` and `predict(X_new)`.
    """
    q_low = alpha_target / 2
    q_high = 1 - alpha_target / 2

    if low_model_factory is None:
        from sklearn.ensemble import GradientBoostingRegressor

        def low_model_factory(alpha: float) -> Any:
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )

    if high_model_factory is None:
        from sklearn.ensemble import GradientBoostingRegressor

        def high_model_factory(alpha: float) -> Any:
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )

    model_low = low_model_factory(q_low)
    model_high = high_model_factory(q_high)

    model_low.fit(X, y.ravel())
    model_high.fit(X, y.ravel())

    return (
        ModelBundle("quantile_low", model_low, lambda X_new: model_low.predict(X_new)),
        ModelBundle("quantile_high", model_high, lambda X_new: model_high.predict(X_new)),
    )


def quantile_midpoint_predictor(low_model: ModelBundle, high_model: ModelBundle) -> Predictor:
    """
    Average the lower/upper quantile predictions to get a smooth center line.
    """
    def predict(X: np.ndarray) -> np.ndarray:
        return 0.5 * (low_model.predict(X) + high_model.predict(X))

    return predict
