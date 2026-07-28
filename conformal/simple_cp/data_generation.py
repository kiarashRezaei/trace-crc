"""
Simple synthetic data generators used by the learning-to-test (LTT) examples.

Each function returns NumPy arrays so statisticians can plug them directly into
calibration code without worrying about custom data structures.
"""

from __future__ import annotations

import numpy as np


def linear_sine(
    n_samples: int = 5000,
    x_range: tuple[float, float] = (-3.0, 3.0),
    noise_std: float = 0.5,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate y = 2x + sin(3x) + Normal(0, noise_std^2) data.

    Returns
    -------
    X : (n_samples, 1) array
    Y : (n_samples, 1) array
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    noise = rng.normal(0.0, noise_std, size=(n_samples, 1))
    Y = 2 * X + np.sin(3 * X) + noise
    return X, Y


def piecewise_linear(
    n_samples: int = 4000,
    breakpoints: tuple[float, float] = (-1.0, 1.0),
    slopes: tuple[float, float, float] = (-1.5, 0.2, 1.7),
    noise_std: float = 0.4,
    seed: int | None = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-linear signal with three regions and Gaussian noise.
    Useful for stress-testing localized coverage guarantees.
    """
    if len(slopes) != 3:
        raise ValueError("Provide exactly three slopes for the left/mid/right segments.")

    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n_samples, 1))
    b1, b2 = breakpoints
    signal = np.empty_like(X, dtype=float)

    left_mask = X < b1
    mid_mask = (X >= b1) & (X <= b2)
    right_mask = X > b2

    signal[left_mask] = slopes[0] * (X[left_mask] - b1)
    signal[mid_mask] = slopes[1] * (X[mid_mask] - b1)
    signal[right_mask] = slopes[2] * (X[right_mask] - b2)

    noise = rng.normal(0.0, noise_std, size=X.shape)
    Y = signal + noise
    return X, Y


def split_train_cal_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_train: int = 2000,
    n_cal: int = 2000,
) -> dict[str, np.ndarray]:
    """
    Deterministic split matching the notebook setup: train / calibration / test.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    if n_train + n_cal >= X.shape[0]:
        raise ValueError("Need leftover samples for the test split.")

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_cal, Y_cal = X[n_train : n_train + n_cal], Y[n_train : n_train + n_cal]
    X_test, Y_test = X[n_train + n_cal :], Y[n_train + n_cal :]

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
    }


DATA_GENERATORS = {
    "linear_sine": linear_sine,
    "piecewise_linear": piecewise_linear,
}


def available_data_generators() -> list[str]:
    return sorted(DATA_GENERATORS.keys())

