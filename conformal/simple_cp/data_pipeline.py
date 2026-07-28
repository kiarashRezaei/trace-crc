"""
Data pipeline for generating or loading datasets, splitting, and saving splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from .data_generation import DATA_GENERATORS, available_data_generators


SplitResult = Dict[str, Dict[str, np.ndarray]]


def _sequential_split(
    X: np.ndarray,
    Y: np.ndarray,
    n_train: int,
    n_val: int,
    n_cal: int,
    seed: int,
    order: np.ndarray | None = None,
) -> SplitResult:
    total = len(X)
    n_test = total - (n_train + n_val + n_cal)
    if n_test <= 0:
        raise ValueError("Need leftover samples for the test split.")

    if order is None:
        order = np.arange(total)
    splits = {}
    cursor = 0
    for name, n in (("train", n_train), ("val", n_val), ("cal", n_cal), ("test", n_test)):
        if n <= 0:
            splits[name] = {"X": np.empty((0, X.shape[1])), "Y": np.empty((0, Y.shape[1])), "indices": np.array([], dtype=int)}
            continue
        idx = order[cursor : cursor + n]
        splits[name] = {"X": X[cursor : cursor + n], "Y": Y[cursor : cursor + n], "indices": idx}
        cursor += n
    return splits


def _random_split(
    X: np.ndarray,
    Y: np.ndarray,
    n_train: int,
    n_val: int,
    n_cal: int,
    seed: int,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_perm = X[perm]
    Y_perm = Y[perm]
    return _sequential_split(X_perm, Y_perm, n_train, n_val, n_cal, seed, order=perm)


SPLIT_FUNCTIONS: Dict[str, Callable[..., SplitResult]] = {
    "sequential": _sequential_split,
    "random": _random_split,
}


def available_split_strategies() -> list[str]:
    return sorted(SPLIT_FUNCTIONS.keys())


@dataclass
class DataPipelineConfig:
    output_dir: Path
    n_samples: int
    n_train: int
    n_val: int
    n_cal: int
    seed: int = 0
    data_fn: str | None = "linear_sine"
    data_params: Dict[str, object] = field(default_factory=dict)
    data_npz: str | None = None
    split_fn: str = "sequential"


def run_data_pipeline(config: DataPipelineConfig) -> Dict[str, object]:
    """
    Generate or load a dataset, split it, and save per-split NPZ files.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, source = _load_or_generate(config)
    splitter = SPLIT_FUNCTIONS.get(config.split_fn)
    if splitter is None:
        raise ValueError(f"Unknown split function '{config.split_fn}'. Available: {', '.join(available_split_strategies())}")

    splits = splitter(X, Y, config.n_train, config.n_val, config.n_cal, config.seed)
    split_paths = {}
    for name, payload in splits.items():
        if payload["X"].size == 0:
            continue
        path = config.output_dir / f"{name}.npz"
        np.savez(path, X=payload["X"], Y=payload["Y"], indices=payload["indices"])
        split_paths[name] = str(path)

    summary = {
        "generator": source,
        "data_params": config.data_params,
        "data_npz": config.data_npz,
        "n_samples": len(X),
        "n_train": config.n_train,
        "n_val": config.n_val,
        "n_cal": config.n_cal,
        "n_test": len(X) - (config.n_train + config.n_val + config.n_cal),
        "seed": config.seed,
        "split_fn": config.split_fn,
        "split_paths": split_paths,
    }
    summary_path = config.output_dir / "data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def _load_or_generate(config: DataPipelineConfig) -> Tuple[np.ndarray, np.ndarray, str]:
    if config.data_npz:
        data = np.load(config.data_npz)
        if "X" not in data or "Y" not in data:
            raise ValueError(f"{config.data_npz} must contain arrays 'X' and 'Y'.")
        X = np.asarray(data["X"])
        Y = np.asarray(data["Y"])
        return X, Y, Path(config.data_npz).name

    if config.data_fn is None:
        raise ValueError("Either --data-fn or --data-npz must be provided for the data pipeline.")

    generator = DATA_GENERATORS.get(config.data_fn)
    if generator is None:
        raise ValueError(f"Unknown data generator '{config.data_fn}'. Available: {', '.join(available_data_generators())}")

    params = dict(config.data_params)
    if "seed" not in params:
        params["seed"] = config.seed
    if "n_samples" not in params:
        params["n_samples"] = config.n_samples

    X, Y = generator(**params)
    return X, Y, config.data_fn

