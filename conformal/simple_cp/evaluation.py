"""
Model evaluation pipeline using saved splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .io_utils import load_split, ensure_dir, load_model_bundle

EvalTarget = Literal["regressor", "quantile"]


@dataclass
class EvaluationConfig:
    target: EvalTarget
    test_data: str
    regressor_path: Optional[str] = None
    quantile_low_path: Optional[str] = None
    quantile_high_path: Optional[str] = None
    output_dir: Path | None = None


def run_evaluation(config: EvaluationConfig) -> Dict[str, object]:
    if config.output_dir is None:
        raise ValueError("Evaluation requires an output directory.")
    ensure_dir(config.output_dir)
    evaluation_dir = ensure_dir(config.output_dir / "evaluation")

    X_test, Y_test = load_split(config.test_data)
    metrics: Dict[str, float] = {}

    if config.target == "regressor":
        if not config.regressor_path:
            raise ValueError("Provide --regressor-path for regressor evaluation.")
        model_bundle = load_model_bundle(config.regressor_path, "regressor")
        preds = model_bundle.predict(X_test)
    else:
        if not (config.quantile_low_path and config.quantile_high_path):
            raise ValueError("Provide both --quantile-low-path and --quantile-high-path for quantile evaluation.")
        low_bundle = load_model_bundle(config.quantile_low_path, "quantile_low")
        high_bundle = load_model_bundle(config.quantile_high_path, "quantile_high")
        # Use ensemble mean predictions if ensembles are detected
        preds = 0.5 * (low_bundle.predict(X_test) + high_bundle.predict(X_test))

    mse = mean_squared_error(Y_test, preds)
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)
    metrics["mse"] = float(mse)
    metrics["rmse"] = float(mse**0.5)
    metrics["mae"] = float(mae)
    metrics["r2"] = float(r2)

    summary = {
        "target": config.target,
        "test_data": config.test_data,
        "metrics": metrics,
        "output_dir": str(config.output_dir),
    }

    _plot_predictions(evaluation_dir / "pred_vs_true.png", Y_test, preds)

    summary_path = config.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    summary["plot_path"] = str(evaluation_dir / "pred_vs_true.png")
    return summary


def _plot_predictions(path: Path, y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="ideal")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs True")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

