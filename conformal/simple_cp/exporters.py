"""
Utilities for persisting conformal results to CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _flatten(array: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(array)
    return arr.ravel()


def save_band_csv(
    path: Path,
    model_name: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    group_labels: Optional[np.ndarray] = None,
) -> None:
    """
    Save per-sample prediction/band information so plots can be recreated later.
    Columns: model_name, upper_band, lower_band, prediction, true_value, group.
    """
    y_true_flat = _flatten(y_true)
    pred_flat = _flatten(prediction)
    lower_flat = _flatten(lower)
    upper_flat = _flatten(upper)

    if not (len(y_true_flat) == len(pred_flat) == len(lower_flat) == len(upper_flat)):
        raise ValueError("Prediction/band arrays must share the same length.")

    if group_labels is None:
        groups = np.full(len(y_true_flat), "global", dtype=object)
    else:
        groups = np.asarray(group_labels)
        if groups.ndim > 1:
            groups = groups.ravel()
        if len(groups) != len(y_true_flat):
            raise ValueError("group_labels must match the number of samples.")
        if np.issubdtype(groups.dtype, np.number):
            groups = groups.astype(object)
            groups[groups < 0] = "global"
        groups = groups.astype(str)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model_name", "upper_band", "lower_band", "prediction", "true_value", "group"])
        for ub, lb, pred, truth, grp in zip(upper_flat, lower_flat, pred_flat, y_true_flat, groups):
            writer.writerow([model_name, ub, lb, pred, truth, grp])


def save_mu_band_csv(
    path: Path,
    model_name: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray,
) -> None:
    """
    Save mu-band (model uncertainty band) predictions.
    Columns: model_name, upper_band, lower_band, prediction, true_value.
    """
    y_true_flat = _flatten(y_true)
    pred_flat = _flatten(prediction)
    upper_flat = _flatten(upper_band)
    lower_flat = _flatten(lower_band)

    if not (len(y_true_flat) == len(pred_flat) == len(upper_flat) == len(lower_flat)):
        raise ValueError("All arrays must share the same length.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model_name", "upper_band", "lower_band", "prediction", "true_value"])
        for ub, lb, pred, truth in zip(upper_flat, lower_flat, pred_flat, y_true_flat):
            writer.writerow([model_name, ub, lb, pred, truth])


def save_report_csv(metrics: Dict[str, object], path: Path) -> None:
    """
    Persist scalar summary metrics (coverage, λ, etc.) as a single-row CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(metrics.keys())
    values = [metrics[k] for k in headers]
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerow(values)


def save_summary_csv(
    path: Path,
    model_name: str,
    alpha: float,
    delta: Optional[float],
    group_stats: list,
    group_test_stats: Optional[list] = None,
    is_empirical: bool = False,
    global_empirical_coverage_cal: Optional[float] = None,
    global_empirical_coverage_test: Optional[float] = None,
    global_ltt_coverage_cal: Optional[float] = None,
    global_ltt_coverage_test: Optional[float] = None,
    global_lambda: Optional[float] = None,
    global_num_safe_lambdas: Optional[int] = None,
    global_avg_length_cal: Optional[float] = None,
    global_avg_length_test: Optional[float] = None,
    min_coverage: Optional[float] = None,
    coverage_tolerance: Optional[float] = None,
) -> None:
    """
    Save summary CSV with standardized format:
    dataset, model_name, alpha, delta, group, empirical_coverage, ltt_coverage,
    num_safe_lambda, best_lambda, average_interval_length

    For marginal pipelines, group_stats should be empty (single global group).
    For conditional pipelines, group_stats contains per-group calibration information.
    group_test_stats contains per-group test information (optional).
    
    Generates separate rows for "calib" and "test" datasets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "dataset",
        "model_name",
        "alpha",
        "delta",
        "min_coverage",
        "coverage_tolerance",
        "group",
        "empirical_coverage",
        "ltt_coverage",
        "num_safe_lambda",
        "best_lambda",
        "average_interval_length",
    ]
    min_cov_val = float(min_coverage) if min_coverage is not None else float("nan")
    cov_tol_val = float(coverage_tolerance) if coverage_tolerance is not None else float("nan")

    
    rows = []
    
    # Marginal pipeline: single global group, two rows (calib and test)
    if not group_stats:
        # Calibration row
        empirical_cov_cal = global_empirical_coverage_cal if global_empirical_coverage_cal is not None else float("nan")
        ltt_cov_cal = global_ltt_coverage_cal if global_ltt_coverage_cal is not None else float("nan")
        num_safe = float(global_num_safe_lambdas) if global_num_safe_lambdas is not None else float("nan")
        best_lam = global_lambda if global_lambda is not None else float("nan")
        avg_len_cal = global_avg_length_cal if global_avg_length_cal is not None else float("nan")
        
        rows.append([
            "calib",
            model_name,
            alpha,
            delta if delta is not None else float("nan"),
            min_cov_val,
            cov_tol_val,
            0,  # Single group for marginal
            float(empirical_cov_cal),
            float(ltt_cov_cal),
            float(num_safe),
            float(best_lam),
            float(avg_len_cal),
        ])
        
        # Test row
        empirical_cov_test = global_empirical_coverage_test if global_empirical_coverage_test is not None else float("nan")
        ltt_cov_test = global_ltt_coverage_test if global_ltt_coverage_test is not None else float("nan")
        avg_len_test = global_avg_length_test if global_avg_length_test is not None else float("nan")
        
        rows.append([
            "test",
            model_name,
            alpha,
            delta if delta is not None else float("nan"),
            min_cov_val,
            cov_tol_val,
            0,  # Single group for marginal
            float(empirical_cov_test),
            float(ltt_cov_test),
            float(num_safe),
            float(best_lam),
            float(avg_len_test),
        ])
    else:
        # Conditional pipeline: two rows per group (calib and test)
        # Create a mapping from group_id to test stats for quick lookup
        test_stats_map = {}
        if group_test_stats:
            for test_stat in group_test_stats:
                group_id = test_stat.get("group") or test_stat.get("bin", 0)
                test_stats_map[group_id] = test_stat
        
        for g_stat in group_stats:
            # Handle both "group" and "bin" keys (CQR uses "bin")
            group_id = g_stat.get("group") or g_stat.get("bin", 0)
            
            # ===== CALIBRATION ROW =====
            # Empirical coverage
            empirical_cov_cal = g_stat.get("coverage") or g_stat.get("empirical_coverage")
            if empirical_cov_cal is None:
                empirical_cov_cal = float("nan")
            
            # LTT coverage (only for LTT pipelines)
            ltt_cov_cal = g_stat.get("certified_coverage")
            if ltt_cov_cal is None or is_empirical:
                ltt_cov_cal = float("nan")
            
            # Best lambda
            best_lam = g_stat.get("lambda_best") or g_stat.get("lambda")
            if best_lam is None:
                best_lam = float("nan")
            
            # Average interval length (handle both "avg_length" and "avg_width")
            avg_len_cal = g_stat.get("avg_length") or g_stat.get("avg_width")
            if avg_len_cal is None:
                avg_len_cal = float("nan")
            
            # Num safe lambdas (only for LTT, use global value for all groups)
            num_safe = g_stat.get("num_safe_lambdas")
            if num_safe is None:
                # Fall back to global value if provided
                num_safe = global_num_safe_lambdas
            if num_safe is None or is_empirical:
                num_safe = float("nan")
            
            rows.append([
                "calib",
                model_name,
                alpha,
                delta if delta is not None else float("nan"),
                min_cov_val,
                cov_tol_val,
                group_id,
                float(empirical_cov_cal),
                float(ltt_cov_cal),
                float(num_safe),
                float(best_lam),
                float(avg_len_cal),
            ])
            
            # ===== TEST ROW =====
            test_stat = test_stats_map.get(group_id)
            if test_stat:
                empirical_cov_test = test_stat.get("coverage") or test_stat.get("empirical_coverage")
                if empirical_cov_test is None:
                    empirical_cov_test = float("nan")
                
                # LTT coverage for test (same as calib for LTT, or NaN for empirical)
                ltt_cov_test = test_stat.get("certified_coverage")
                if ltt_cov_test is None or is_empirical:
                    ltt_cov_test = float("nan")
                
                avg_len_test = test_stat.get("avg_length") or test_stat.get("avg_width")
                if avg_len_test is None:
                    avg_len_test = float("nan")
            else:
                # No test stats for this group
                empirical_cov_test = float("nan")
                ltt_cov_test = float("nan")
                avg_len_test = float("nan")
            
            rows.append([
                "test",
                model_name,
                alpha,
                delta if delta is not None else float("nan"),
                min_cov_val,
                cov_tol_val,
                group_id,
                float(empirical_cov_test),
                float(ltt_cov_test),
                float(num_safe),
                float(best_lam),
                float(avg_len_test),
            ])
    
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

