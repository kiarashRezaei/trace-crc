"""
Plotting utilities mirroring the notebook figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig: plt.Figure, output_file: Optional[Path], show: bool) -> None:
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")
    if show:
        fig.show()
    plt.close(fig)


def _select_plot_indices(n: int, plot_options: Optional[Dict[str, object]]) -> np.ndarray:
    """
    Determine which sorted indices to display based on the requested interval
    and/or downsampling settings. Returns a numpy array of integer positions in
    the sorted order.
    """
    if n == 0:
        return np.array([], dtype=int)

    start = 0
    end = n
    max_points: Optional[int] = None

    if plot_options:
        index_range = plot_options.get("index_range")
        if index_range:
            start = max(0, min(n - 1, int(index_range[0])))
            end = min(n, max(start + 1, int(index_range[1])))
        max_points_val = plot_options.get("max_points")
        if isinstance(max_points_val, int):
            max_points = max_points_val
    if start >= end:
        raise ValueError("Plot index range filters out all available samples.")

    indices = np.arange(start, end, dtype=int)
    if max_points and max_points < len(indices):
        selection = np.linspace(0, len(indices) - 1, max_points, dtype=int)
        indices = indices[selection]
    return indices


def plot_marginal_results(result: dict, output_dir: Optional[Path] = None, show: bool = False) -> None:
    X_test = np.asarray(result["X_test"]).ravel()
    Y_test = np.asarray(result["Y_test"]).ravel()
    lower = np.asarray(result["lower_test"]).ravel()
    upper = np.asarray(result["upper_test"]).ravel()
    y_hat = np.asarray(result["Y_hat_test"]).ravel()

    order = np.argsort(X_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_test[order], y_hat[order], label="prediction ŷ(x)", linewidth=2)
    ax.fill_between(X_test[order], lower[order], upper[order], alpha=0.2, label="conformal band")
    ax.scatter(
        X_test,
        Y_test,
        s=12,
        c=np.where((Y_test >= lower) & (Y_test <= upper), "tab:blue", "tab:red"),
        label="test samples",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Marginal conformal prediction band (test set)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "marginal_test_band.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Calibration set figure
    X_cal = np.asarray(result["X_cal"]).ravel()
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    lower_cal = np.asarray(result["lower_cal"]).ravel()
    upper_cal = np.asarray(result["upper_cal"]).ravel()
    y_hat_cal = np.asarray(result["Y_hat_cal"]).ravel()
    order_cal = np.argsort(X_cal)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_cal[order_cal], y_hat_cal[order_cal], label="prediction ŷ(x)", linewidth=2)
    ax.fill_between(
        X_cal[order_cal],
        lower_cal[order_cal],
        upper_cal[order_cal],
        alpha=0.2,
        label="conformal band",
    )
    in_band_cal = (Y_cal >= lower_cal) & (Y_cal <= upper_cal)
    ax.scatter(
        X_cal[in_band_cal],
        Y_cal[in_band_cal],
        s=12,
        label="covered calibration samples",
    )
    ax.scatter(
        X_cal[~in_band_cal],
        Y_cal[~in_band_cal],
        s=12,
        marker="x",
        label="not covered (calibration)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Marginal conformal prediction band (calibration set)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "marginal_calibration_band.png" if output_dir else None
    _finalize(fig, out_file, show)


def plot_cqr_results(result: dict, output_dir: Optional[Path] = None, show: bool = False) -> None:
    X_test = np.asarray(result["X_test"]).ravel()
    Y_test = np.asarray(result["Y_test"]).ravel()
    lower = np.asarray(result["lower_test"]).ravel()
    upper = np.asarray(result["upper_test"]).ravel()
    y_mid = 0.5 * (np.asarray(result["q_low_test"]) + np.asarray(result["q_high_test"]))

    order = np.argsort(X_test)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_test[order], y_mid[order], label="CQR center (mid-quantile)", linewidth=2)
    in_band = (Y_test >= lower) & (Y_test <= upper)
    bin_stats = result.get("bin_stats") or []
    bin_source = result.get("bin_stats_source", "quantiles")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(bin_stats), 1)))
    if bin_stats:
        for stats, color in zip(bin_stats, colors):
            mask = (X_test >= stats["x_min"]) & (X_test <= stats["x_max"])
            if not np.any(mask):
                continue
            order_mask = np.argsort(X_test[mask])
            label_id = stats.get("group") if bin_source == "groups" else stats.get("bin")
            label_prefix = "group" if bin_source == "groups" else "bin"
            ax.fill_between(
                X_test[mask][order_mask],
                lower[mask][order_mask],
                upper[mask][order_mask],
                alpha=0.25,
                color=color,
                label=f"{label_prefix} {label_id} interval",
            )
            ax.scatter(
                X_test[mask & in_band],
                Y_test[mask & in_band],
                s=15,
                color=color,
                marker="o",
            )
            ax.scatter(
                X_test[mask & ~in_band],
                Y_test[mask & ~in_band],
                s=25,
                color=color,
                marker="x",
            )
    else:
        ax.fill_between(
            X_test[order],
            lower[order],
            upper[order],
            alpha=0.25,
            label=f"CQR interval (target {(1 - result['alpha'])*100:.0f}% coverage)",
        )
        ax.scatter(X_test[in_band], Y_test[in_band], s=15, label="covered test points")
        ax.scatter(X_test[~in_band], Y_test[~in_band], s=25, marker="x", label="not covered")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Conformalized Quantile Regression (test set)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "cqr_test_intervals.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Calibration set plot
    X_cal = np.asarray(result["X_cal"]).ravel()
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    lower_cal = np.asarray(result["lower_cal"]).ravel()
    upper_cal = np.asarray(result["upper_cal"]).ravel()
    y_mid_cal = 0.5 * (np.asarray(result["q_low_cal"]) + np.asarray(result["q_high_cal"]))
    order_cal = np.argsort(X_cal)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_cal[order_cal], y_mid_cal[order_cal], color="black", linewidth=2, label="CQR center")

    bin_stats = result.get("bin_stats") or []
    bin_source = result.get("bin_stats_source", "quantiles")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(bin_stats), 1)))

    if bin_stats:
        in_band_cal = (Y_cal >= lower_cal) & (Y_cal <= upper_cal)
        for stats, color in zip(bin_stats, colors):
            mask = (X_cal >= stats["x_min"]) & (X_cal <= stats["x_max"])
            if not np.any(mask):
                continue
            X_g = X_cal[mask]
            order_g = np.argsort(X_g)
            lower_g = lower_cal[mask][order_g]
            upper_g = upper_cal[mask][order_g]
            label_id = stats.get("group") if bin_source == "groups" else stats.get("bin")
            label_prefix = "group" if bin_source == "groups" else "bin"
            ax.fill_between(
                X_g[order_g],
                lower_g,
                upper_g,
                alpha=0.25,
                color=color,
                label=f"{label_prefix} {label_id}",
            )
            ax.scatter(
                X_g[in_band_cal[mask]],
                Y_cal[mask][in_band_cal[mask]],
                s=10,
                color=color,
                marker="o",
            )
            ax.scatter(
                X_g[~in_band_cal[mask]],
                Y_cal[mask][~in_band_cal[mask]],
                s=20,
                color=color,
                marker="x",
            )
    else:
        ax.fill_between(
            X_cal[order_cal],
            lower_cal[order_cal],
            upper_cal[order_cal],
            alpha=0.25,
            label="CQR interval",
        )
        in_band_cal = (Y_cal >= lower_cal) & (Y_cal <= upper_cal)
        ax.scatter(
            X_cal[in_band_cal],
            Y_cal[in_band_cal],
            s=15,
            label="covered calibration",
        )
        ax.scatter(
            X_cal[~in_band_cal],
            Y_cal[~in_band_cal],
            s=25,
            marker="x",
            label="not covered (calibration)",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Conformalized Quantile Regression (calibration set)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "cqr_calibration_intervals.png" if output_dir else None
    _finalize(fig, out_file, show)


def plot_ltt_results(result: dict, output_dir: Optional[Path] = None, show: bool = False) -> None:
    lambda_grid = np.asarray(result["lambda_grid"])
    R_hat = np.asarray(result["R_hat"])
    R_upper = np.asarray(result["R_upper"])
    alpha = float(result["alpha"])
    safe_mask = np.asarray(result["safe_mask"])
    safe_lambdas = np.asarray(result["safe_lambdas"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambda_grid, R_hat, label=r"empirical risk $\hat R_n(\lambda)$")
    ax.plot(lambda_grid, R_upper, label="uniform upper bound $R(\\lambda)$")
    ax.axhline(alpha, color="k", linestyle="--", label=f"target risk α = {alpha}")
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Risk")
    ax.set_title("Uniform-in-λ risk bounds")
    ax.grid(True)
    out_file = output_dir / "ltt_uniform_bounds.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Calibration band plot
    X_cal = np.asarray(result["X_cal"]).ravel()
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    y_hat_cal = np.asarray(result["y_hat_cal"]).ravel()
    lower = np.asarray(result["lower_cal"]).ravel()
    upper = np.asarray(result["upper_cal"]).ravel()
    order = np.argsort(X_cal)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_cal[order], y_hat_cal[order], label="prediction ŷ(x)", linewidth=2)
    ax.fill_between(X_cal[order], lower[order], upper[order], alpha=0.2, label="certified band")
    in_band = (Y_cal >= lower) & (Y_cal <= upper)
    ax.scatter(X_cal[in_band], Y_cal[in_band], s=15, label="covered samples")
    ax.scatter(X_cal[~in_band], Y_cal[~in_band], s=20, marker="x", label="not covered")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Best LTT-certified band (calibration set, λ = {result['lambda_best']:.3f})")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_calibration_band.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Test set plot
    X_test = np.asarray(result["X_test"]).ravel()
    Y_test = np.asarray(result["Y_test"]).ravel()
    lower_test = np.asarray(result["lower_test"]).ravel()
    upper_test = np.asarray(result["upper_test"]).ravel()
    y_hat_test = np.asarray(result["y_hat_test"]).ravel()
    order_test = np.argsort(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_test[order_test], y_hat_test[order_test], label="prediction ŷ(x)", linewidth=2)
    ax.fill_between(
        X_test[order_test],
        lower_test[order_test],
        upper_test[order_test],
        alpha=0.2,
        label="certified band",
    )
    in_band_test = (Y_test >= lower_test) & (Y_test <= upper_test)
    ax.scatter(
        X_test[in_band_test],
        Y_test[in_band_test],
        s=15,
        label="covered test samples",
    )
    ax.scatter(
        X_test[~in_band_test],
        Y_test[~in_band_test],
        s=20,
        marker="x",
        label="not covered (test)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Best LTT-certified band (test set, λ = {result['lambda_best']:.3f})")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_test_band.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Safe λ scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambda_grid, R_upper, label="uniform upper bound")
    ax.axhline(alpha, color="k", linestyle="--", label="target α")
    ax.scatter(lambda_grid[safe_mask], R_upper[safe_mask], label="certified safe λ", zorder=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Risk upper bound")
    ax.set_title("Safe tolerances λ where risk ≤ α (uniform guarantee)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_safe_lambdas.png" if output_dir else None
    _finalize(fig, out_file, show)
    return

    # Calibration coverage across all safe λ values
    if safe_lambdas.size:
        colors = plt.cm.viridis(np.linspace(0, 1, len(safe_lambdas)))
        fig, ax = plt.subplots(figsize=(8, 6))
        for lam, color in zip(safe_lambdas, colors):
            band_lower = y_hat_cal - lam
            band_upper = y_hat_cal + lam
            ax.fill_between(
                X_cal[order],
                band_lower[order],
                band_upper[order],
                alpha=0.1,
                color=color,
                label=f"λ = {lam:.3f}",
            )
        ax.plot(X_cal[order], y_hat_cal[order], color="black", linewidth=2, label="ŷ(x)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("LTT-certified bands for all safe λ values")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)
        fig.tight_layout()
        out_file = output_dir / "ltt_safe_bands.png" if output_dir else None
        _finalize(fig, out_file, show)

    # Group-specific bands
    group_bands = result.get("group_bands", [])
    group_test_bands = result.get("group_test_bands", [])
    if group_bands:
        fig, ax = plt.subplots(figsize=(9, 5))
        # Fixed color palette that cycles (matching notebook)
        color_palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        for band in group_bands:
            color = color_palette[band["group"] % len(color_palette)]
            X_g = np.asarray(band["X"]).ravel()
            Y_g = np.asarray(band["Y"]).ravel()
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band_g = np.asarray(band["in_band"]).ravel()
            order_g = np.argsort(X_g)
            ax.plot(
                X_g[order_g],
                Yhat_g[order_g],
                color=color,
                linewidth=1.5,
                label=f"group {band['group']} ŷ(x)",
            )
            ax.fill_between(
                X_g[order_g],
                lower_g[order_g],
                upper_g[order_g],
                color=color,
                alpha=0.15,
            )
            ax.scatter(
                X_g[in_band_g],
                Y_g[in_band_g],
                s=15,
                color=color,
                marker="o",
                label=None,
            )
            ax.scatter(
                X_g[~in_band_g],
                Y_g[~in_band_g],
                s=25,
                color=color,
                marker="x",
                label=None,
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Group-conditional LTT bands (calibration)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_file = output_dir / "ltt_group_bands.png" if output_dir else None
        _finalize(fig, out_file, show)

    if group_test_bands:
        fig, ax = plt.subplots(figsize=(9, 5))
        color_palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        for band in group_test_bands:
            color = color_palette[band["group"] % len(color_palette)]
            X_g = np.asarray(band["X"]).ravel()
            Y_g = np.asarray(band["Y"]).ravel()
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band_g = np.asarray(band["in_band"]).ravel()
            order_g = np.argsort(X_g)
            ax.plot(
                X_g[order_g],
                Yhat_g[order_g],
                color=color,
                linewidth=1.5,
                label=f"group {band['group']} ŷ(x)",
            )
            ax.fill_between(
                X_g[order_g],
                lower_g[order_g],
                upper_g[order_g],
                color=color,
                alpha=0.15,
            )
            if len(X_g):
                ax.scatter(
                    X_g[in_band_g],
                    Y_g[in_band_g],
                    s=15,
                    color=color,
                    marker="o",
                    label=None,
                )
                ax.scatter(
                    X_g[~in_band_g],
                    Y_g[~in_band_g],
                    s=25,
                    color=color,
                    marker="x",
                    label=None,
                )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Group-conditional LTT bands (test set)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_file = output_dir / "ltt_group_test_bands.png" if output_dir else None
        _finalize(fig, out_file, show)


def plot_single_empirical_results(result: dict, output_dir: Optional[Path] = None, show: bool = False) -> None:
    X_cal = np.asarray(result["X_cal"]).ravel()
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    y_hat_cal = np.asarray(result["y_hat_cal"]).ravel()
    X_test = np.asarray(result["X_test"]).ravel()
    Y_test = np.asarray(result["Y_test"]).ravel()
    y_hat_test = np.asarray(result["y_hat_test"]).ravel()
    lam = float(result["lambda"])

    lower_cal = y_hat_cal - lam
    upper_cal = y_hat_cal + lam
    lower_test = y_hat_test - lam
    upper_test = y_hat_test + lam

    group_bands = result.get("group_bands", [])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_bands), 1)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_cal[np.argsort(X_cal)], y_hat_cal[np.argsort(X_cal)], color="black", linewidth=2, label="ŷ(x)")
    if group_bands:
        for band, color in zip(group_bands, colors):
            X_g = np.asarray(band["X"]).ravel()
            order_g = np.argsort(X_g)
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            ax.fill_between(
                X_g[order_g],
                lower_g[order_g],
                upper_g[order_g],
                alpha=0.25,
                color=color,
                label=f"group {band['group']}",
            )
            in_band = np.asarray(band["in_band"]).ravel()
            ax.scatter(X_g[in_band], np.asarray(band["Y"]).ravel()[in_band], s=10, color=color, marker="o")
            ax.scatter(X_g[~in_band], np.asarray(band["Y"]).ravel()[~in_band], s=20, color=color, marker="x")
    else:
        order = np.argsort(X_cal)
        ax.fill_between(X_cal[order], lower_cal[order], upper_cal[order], alpha=0.25, label="interval")
        in_band = (Y_cal >= lower_cal) & (Y_cal <= upper_cal)
        ax.scatter(X_cal[in_band], Y_cal[in_band], s=10, label="covered")
        ax.scatter(X_cal[~in_band], Y_cal[~in_band], s=20, marker="x", label="missed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Single-model empirical bands (calibration)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "single_empirical_calibration.png" if output_dir else None
    _finalize(fig, out_file, show)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_test[np.argsort(X_test)], y_hat_test[np.argsort(X_test)], color="black", linewidth=2, label="ŷ(x)")
    group_test = result.get("group_test_bands", [])
    if group_test:
        for band, color in zip(group_test, colors):
            X_g = np.asarray(band["X"]).ravel()
            if X_g.size == 0:
                continue
            order_g = np.argsort(X_g)
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            ax.fill_between(
                X_g[order_g],
                lower_g[order_g],
                upper_g[order_g],
                alpha=0.25,
                color=color,
                label=f"group {band['group']}",
            )
            in_band = np.asarray(band["in_band"]).ravel()
            ax.scatter(X_g[in_band], np.asarray(band["Y"]).ravel()[in_band], s=10, color=color, marker="o")
            ax.scatter(X_g[~in_band], np.asarray(band["Y"]).ravel()[~in_band], s=20, color=color, marker="x")
    else:
        order = np.argsort(X_test)
        ax.fill_between(X_test[order], lower_test[order], upper_test[order], alpha=0.25, label="interval")
        in_band = (Y_test >= lower_test) & (Y_test <= upper_test)
        ax.scatter(X_test[in_band], Y_test[in_band], s=10, label="covered")
        ax.scatter(X_test[~in_band], Y_test[~in_band], s=20, marker="x", label="missed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Single-model empirical bands (test)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "single_empirical_test.png" if output_dir else None
    _finalize(fig, out_file, show)


# =============================================================================
# Multi-dimensional variants (X has more than one feature)
# These use sorted sample indices on the x-axis and reuse the same band/marker
# semantics as the 1D notebook-style plots, without modifying the original
# 1D plotting functions.
# =============================================================================


def plot_marginal_results_multi(
    result: dict,
    output_dir: Optional[Path] = None,
    show: bool = False,
    plot_options: Optional[Dict[str, object]] = None,
) -> None:
    X_test = np.asarray(result["X_test"])
    Y_test = np.asarray(result["Y_test"]).ravel()
    lower = np.asarray(result["lower_test"]).ravel()
    upper = np.asarray(result["upper_test"]).ravel()
    y_hat = np.asarray(result["Y_hat_test"]).ravel()

    order = np.argsort(Y_test)
    Y_test_sorted = Y_test[order]
    y_hat_sorted = y_hat[order]
    lower_sorted = lower[order]
    upper_sorted = upper[order]
    in_band_sorted = ((Y_test >= lower) & (Y_test <= upper))[order]
    sel_test = _select_plot_indices(len(Y_test_sorted), plot_options)
    x = sel_test
    Y_test_view = Y_test_sorted[sel_test]
    y_hat_view = y_hat_sorted[sel_test]
    lower_view = lower_sorted[sel_test]
    upper_view = upper_sorted[sel_test]
    in_band_view = in_band_sorted[sel_test]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_hat_view, label="prediction ŷ", linewidth=2)
    ax.plot(x, Y_test_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(x, lower_view, upper_view, alpha=0.2, label="conformal band")
    viol_idx = np.where(~in_band_view)[0]
    if len(viol_idx) > 0:
        ax.scatter(viol_idx, Y_test_view[viol_idx], s=12, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Marginal conformal prediction band (test set, multi-D)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "marginal_test_band_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Calibration set
    X_cal = np.asarray(result["X_cal"])
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    lower_cal = np.asarray(result["lower_cal"]).ravel()
    upper_cal = np.asarray(result["upper_cal"]).ravel()
    y_hat_cal = np.asarray(result["Y_hat_cal"]).ravel()

    order_cal = np.argsort(Y_cal)
    Y_cal_sorted = Y_cal[order_cal]
    y_hat_cal_sorted = y_hat_cal[order_cal]
    lower_cal_sorted = lower_cal[order_cal]
    upper_cal_sorted = upper_cal[order_cal]
    in_band_cal_sorted = ((Y_cal >= lower_cal) & (Y_cal <= upper_cal))[order_cal]
    sel_cal = _select_plot_indices(len(Y_cal_sorted), plot_options)
    x_cal = sel_cal
    Y_cal_view = Y_cal_sorted[sel_cal]
    y_hat_cal_view = y_hat_cal_sorted[sel_cal]
    lower_cal_view = lower_cal_sorted[sel_cal]
    upper_cal_view = upper_cal_sorted[sel_cal]
    in_band_cal_view = in_band_cal_sorted[sel_cal]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_cal, y_hat_cal_view, label="prediction ŷ", linewidth=2)
    ax.plot(x_cal, Y_cal_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(x_cal, lower_cal_view, upper_cal_view, alpha=0.2, label="conformal band")
    viol_idx_cal = np.where(~in_band_cal_view)[0]
    if len(viol_idx_cal) > 0:
        ax.scatter(
            viol_idx_cal,
            Y_cal_view[viol_idx_cal],
            s=12,
            marker="x",
            color="red",
            label="violations",
        )
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Marginal conformal prediction band (calibration set, multi-D)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "marginal_calibration_band_multi.png" if output_dir else None
    _finalize(fig, out_file, show)


def plot_cqr_results_multi(
    result: dict,
    output_dir: Optional[Path] = None,
    show: bool = False,
    plot_options: Optional[Dict[str, object]] = None,
) -> None:
    X_test = np.asarray(result["X_test"])
    Y_test = np.asarray(result["Y_test"]).ravel()
    lower = np.asarray(result["lower_test"]).ravel()
    upper = np.asarray(result["upper_test"]).ravel()
    y_mid = 0.5 * (np.asarray(result["q_low_test"]) + np.asarray(result["q_high_test"]))

    order = np.argsort(Y_test)
    Y_test_sorted = Y_test[order]
    y_mid_sorted = y_mid[order]
    lower_sorted = lower[order]
    upper_sorted = upper[order]
    in_band_sorted = ((Y_test >= lower) & (Y_test <= upper))[order]
    sel_test = _select_plot_indices(len(Y_test_sorted), plot_options)
    x = sel_test
    y_mid_view = y_mid_sorted[sel_test]
    Y_test_view = Y_test_sorted[sel_test]
    lower_view = lower_sorted[sel_test]
    upper_view = upper_sorted[sel_test]
    in_band_view = in_band_sorted[sel_test]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_mid_view, label="CQR center (mid-quantile)", linewidth=2)
    ax.plot(x, Y_test_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(
        x,
        lower_view,
        upper_view,
        alpha=0.25,
        label=f"CQR interval (target {(1 - result['alpha']) * 100:.0f}% coverage)",
    )
    viol_idx = np.where(~in_band_view)[0]
    if len(viol_idx) > 0:
        ax.scatter(viol_idx, Y_test_view[viol_idx], s=15, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Conformalized Quantile Regression (test set, multi-D)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "cqr_test_intervals_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Calibration set
    X_cal = np.asarray(result["X_cal"])
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    lower_cal = np.asarray(result["lower_cal"]).ravel()
    upper_cal = np.asarray(result["upper_cal"]).ravel()
    y_mid_cal = 0.5 * (np.asarray(result["q_low_cal"]) + np.asarray(result["q_high_cal"]))

    order_cal = np.argsort(Y_cal)
    Y_cal_sorted = Y_cal[order_cal]
    y_mid_cal_sorted = y_mid_cal[order_cal]
    lower_cal_sorted = lower_cal[order_cal]
    upper_cal_sorted = upper_cal[order_cal]
    in_band_cal_sorted = ((Y_cal >= lower_cal) & (Y_cal <= upper_cal))[order_cal]
    sel_cal = _select_plot_indices(len(Y_cal_sorted), plot_options)
    x_cal = sel_cal
    y_mid_cal_view = y_mid_cal_sorted[sel_cal]
    Y_cal_view = Y_cal_sorted[sel_cal]
    lower_cal_view = lower_cal_sorted[sel_cal]
    upper_cal_view = upper_cal_sorted[sel_cal]
    in_band_cal_view = in_band_cal_sorted[sel_cal]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_cal, y_mid_cal_view, color="black", linewidth=2, label="CQR center")
    ax.plot(x_cal, Y_cal_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(x_cal, lower_cal_view, upper_cal_view, alpha=0.25, label="CQR interval")
    viol_idx_cal = np.where(~in_band_cal_view)[0]
    if len(viol_idx_cal) > 0:
        ax.scatter(viol_idx_cal, Y_cal_view[viol_idx_cal], s=15, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Conformalized Quantile Regression (calibration set, multi-D)")
    ax.legend()
    ax.grid(True)

    out_file = output_dir / "cqr_calibration_intervals_multi.png" if output_dir else None
    _finalize(fig, out_file, show)


def plot_ltt_results_multi(
    result: dict,
    output_dir: Optional[Path] = None,
    show: bool = False,
    plot_options: Optional[Dict[str, object]] = None,
) -> None:
    """
    Multi-dimensional variant: same λ-plots as before, but calibration/test and
    group plots use sorted sample indices on the x-axis.
    """
    lambda_grid = np.asarray(result["lambda_grid"])
    R_hat = np.asarray(result["R_hat"])
    R_upper = np.asarray(result["R_upper"])
    alpha = float(result["alpha"])
    safe_mask = np.asarray(result["safe_mask"])
    safe_lambdas = np.asarray(result["safe_lambdas"])

    # Uniform risk bounds (unchanged)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambda_grid, R_hat, label=r"empirical risk $\hat R_n(\lambda)$")
    ax.plot(lambda_grid, R_upper, label="uniform upper bound $R(\\lambda)$")
    ax.axhline(alpha, color="k", linestyle="--", label=f"target risk α = {alpha}")
    ax.legend()
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Risk")
    ax.set_title("Uniform-in-λ risk bounds")
    ax.grid(True)
    out_file = output_dir / "ltt_uniform_bounds_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    Y_cal = np.asarray(result["Y_cal"]).ravel()
    y_hat_cal = np.asarray(result["y_hat_cal"]).ravel()
    lower_cal = np.asarray(result["lower_cal"]).ravel()
    upper_cal = np.asarray(result["upper_cal"]).ravel()
    order_cal = np.argsort(Y_cal)
    Y_cal_sorted = Y_cal[order_cal]
    y_hat_cal_sorted = y_hat_cal[order_cal]
    lower_cal_sorted = lower_cal[order_cal]
    upper_cal_sorted = upper_cal[order_cal]
    in_band_cal_sorted = ((Y_cal >= lower_cal) & (Y_cal <= upper_cal))[order_cal]
    sel_cal = _select_plot_indices(len(Y_cal_sorted), plot_options)
    x_cal = sel_cal
    Y_cal_view = Y_cal_sorted[sel_cal]
    y_hat_cal_view = y_hat_cal_sorted[sel_cal]
    lower_cal_view = lower_cal_sorted[sel_cal]
    upper_cal_view = upper_cal_sorted[sel_cal]
    in_band_cal_view = in_band_cal_sorted[sel_cal]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_cal, y_hat_cal_view, label="prediction ŷ", linewidth=2)
    ax.plot(x_cal, Y_cal_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(x_cal, lower_cal_view, upper_cal_view, alpha=0.2, label="certified band")
    viol_idx_cal = np.where(~in_band_cal_view)[0]
    if len(viol_idx_cal) > 0:
        ax.scatter(viol_idx_cal, Y_cal_view[viol_idx_cal], s=15, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title(f"Best LTT-certified band (calibration set, λ = {result['lambda_best']:.3f})")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_calibration_band_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    Y_test = np.asarray(result["Y_test"]).ravel()
    y_hat_test = np.asarray(result["y_hat_test"]).ravel()
    lower_test = np.asarray(result["lower_test"]).ravel()
    upper_test = np.asarray(result["upper_test"]).ravel()
    order_test = np.argsort(Y_test)
    Y_test_sorted = Y_test[order_test]
    y_hat_test_sorted = y_hat_test[order_test]
    lower_test_sorted = lower_test[order_test]
    upper_test_sorted = upper_test[order_test]
    in_band_test_sorted = ((Y_test >= lower_test) & (Y_test <= upper_test))[order_test]
    sel_test = _select_plot_indices(len(Y_test_sorted), plot_options)
    x_test = sel_test
    Y_test_view = Y_test_sorted[sel_test]
    y_hat_test_view = y_hat_test_sorted[sel_test]
    lower_test_view = lower_test_sorted[sel_test]
    upper_test_view = upper_test_sorted[sel_test]
    in_band_test_view = in_band_test_sorted[sel_test]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_test, y_hat_test_view, label="prediction ŷ", linewidth=2)
    ax.plot(x_test, Y_test_view, linewidth=1.2, linestyle="--", label="truth")
    ax.fill_between(x_test, lower_test_view, upper_test_view, alpha=0.2, label="certified band")
    viol_idx_test = np.where(~in_band_test_view)[0]
    if len(viol_idx_test) > 0:
        ax.scatter(viol_idx_test, Y_test_view[viol_idx_test], s=15, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title(f"Best LTT-certified band (test set, λ = {result['lambda_best']:.3f})")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_test_band_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Safe λ scatter (unchanged)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambda_grid, R_upper, label="uniform upper bound")
    ax.axhline(alpha, color="k", linestyle="--", label="target α")
    ax.scatter(lambda_grid[safe_mask], R_upper[safe_mask], label="certified safe λ", zorder=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Risk upper bound")
    ax.set_title("Safe tolerances λ where risk ≤ α (uniform guarantee)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "ltt_safe_lambdas_multi.png" if output_dir else None
    _finalize(fig, out_file, show)
    return

    # Group-specific bands using indices
    group_bands = result.get("group_bands", [])
    group_test_bands = result.get("group_test_bands", [])
    if group_bands:
        fig, ax = plt.subplots(figsize=(9, 5))
        color_palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        selected_cal_positions = sel_cal if "sel_cal" in locals() else np.arange(len(Y_cal_sorted))

        for band in group_bands:
            color = color_palette[band["group"] % len(color_palette)]
            Y_g = np.asarray(band["Y"]).ravel()
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band_g = np.asarray(band["in_band"]).ravel()

            global_positions = []
            used_positions = set()
            for y_val in Y_g:
                matches = np.where(np.abs(Y_cal_sorted - y_val) < 1e-10)[0]
                if len(matches) > 0:
                    for pos in matches:
                        if pos not in used_positions:
                            global_positions.append(pos)
                            used_positions.add(pos)
                            break
                    else:
                        global_positions.append(matches[0])
                else:
                    closest = np.argmin(np.abs(Y_cal_sorted - y_val))
                    global_positions.append(closest)

            global_positions = np.array(global_positions)
            sort_order = np.argsort(global_positions)
            global_positions_sorted = global_positions[sort_order]

            mask = np.isin(global_positions_sorted, selected_cal_positions)
            if not np.any(mask):
                continue
            x_vals = global_positions_sorted[mask]
            Y_g_sorted = Y_g[sort_order][mask]
            Yhat_g_sorted = Yhat_g[sort_order][mask]
            lower_g_sorted = lower_g[sort_order][mask]
            upper_g_sorted = upper_g[sort_order][mask]
            in_band_g_sorted = in_band_g[sort_order][mask]

            ax.plot(
                x_vals,
                Yhat_g_sorted,
                color=color,
                linewidth=1.5,
                label=f"group {band['group']} ŷ",
            )
            ax.fill_between(
                x_vals,
                lower_g_sorted,
                upper_g_sorted,
                color=color,
                alpha=0.15,
            )
            ax.scatter(
                x_vals[in_band_g_sorted],
                Y_g_sorted[in_band_g_sorted],
                s=15,
                color=color,
                marker="o",
                label=None,
            )
            ax.scatter(
                x_vals[~in_band_g_sorted],
                Y_g_sorted[~in_band_g_sorted],
                s=25,
                color=color,
                marker="x",
                label=None,
            )
        ax.set_xlabel("sorted sample index")
        ax.set_ylabel("y")
        ax.set_title("Group-conditional LTT bands (calibration, multi-D)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_file = output_dir / "ltt_group_bands_multi.png" if output_dir else None
        _finalize(fig, out_file, show)

    if group_test_bands:
        fig, ax = plt.subplots(figsize=(9, 5))
        color_palette = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        selected_test_positions = sel_test if "sel_test" in locals() else np.arange(len(Y_test_sorted))

        for band in group_test_bands:
            color = color_palette[band["group"] % len(color_palette)]
            Y_g = np.asarray(band["Y"]).ravel()
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band_g = np.asarray(band["in_band"]).ravel()
            if Y_g.size == 0:
                continue

            global_positions = []
            used_positions = set()
            for y_val in Y_g:
                matches = np.where(np.abs(Y_test_sorted - y_val) < 1e-10)[0]
                if len(matches) > 0:
                    for pos in matches:
                        if pos not in used_positions:
                            global_positions.append(pos)
                            used_positions.add(pos)
                            break
                    else:
                        global_positions.append(matches[0])
                else:
                    closest = np.argmin(np.abs(Y_test_sorted - y_val))
                    global_positions.append(closest)

            global_positions = np.array(global_positions)
            sort_order = np.argsort(global_positions)
            global_positions_sorted = global_positions[sort_order]

            mask = np.isin(global_positions_sorted, selected_test_positions)
            if not np.any(mask):
                continue
            x_vals = global_positions_sorted[mask]
            Y_g_sorted = Y_g[sort_order][mask]
            Yhat_g_sorted = Yhat_g[sort_order][mask]
            lower_g_sorted = lower_g[sort_order][mask]
            upper_g_sorted = upper_g[sort_order][mask]
            in_band_g_sorted = in_band_g[sort_order][mask]

            ax.plot(
                x_vals,
                Yhat_g_sorted,
                color=color,
                linewidth=1.5,
                label=f"group {band['group']} ŷ",
            )
            ax.fill_between(
                x_vals,
                lower_g_sorted,
                upper_g_sorted,
                color=color,
                alpha=0.15,
            )
            ax.scatter(
                x_vals[in_band_g_sorted],
                Y_g_sorted[in_band_g_sorted],
                s=15,
                color=color,
                marker="o",
                label=None,
            )
            ax.scatter(
                x_vals[~in_band_g_sorted],
                Y_g_sorted[~in_band_g_sorted],
                s=25,
                color=color,
                marker="x",
                label=None,
            )
        ax.set_xlabel("sorted sample index")
        ax.set_ylabel("y")
        ax.set_title("Group-conditional LTT bands (test set, multi-D)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_file = output_dir / "ltt_group_test_bands_multi.png" if output_dir else None
        _finalize(fig, out_file, show)


def plot_single_empirical_results_multi(
    result: dict,
    output_dir: Optional[Path] = None,
    show: bool = False,
    plot_options: Optional[Dict[str, object]] = None,
) -> None:
    """
    Multi-dimensional variant for the single-model empirical case: x-axis is
    the sorted sample index, with the same group coloring and band logic.
    """
    Y_cal = np.asarray(result["Y_cal"]).ravel()
    y_hat_cal = np.asarray(result["y_hat_cal"]).ravel()
    Y_test = np.asarray(result["Y_test"]).ravel()
    y_hat_test = np.asarray(result["y_hat_test"]).ravel()
    lam = float(result["lambda"])

    lower_cal = y_hat_cal - lam
    upper_cal = y_hat_cal + lam
    lower_test = y_hat_test - lam
    upper_test = y_hat_test + lam

    group_bands = result.get("group_bands", [])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_bands), 1)))

    # Calibration
    fig, ax = plt.subplots(figsize=(8, 5))
    # Sort by true target for calibration
    order_cal = np.argsort(Y_cal)
    x_cal = np.arange(len(Y_cal))
    
    # Sort all arrays by true target
    Y_cal_sorted = Y_cal[order_cal]
    y_hat_cal_sorted = y_hat_cal[order_cal]
    lower_cal_sorted = lower_cal[order_cal]
    upper_cal_sorted = upper_cal[order_cal]
    in_band_cal = (Y_cal >= lower_cal) & (Y_cal <= upper_cal)
    in_band_cal_sorted = in_band_cal[order_cal]
    
    ax.plot(x_cal, y_hat_cal_sorted, color="black", linewidth=2, label="ŷ")
    ax.plot(x_cal, Y_cal_sorted, linewidth=1.2, linestyle="--", label="truth")
    
    if group_bands:
        for band, color in zip(group_bands, colors):
            Y_g = np.asarray(band["Y"]).ravel()
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band = np.asarray(band["in_band"]).ravel()
            
            # For each Y_g value, find its position in the sorted Y_cal array
            global_positions = []
            used_positions = set()
            for y_val in Y_g:
                matches = np.where(np.abs(Y_cal_sorted - y_val) < 1e-10)[0]
                if len(matches) > 0:
                    for pos in matches:
                        if pos not in used_positions:
                            global_positions.append(pos)
                            used_positions.add(pos)
                            break
                    else:
                        global_positions.append(matches[0])
                else:
                    closest = np.argmin(np.abs(Y_cal_sorted - y_val))
                    global_positions.append(closest)
            
            global_positions = np.array(global_positions)
            sort_order = np.argsort(global_positions)
            global_positions_sorted = global_positions[sort_order]
            
            Y_g_sorted = Y_g[sort_order]
            lower_g_sorted = lower_g[sort_order]
            upper_g_sorted = upper_g[sort_order]
            in_band_sorted = in_band[sort_order]
            
            ax.fill_between(
                global_positions_sorted,
                lower_g_sorted,
                upper_g_sorted,
                alpha=0.25,
                color=color,
                label=f"group {band['group']}",
            )
            ax.scatter(global_positions_sorted[in_band_sorted], Y_g_sorted[in_band_sorted], s=10, color=color, marker="o")
            ax.scatter(global_positions_sorted[~in_band_sorted], Y_g_sorted[~in_band_sorted], s=20, color=color, marker="x")
    else:
        ax.fill_between(x_cal, lower_cal_sorted, upper_cal_sorted, alpha=0.25, label="interval")
        viol_idx_cal = np.where(~in_band_cal_sorted)[0]
        if len(viol_idx_cal) > 0:
            ax.scatter(viol_idx_cal, Y_cal_sorted[viol_idx_cal], s=10, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Single-model empirical bands (calibration, multi-D)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "single_empirical_calibration_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

    # Test
    fig, ax = plt.subplots(figsize=(8, 5))
    # Sort by true target on test
    order_test = np.argsort(Y_test)
    x_test = np.arange(len(Y_test))
    
    # Sort all arrays by true target
    Y_test_sorted = Y_test[order_test]
    y_hat_test_sorted = y_hat_test[order_test]
    lower_test_sorted = lower_test[order_test]
    upper_test_sorted = upper_test[order_test]
    in_band_test = (Y_test >= lower_test) & (Y_test <= upper_test)
    in_band_test_sorted = in_band_test[order_test]
    
    ax.plot(x_test, y_hat_test_sorted, color="black", linewidth=2, label="ŷ")
    ax.plot(x_test, Y_test_sorted, linewidth=1.2, linestyle="--", label="truth")
    
    group_test = result.get("group_test_bands", [])
    if group_test:
        # Plot group-specific bands
        for band, color in zip(group_test, colors):
            Y_g = np.asarray(band["Y"]).ravel()
            if Y_g.size == 0:
                continue
            Yhat_g = np.asarray(band["Yhat"]).ravel()
            lower_g = np.asarray(band["lower"]).ravel()
            upper_g = np.asarray(band["upper"]).ravel()
            in_band = np.asarray(band["in_band"]).ravel()
            
            # For each Y_g value, find its position in the sorted Y_test array
            global_positions = []
            used_positions = set()
            for y_val in Y_g:
                matches = np.where(np.abs(Y_test_sorted - y_val) < 1e-10)[0]
                if len(matches) > 0:
                    for pos in matches:
                        if pos not in used_positions:
                            global_positions.append(pos)
                            used_positions.add(pos)
                            break
                    else:
                        global_positions.append(matches[0])
                else:
                    closest = np.argmin(np.abs(Y_test_sorted - y_val))
                    global_positions.append(closest)
            
            global_positions = np.array(global_positions)
            sort_order = np.argsort(global_positions)
            global_positions_sorted = global_positions[sort_order]
            
            Y_g_sorted = Y_g[sort_order]
            lower_g_sorted = lower_g[sort_order]
            upper_g_sorted = upper_g[sort_order]
            in_band_sorted = in_band[sort_order]
            
            ax.fill_between(
                global_positions_sorted,
                lower_g_sorted,
                upper_g_sorted,
                alpha=0.25,
                color=color,
                label=f"group {band['group']}",
            )
            ax.scatter(global_positions_sorted[in_band_sorted], Y_g_sorted[in_band_sorted], s=10, color=color, marker="o")
            ax.scatter(global_positions_sorted[~in_band_sorted], Y_g_sorted[~in_band_sorted], s=20, color=color, marker="x")
    # Always show overall test band
    ax.fill_between(x_test, lower_test_sorted, upper_test_sorted, alpha=0.15, color="gray", label="overall interval")
    viol_idx_test = np.where(~in_band_test_sorted)[0]
    if len(viol_idx_test) > 0:
        ax.scatter(viol_idx_test, Y_test_sorted[viol_idx_test], s=10, marker="x", color="red", label="violations")
    ax.set_xlabel("sorted sample index")
    ax.set_ylabel("y")
    ax.set_title("Single-model empirical bands (test, multi-D)")
    ax.legend()
    ax.grid(True)
    out_file = output_dir / "single_empirical_test_multi.png" if output_dir else None
    _finalize(fig, out_file, show)

