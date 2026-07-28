"""
Standalone model training pipeline so statisticians can reuse trained models.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone

try:
    from xgboost import XGBRegressor, Booster  # type: ignore
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None
    Booster = None
    xgb = None

from .models import ModelBundle, fit_quantile_regressors, fit_regressor
from .io_utils import load_split, ensure_dir
from .exporters import save_mu_band_csv

TrainingTarget = Literal["regressor", "quantile"]


@dataclass
class TrainingConfig:
    train_data: str
    val_data: Optional[str] = None
    tl_data: Optional[str] = None
    alpha: float = 0.1
    target: TrainingTarget = "regressor"
    regressor_model: str = "linear"
    regressor_params: Dict[str, object] = field(default_factory=dict)
    quantile_model: str = "gbr"
    quantile_params: Dict[str, object] = field(default_factory=dict)
    output_dir: Path | None = None
    seed: int = 0
    tl_add_trees: int = 200
    tl_eta: float = 0.05
    mu_band: bool = False
    mu_seeds: Optional[list[int]] = None
    mu_bootstrap: bool = False
    mu_bootstrap_frac: float = 1.0


def _transfer_learn_xgb(
    zs_model: XGBRegressor,
    X_tl: np.ndarray,
    y_tl: np.ndarray,
    add_trees: int,
    eta_ft: float,
) -> XGBRegressor:
    """
    Fine-tune an XGBoost model on transfer learning data by continuing training.
    
    Args:
        zs_model: Zero-shot model trained on source data
        X_tl: Transfer learning features
        y_tl: Transfer learning targets
        add_trees: Number of additional trees to add
        eta_ft: Learning rate for fine-tuning
    
    Returns:
        Fine-tuned model
    """
    # Get the booster from the zero-shot model
    booster = zs_model.get_booster()
    
    # Get current number of trees
    curr_trees = zs_model.get_params().get("n_estimators", 300)
    
    # Create a new model with updated parameters
    # Exclude xgb_model, learning_rate, and n_estimators from copied params to avoid conflicts
    zs_params = zs_model.get_params()
    tl_params = {k: v for k, v in zs_params.items() if k not in {"xgb_model", "learning_rate", "n_estimators"}}
    tl_model = XGBRegressor(
        **tl_params,
        learning_rate=eta_ft,
        n_estimators=curr_trees + add_trees,
    )
    
    # Fine-tune on TL data, continuing from the existing booster
    # Pass xgb_model to fit() method, not constructor
    tl_model.fit(X_tl, y_tl.ravel(), xgb_model=booster)
    
    return tl_model


def _transfer_learn_gbr(
    zs_model: GradientBoostingRegressor,
    X_tl: np.ndarray,
    y_tl: np.ndarray,
    add_trees: int,
    eta_ft: float,
) -> GradientBoostingRegressor:
    """
    Fine-tune a GradientBoostingRegressor on transfer learning data.
    
    Uses warm_start to continue training from the zero-shot model.
    Note: sklearn's warm_start requires the model to be fitted first.
    """
    # Get current parameters
    curr_params = zs_model.get_params()
    curr_trees = curr_params.get("n_estimators", 100)
    
    # Create a new model with warm_start, copying zero-shot model's parameters
    tl_model = GradientBoostingRegressor(
        loss=curr_params.get("loss", "squared_error"),
        learning_rate=eta_ft,
        n_estimators=curr_trees,  # Start with current number
        max_depth=curr_params.get("max_depth", 3),
        min_samples_split=curr_params.get("min_samples_split", 2),
        min_samples_leaf=curr_params.get("min_samples_leaf", 1),
        subsample=curr_params.get("subsample", 1.0),
        max_features=curr_params.get("max_features", None),
        random_state=curr_params.get("random_state", None),
        warm_start=True,
    )
    
    # Fit with current number of trees to establish state
    tl_model.fit(X_tl, y_tl.ravel())
    
    # Manually copy the zero-shot model's estimators
    tl_model.estimators_ = zs_model.estimators_
    tl_model.train_score_ = zs_model.train_score_
    tl_model.init_ = zs_model.init_
    tl_model.loss_ = zs_model.loss_
    
    # Now increase n_estimators and continue training
    tl_model.set_params(n_estimators=curr_trees + add_trees)
    tl_model.fit(X_tl, y_tl.ravel())
    
    return tl_model


def _transfer_learn_linear(
    zs_model: LinearRegression,
    X_tl: np.ndarray,
    y_tl: np.ndarray,
    add_trees: int,  # Not used for linear, but kept for API consistency
    eta_ft: float,   # Not used for linear, but kept for API consistency
) -> LinearRegression:
    """
    For linear models, we simply retrain on TL data.
    Linear models don't have a concept of "continuing training" like tree models.
    """
    # Create a new linear model and fit on TL data
    tl_model = LinearRegression()
    tl_model.fit(X_tl, y_tl.ravel())
    return tl_model


def _clone_booster(booster: Booster) -> Booster:
    """Make an independent copy of an xgboost.Booster (to avoid in-place mutations)."""
    if xgb is None:
        raise ImportError("xgboost is required for booster cloning.")
    raw = booster.save_raw()
    b = xgb.Booster()
    b.load_model(raw)
    return b


def _train_ensemble_xgb(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_tl: Optional[np.ndarray],
    Y_tl: Optional[np.ndarray],
    seeds: list[int],
    model_params: Dict[str, object],
    add_trees: int,
    eta_ft: float,
    bootstrap: bool,
    bootstrap_frac: float,
    quantile_alpha: Optional[float] = None,
) -> list[XGBRegressor]:
    """
    Train an ensemble of XGBoost models with different seeds.
    
    If TL data is provided, each member continues from a cloned zero-shot booster.
    """
    if XGBRegressor is None:
        raise ImportError("xgboost is required for XGBoost ensemble training.")
    
    ensemble = []
    n_models = len(seeds)
    
    if X_tl is not None:
        # Step 1: Train zero-shot model as base for TL
        zs_params = dict(model_params)
        zs_params.setdefault("random_state", seeds[0])
        if quantile_alpha is not None:
            zs_params["quantile_alpha"] = quantile_alpha
        zs_model = XGBRegressor(**zs_params)
        zs_model.fit(X_train, Y_train.ravel())
        
        # Get zero-shot booster for TL continuation
        booster0 = zs_model.get_booster()
        curr_trees = zs_model.get_params().get("n_estimators")
        if curr_trees is None:
            curr_trees = model_params.get("n_estimators", 300)
        if curr_trees is None:
            curr_trees = 300
    else:
        booster0 = None
        curr_trees = None
    
    idx_all = np.arange(len(Y_train)) if X_tl is None else np.arange(len(Y_tl))
    
    for seed in seeds:
        if X_tl is None:
            # No TL: train each member independently on source data
            member_params = dict(model_params)
            member_params["random_state"] = seed
            if quantile_alpha is not None:
                member_params["quantile_alpha"] = quantile_alpha
            member = XGBRegressor(**member_params)
            member.fit(X_train, Y_train.ravel())
        else:
            # With TL: continue from cloned zero-shot booster
            # Optionally bootstrap TL data
            if bootstrap:
                rng = np.random.default_rng(seed)
                m = max(1, int(round(bootstrap_frac * len(Y_tl))))
                boot_idx = rng.choice(idx_all, size=m, replace=True)
                Xb = X_tl[boot_idx]
                yb = Y_tl[boot_idx]
            else:
                Xb = X_tl
                yb = Y_tl
            
            # Clone booster for independence
            b = _clone_booster(booster0)
            
            # Create member with continued training params
            member_params = {k: v for k, v in model_params.items() if k not in {"random_state", "learning_rate", "n_estimators"}}
            if quantile_alpha is not None:
                member_params["quantile_alpha"] = quantile_alpha
            member = XGBRegressor(
                **member_params,
                learning_rate=eta_ft,
                n_estimators=curr_trees + add_trees,
                random_state=seed,
            )
            member.fit(Xb, yb.ravel(), xgb_model=b)
        
        ensemble.append(member)
    
    return ensemble


def _train_ensemble_gbr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_tl: Optional[np.ndarray],
    Y_tl: Optional[np.ndarray],
    seeds: list[int],
    model_params: Dict[str, object],
    add_trees: int,
    eta_ft: float,
    bootstrap: bool,
    bootstrap_frac: float,
) -> list[GradientBoostingRegressor]:
    """
    Train an ensemble of GradientBoostingRegressor models with different seeds.
    """
    ensemble = []
    
    # Step 1: Train zero-shot model
    zs_params = dict(model_params)
    zs_params.setdefault("random_state", seeds[0])
    zs_model = GradientBoostingRegressor(**zs_params)
    zs_model.fit(X_train, Y_train.ravel())
    
    curr_trees = zs_model.get_params().get("n_estimators", 100)
    idx_all = np.arange(len(Y_train)) if X_tl is None else np.arange(len(Y_tl))
    
    for seed in seeds:
        if X_tl is None:
            # No TL: train each member independently
            member_params = dict(model_params)
            member_params["random_state"] = seed
            member = GradientBoostingRegressor(**member_params)
            member.fit(X_train, Y_train.ravel())
        else:
            # With TL: continue from zero-shot with warm_start
            if bootstrap:
                rng = np.random.default_rng(seed)
                m = max(1, int(round(bootstrap_frac * len(Y_tl))))
                boot_idx = rng.choice(idx_all, size=m, replace=True)
                Xb = X_tl[boot_idx]
                yb = Y_tl[boot_idx]
            else:
                Xb = X_tl
                yb = Y_tl
            
            # Create member with warm_start
            curr_params = zs_model.get_params()
            member = GradientBoostingRegressor(
                loss=curr_params.get("loss", "squared_error"),
                learning_rate=eta_ft,
                n_estimators=curr_trees,
                max_depth=curr_params.get("max_depth", 3),
                min_samples_split=curr_params.get("min_samples_split", 2),
                min_samples_leaf=curr_params.get("min_samples_leaf", 1),
                subsample=curr_params.get("subsample", 1.0),
                max_features=curr_params.get("max_features", None),
                random_state=seed,
                warm_start=True,
            )
            member.fit(Xb, yb.ravel())
            
            # Copy zero-shot estimators
            member.estimators_ = deepcopy(zs_model.estimators_)
            member.train_score_ = deepcopy(zs_model.train_score_)
            member.init_ = deepcopy(zs_model.init_)
            member.loss_ = deepcopy(zs_model.loss_)
            
            # Continue training
            member.set_params(n_estimators=curr_trees + add_trees)
            member.fit(Xb, yb.ravel())
        
        ensemble.append(member)
    
    return ensemble


def _train_ensemble_linear(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_tl: Optional[np.ndarray],
    Y_tl: Optional[np.ndarray],
    seeds: list[int],
    model_params: Dict[str, object],
    add_trees: int,  # Not used
    eta_ft: float,   # Not used
    bootstrap: bool,
    bootstrap_frac: float,
) -> list[LinearRegression]:
    """
    Train an ensemble of LinearRegression models with different seeds.
    Note: Linear models don't use seeds, but we train on different data if bootstrapped.
    """
    ensemble = []
    idx_all = np.arange(len(Y_train)) if X_tl is None else np.arange(len(Y_tl))
    
    for seed in seeds:
        if X_tl is None:
            if bootstrap:
                rng = np.random.default_rng(seed)
                m = max(1, int(round(bootstrap_frac * len(Y_train))))
                boot_idx = rng.choice(idx_all, size=m, replace=True)
                Xb = X_train[boot_idx]
                yb = Y_train[boot_idx]
            else:
                Xb = X_train
                yb = Y_train
        else:
            if bootstrap:
                rng = np.random.default_rng(seed)
                m = max(1, int(round(bootstrap_frac * len(Y_tl))))
                boot_idx = rng.choice(idx_all, size=m, replace=True)
                Xb = X_tl[boot_idx]
                yb = Y_tl[boot_idx]
            else:
                Xb = X_tl
                yb = Y_tl
        
        member = LinearRegression(**model_params)
        member.fit(Xb, yb.ravel())
        ensemble.append(member)
    
    return ensemble


def run_training(config: TrainingConfig) -> Dict[str, object]:
    """
    Train either a point regressor or quantile regressors and persist them to disk.
    """
    output_dir = config.output_dir
    if output_dir is None:
        raise ValueError("Training requires an output directory (set --output-dir).")
    ensure_dir(output_dir)
    training_dir = ensure_dir(output_dir / "training")

    X_train, Y_train = load_split(config.train_data)
    X_val = Y_val = None
    if config.val_data:
        X_val, Y_val = load_split(config.val_data)
    
    X_tl = Y_tl = None
    if config.tl_data:
        X_tl, Y_tl = load_split(config.tl_data)

    summary: Dict[str, object] = {
        "target": config.target,
        "train_data": config.train_data,
        "val_data": config.val_data,
        "tl_data": config.tl_data,
        "alpha": config.alpha,
        "output_dir": str(output_dir),
        "artifacts": {},
        "model_params": {},
    }

    def _save_mu_bands_for_ensemble(
        ensemble: list[object],
        X: Optional[np.ndarray],
        Y: Optional[np.ndarray],
        split_name: str,
        model_name: str,
    ) -> None:
        """
        Save mu-band (mean ± std across ensemble members) for a given split.
        Columns: model_name, upper_band, lower_band, prediction, true_value.
        """
        if X is None or Y is None:
            return
        if len(X) == 0:
            return
        # Compute predictions from each ensemble member
        preds_matrix = np.array([m.predict(X) for m in ensemble])  # (n_members, n_samples)
        mean_preds = preds_matrix.mean(axis=0)
        std_preds = preds_matrix.std(axis=0)
        upper = mean_preds + std_preds
        lower = mean_preds - std_preds
        mu_dir = ensure_dir(output_dir / "mu")
        save_mu_band_csv(
            mu_dir / f"{model_name}_mu_{split_name}.csv",
            model_name,
            Y,
            mean_preds,
            upper,
            lower,
        )

    if config.target == "regressor":
        summary["model"] = config.regressor_model
        regressor_params = dict(config.regressor_params or {})
        regressor_params.setdefault("random_state", config.seed)
        
        if config.mu_band and config.mu_seeds:
            # Ensemble training for mu-band
            if config.regressor_model == "xgb":
                if XGBRegressor is None:
                    raise ImportError("xgboost is required for XGBoost ensemble training.")
                ensemble = _train_ensemble_xgb(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, regressor_params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                )
            elif config.regressor_model == "gbr":
                ensemble = _train_ensemble_gbr(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, regressor_params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                )
            elif config.regressor_model == "linear":
                ensemble = _train_ensemble_linear(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, regressor_params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                )
            else:
                raise ValueError(f"Ensemble training not supported for model '{config.regressor_model}'.")
            
            # Save all ensemble members
            ensemble_dir = ensure_dir(output_dir / "ensemble")
            ensemble_paths = []
            for i, member in enumerate(ensemble):
                member_bundle = ModelBundle(
                    f"{config.regressor_model}_ensemble_{i}",
                    member,
                    lambda X_new, m=member: m.predict(X_new),
                )
                member_path = _dump_model(member_bundle, ensemble_dir, f"member_{i:03d}.joblib")
                ensemble_paths.append(member_path)
            
            summary["artifacts"]["regressor_ensemble"] = ensemble_paths
            summary["artifacts"]["regressor_ensemble_size"] = len(ensemble)
            summary["artifacts"]["regressor_ensemble_seeds"] = config.mu_seeds

            # Save mu-band CSVs for available splits (train/val/TL)
            _save_mu_bands_for_ensemble(ensemble, X_train, Y_train, "train", f"{config.regressor_model}_ensemble")
            if X_val is not None and Y_val is not None:
                _save_mu_bands_for_ensemble(ensemble, X_val, Y_val, "val", f"{config.regressor_model}_ensemble")
            if X_tl is not None and Y_tl is not None:
                _save_mu_bands_for_ensemble(ensemble, X_tl, Y_tl, "tl", f"{config.regressor_model}_ensemble")
        else:
            # Single model training (original logic)
            # Step 1: Train zero-shot model on source data
            zs_bundle = fit_regressor(
                X_train,
                Y_train,
                model_factory=_build_regressor_factory(config.regressor_model, regressor_params),
                model_name=config.regressor_model,
            )
            zs_path = _dump_model(zs_bundle, output_dir, f"{zs_bundle.name}_model.joblib")
            summary["artifacts"]["regressor_model"] = zs_path
            
            # Step 2: Fine-tune on TL data if provided
            if config.tl_data and X_tl is not None and len(X_tl) > 0:
                zs_model = zs_bundle.model
                if config.regressor_model == "xgb":
                    if XGBRegressor is None:
                        raise ImportError("xgboost is required for XGBoost transfer learning.")
                    tl_model = _transfer_learn_xgb(zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                elif config.regressor_model == "gbr":
                    tl_model = _transfer_learn_gbr(zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                elif config.regressor_model == "linear":
                    tl_model = _transfer_learn_linear(zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                else:
                    raise ValueError(f"Transfer learning not supported for model '{config.regressor_model}'.")
                
                # Save fine-tuned model
                tl_bundle = ModelBundle(f"{zs_bundle.name}_tl", tl_model, lambda X_new: tl_model.predict(X_new))
                tl_path = _dump_model(tl_bundle, output_dir, f"{zs_bundle.name}_tl_model.joblib")
                summary["artifacts"]["regressor_tl_model"] = tl_path
        
        params_for_summary = dict(regressor_params)
        params_for_summary.setdefault("seed", config.seed)
        if config.tl_data:
            params_for_summary["tl_add_trees"] = config.tl_add_trees
            params_for_summary["tl_eta"] = config.tl_eta
        if config.mu_band:
            params_for_summary["mu_seeds"] = config.mu_seeds
            params_for_summary["mu_bootstrap"] = config.mu_bootstrap
            params_for_summary["mu_bootstrap_frac"] = config.mu_bootstrap_frac
        summary["model_params"] = params_for_summary
    else:
        summary["model"] = config.quantile_model
        params = dict(config.quantile_params or {})
        params.setdefault("seed", config.seed)
        
        if config.mu_band and config.mu_seeds:
            # Ensemble training for mu-band (quantile models)
            q_low = config.alpha / 2
            q_high = 1 - config.alpha / 2
            
            if config.quantile_model == "xgb":
                if XGBRegressor is None:
                    raise ImportError("xgboost is required for XGBoost ensemble training.")
                # Train low and high quantile ensembles separately
                low_ensemble = _train_ensemble_xgb(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                    quantile_alpha=q_low,
                )
                high_ensemble = _train_ensemble_xgb(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                    quantile_alpha=q_high,
                )
            elif config.quantile_model == "gbr":
                # For GBR, we need to add alpha to params for low/high
                low_params = dict(params)
                low_params["alpha"] = q_low
                low_params["loss"] = "quantile"
                high_params = dict(params)
                high_params["alpha"] = q_high
                high_params["loss"] = "quantile"
                
                low_ensemble = _train_ensemble_gbr(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, low_params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                )
                high_ensemble = _train_ensemble_gbr(
                    X_train, Y_train, X_tl, Y_tl,
                    config.mu_seeds, high_params,
                    config.tl_add_trees, config.tl_eta,
                    config.mu_bootstrap, config.mu_bootstrap_frac,
                )
            else:
                raise ValueError(f"Ensemble training not supported for quantile model '{config.quantile_model}'.")
            
            # Save all ensemble members
            ensemble_dir = ensure_dir(output_dir / "ensemble")
            low_ensemble_paths = []
            high_ensemble_paths = []
            for i, (low_member, high_member) in enumerate(zip(low_ensemble, high_ensemble)):
                # Save low quantile member
                low_member_bundle = ModelBundle(
                    f"quantile_low_ensemble_{i}",
                    low_member,
                    lambda X_new, m=low_member: m.predict(X_new),
                )
                low_member_path = _dump_model(low_member_bundle, ensemble_dir, f"quantile_low_member_{i:03d}.joblib")
                low_ensemble_paths.append(low_member_path)
                
                # Save high quantile member
                high_member_bundle = ModelBundle(
                    f"quantile_high_ensemble_{i}",
                    high_member,
                    lambda X_new, m=high_member: m.predict(X_new),
                )
                high_member_path = _dump_model(high_member_bundle, ensemble_dir, f"quantile_high_member_{i:03d}.joblib")
                high_ensemble_paths.append(high_member_path)
            
            summary["artifacts"]["quantile_low_ensemble"] = low_ensemble_paths
            summary["artifacts"]["quantile_high_ensemble"] = high_ensemble_paths
            summary["artifacts"]["quantile_ensemble_size"] = len(low_ensemble)
            summary["artifacts"]["quantile_ensemble_seeds"] = config.mu_seeds

            # Save mu-band CSVs for available splits (mean of midpoint, std across midpoint)
            def _save_quantile_mu(
                X: Optional[np.ndarray],
                Y: Optional[np.ndarray],
                split_name: str,
            ) -> None:
                if X is None or Y is None or len(X) == 0:
                    return
                low_preds = np.array([m.predict(X) for m in low_ensemble])   # (n_members, n_samples)
                high_preds = np.array([m.predict(X) for m in high_ensemble]) # (n_members, n_samples)
                midpoint = 0.5 * (low_preds + high_preds)                    # (n_members, n_samples)
                mean_mid = midpoint.mean(axis=0)
                std_mid = midpoint.std(axis=0)
                upper = mean_mid + std_mid
                lower = mean_mid - std_mid
                mu_dir = ensure_dir(output_dir / "mu")
                save_mu_band_csv(
                    mu_dir / f"{config.quantile_model}_quantile_mu_{split_name}.csv",
                    f"{config.quantile_model}_quantile",
                    Y,
                    mean_mid,
                    upper,
                    lower,
                )

            _save_quantile_mu(X_train, Y_train, "train")
            if X_val is not None and Y_val is not None:
                _save_quantile_mu(X_val, Y_val, "val")
            if X_tl is not None and Y_tl is not None:
                _save_quantile_mu(X_tl, Y_tl, "tl")
        else:
            # Single model training (original logic)
            # Step 1: Train zero-shot quantile models on source data
            low_bundle, high_bundle = fit_quantile_regressors(
                X_train,
                Y_train,
                alpha_target=config.alpha,
                low_model_factory=_build_quantile_factory(config.quantile_model, config.quantile_params),
                high_model_factory=_build_quantile_factory(config.quantile_model, config.quantile_params),
            )
            low_path = _dump_model(low_bundle, output_dir, f"{low_bundle.name}_model.joblib")
            high_path = _dump_model(high_bundle, output_dir, f"{high_bundle.name}_model.joblib")
            summary["artifacts"]["quantile_low_model"] = low_path
            summary["artifacts"]["quantile_high_model"] = high_path
            
            # Step 2: Fine-tune on TL data if provided
            if config.tl_data and X_tl is not None and len(X_tl) > 0:
                low_zs_model = low_bundle.model
                high_zs_model = high_bundle.model
                
                if config.quantile_model == "xgb":
                    if XGBRegressor is None:
                        raise ImportError("xgboost is required for XGBoost transfer learning.")
                    low_tl_model = _transfer_learn_xgb(low_zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                    high_tl_model = _transfer_learn_xgb(high_zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                elif config.quantile_model == "gbr":
                    low_tl_model = _transfer_learn_gbr(low_zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                    high_tl_model = _transfer_learn_gbr(high_zs_model, X_tl, Y_tl, config.tl_add_trees, config.tl_eta)
                else:
                    raise ValueError(f"Transfer learning not supported for quantile model '{config.quantile_model}'.")
                
                # Save fine-tuned models
                low_tl_bundle = ModelBundle(f"{low_bundle.name}_tl", low_tl_model, lambda X_new: low_tl_model.predict(X_new))
                high_tl_bundle = ModelBundle(f"{high_bundle.name}_tl", high_tl_model, lambda X_new: high_tl_model.predict(X_new))
                low_tl_path = _dump_model(low_tl_bundle, output_dir, f"{low_bundle.name}_tl_model.joblib")
                high_tl_path = _dump_model(high_tl_bundle, output_dir, f"{high_bundle.name}_tl_model.joblib")
                summary["artifacts"]["quantile_low_tl_model"] = low_tl_path
                summary["artifacts"]["quantile_high_tl_model"] = high_tl_path
        
        if config.tl_data:
            params["tl_add_trees"] = config.tl_add_trees
            params["tl_eta"] = config.tl_eta
        if config.mu_band:
            params["mu_seeds"] = config.mu_seeds
            params["mu_bootstrap"] = config.mu_bootstrap
            params["mu_bootstrap_frac"] = config.mu_bootstrap_frac
        summary["model_params"] = params

    train_losses = _compute_mse_per_model(summary["artifacts"], config.target, X_train, Y_train)
    val_losses = (
        _compute_mse_per_model(summary["artifacts"], config.target, X_val, Y_val)
        if X_val is not None and len(X_val)
        else {}
    )
    
    # Compute TL losses if TL data is available
    tl_losses = {}
    if config.tl_data and X_tl is not None and len(X_tl) > 0:
        tl_losses = _compute_mse_per_model(summary["artifacts"], config.target, X_tl, Y_tl)
    
    summary["train_mse"] = train_losses
    summary["val_mse"] = val_losses if val_losses else None
    summary["tl_mse"] = tl_losses if tl_losses else None

    for model_name, loss_value in train_losses.items():
        val_loss = val_losses.get(model_name) if summary["val_mse"] else None
        _plot_losses(training_dir / f"{model_name}_loss.png", model_name, loss_value, val_loss)
    
    # Plot TL losses for TL models
    if tl_losses:
        for model_name, loss_value in tl_losses.items():
            if "_tl" in model_name:  # Only plot TL models
                _plot_losses(training_dir / f"{model_name}_loss.png", model_name, loss_value, None)

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def _dump_model(bundle: ModelBundle, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    joblib.dump(bundle.model, path)
    return str(path)


def _build_regressor_factory(name: str, params: Dict[str, object]) -> Callable[[], object]:
    params = dict(params or {})
    if name == "linear":
        return lambda: LinearRegression(**params)
    if name == "gbr":
        defaults = {"loss": "squared_error"}
        defaults.update(params)
        return lambda: GradientBoostingRegressor(**defaults)
    if name == "xgb":
        if XGBRegressor is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is not installed. Install it via `pip install xgboost` to use --regressor-model xgb."
            )
        user_params = dict(params)
        eta = user_params.pop("eta", None)
        if eta is not None and "learning_rate" not in user_params:
            user_params["learning_rate"] = eta
        defaults = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "auto",
        }
        defaults.update(user_params)
        return lambda: XGBRegressor(**defaults)
    raise ValueError(f"Unsupported regressor model '{name}'.")


def _build_quantile_factory(
    name: str, params: Dict[str, object]
) -> Callable[[float], object]:
    params = dict(params or {})
    if name == "gbr":
        user_params = {k: v for k, v in params.items() if k not in {"loss", "alpha"}}
        seed = user_params.pop("seed", None)
        if seed is not None and "random_state" not in user_params:
            user_params["random_state"] = seed

        def factory(alpha: float) -> GradientBoostingRegressor:
            cfg = {"loss": "quantile", "alpha": alpha}
            cfg.update(user_params)
            return GradientBoostingRegressor(**cfg)

        return factory

    if name == "xgb":
        if XGBRegressor is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is not installed. Install it via `pip install xgboost` to use --quantile-model xgb."
            )
        user_params = dict(params)
        eta = user_params.pop("eta", None)
        if eta is not None and "learning_rate" not in user_params:
            user_params["learning_rate"] = eta
        seed = user_params.pop("seed", None)
        if seed is not None and "random_state" not in user_params:
            user_params["random_state"] = seed
        defaults = {
            "objective": "reg:quantileerror",
            "learning_rate": 0.1,
            "n_estimators": 300,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "auto",
        }
        defaults.update(user_params)

        def factory(alpha: float) -> XGBRegressor:
            cfg = dict(defaults)
            cfg["quantile_alpha"] = alpha
            return XGBRegressor(**cfg)

        return factory

    raise ValueError(f"Unsupported quantile model '{name}'.")


def _compute_mse_per_model(
    artifacts: Dict[str, str],
    target: TrainingTarget,
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    if X is None or Y is None or len(X) == 0:
        return {}
    results: Dict[str, float] = {}
    if target == "regressor":
        if "regressor_ensemble" in artifacts:
            # Load ensemble and compute mean predictions
            ensemble_paths = artifacts["regressor_ensemble"]
            ensemble_models = [joblib.load(p) for p in ensemble_paths]
            predictions = np.array([model.predict(X) for model in ensemble_models])
            preds = predictions.mean(axis=0)
            results["regressor_ensemble"] = float(mean_squared_error(Y, preds))
        elif "regressor_model" in artifacts:
            model = joblib.load(artifacts["regressor_model"])
            preds = model.predict(X)
            results["regressor_model"] = float(mean_squared_error(Y, preds))
        return results

    # Handle quantile models (single or ensemble)
    if "quantile_low_ensemble" in artifacts and "quantile_high_ensemble" in artifacts:
        # Load ensembles and compute mean predictions
        low_ensemble_paths = artifacts["quantile_low_ensemble"]
        high_ensemble_paths = artifacts["quantile_high_ensemble"]
        low_ensemble_models = [joblib.load(p) for p in low_ensemble_paths]
        high_ensemble_models = [joblib.load(p) for p in high_ensemble_paths]
        
        # Compute mean predictions from ensembles
        low_predictions = np.array([model.predict(X) for model in low_ensemble_models])
        high_predictions = np.array([model.predict(X) for model in high_ensemble_models])
        low_preds = low_predictions.mean(axis=0)
        high_preds = high_predictions.mean(axis=0)
        
        results["quantile_low_ensemble"] = float(mean_squared_error(Y, low_preds))
        results["quantile_high_ensemble"] = float(mean_squared_error(Y, high_preds))
    elif "quantile_low_model" in artifacts and "quantile_high_model" in artifacts:
        low_model = joblib.load(artifacts["quantile_low_model"])
        high_model = joblib.load(artifacts["quantile_high_model"])
        results["quantile_low_model"] = float(mean_squared_error(Y, low_model.predict(X)))
        results["quantile_high_model"] = float(mean_squared_error(Y, high_model.predict(X)))
    
    return results


def _plot_losses(path: Path, model_name: str, train_mse: float, val_mse: Optional[float]) -> None:
    labels = ["train"]
    values = [train_mse]
    colors = ["tab:blue"]
    if val_mse is not None:
        labels.append("val")
        values.append(val_mse)
        colors.append("tab:orange")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("MSE")
    ax.set_title(f"{model_name} loss")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

