"""
Shared helpers for loading/saving datasets and ensuring directories exist.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from .models import ModelBundle


def load_split(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "X" not in data or "Y" not in data:
        raise ValueError(f"{path} must contain 'X' and 'Y' arrays.")
    return np.asarray(data["X"]), np.asarray(data["Y"])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_ensemble_members(ensemble_dir: Path, fallback_name: str) -> list[Path]:
    """
    Find ensemble member files in a directory.
    
    Args:
        ensemble_dir: Directory containing ensemble member files
        fallback_name: Name to use for pattern matching
    
    Returns:
        List of paths to ensemble member files
    """
    patterns = [
        f"quantile_{fallback_name.split('_')[-1]}_member_*.joblib",  # quantile_low_member_*.joblib
        f"{fallback_name}_member_*.joblib",  # quantile_low_member_*.joblib (if fallback_name is full)
        "member_*.joblib",  # Generic member_*.joblib
    ]
    
    member_files = []
    for pattern in patterns:
        member_files = sorted(glob.glob(str(ensemble_dir / pattern)))
        if member_files:
            break
    
    if not member_files:
        raise ValueError(f"No ensemble members found in {ensemble_dir} with patterns {patterns}")
    
    return [Path(f) for f in member_files]


def load_ensemble_bundle(ensemble_dir: Path, fallback_name: str) -> ModelBundle:
    """
    Load an ensemble of models and create a ModelBundle that predicts the mean.
    
    Args:
        ensemble_dir: Directory containing ensemble member files (member_*.joblib or quantile_*_member_*.joblib)
        fallback_name: Name to use if ensemble type cannot be determined
    
    Returns:
        ModelBundle with ensemble mean predictor
    """
    member_files = _find_ensemble_members(ensemble_dir, fallback_name)
    
    # Load all ensemble members
    ensemble_models = []
    for member_file in member_files:
        model = joblib.load(member_file)
        ensemble_models.append(model)
    
    # Create ensemble predictor that computes mean
    def ensemble_predict(X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in ensemble_models])
        return predictions.mean(axis=0)
    
    name = f"{fallback_name}_ensemble"
    # Store first model as representative (for compatibility)
    return ModelBundle(name=name, model=ensemble_models[0], predict=ensemble_predict)


def compute_ensemble_predictions(ensemble_dir: Path, fallback_name: str, X: np.ndarray) -> np.ndarray:
    """
    Compute predictions from all ensemble members.
    
    Args:
        ensemble_dir: Directory containing ensemble member files
        fallback_name: Name to use for pattern matching
        X: Input features
    
    Returns:
        Array of shape (n_members, n_samples) with predictions from each member
    """
    member_files = _find_ensemble_members(ensemble_dir, fallback_name)
    
    predictions = []
    for member_file in member_files:
        model = joblib.load(member_file)
        pred = model.predict(X)
        predictions.append(pred)
    
    return np.array(predictions)


def load_model_bundle(path: str, fallback_name: str) -> ModelBundle:
    """
    Load a model bundle, automatically detecting if it's an ensemble.
    
    Detection logic:
    1. If path is a directory with 'ensemble' subdirectory -> load ensemble
    2. If path is a directory with training_summary.json indicating ensembles -> load ensemble
    3. If path is a directory, try to find model file in it
    4. Otherwise, load as single model file
    
    Args:
        path: Path to model file, training directory, or ensemble directory
        fallback_name: Name to use for the bundle (e.g., "quantile_low", "quantile_high")
    
    Returns:
        ModelBundle with single model or ensemble mean predictor
    """
    path_obj = Path(path)
    
    # Case 1: Path is a directory - check for ensemble
    if path_obj.is_dir():
        # First check for ensemble subdirectory
        ensemble_dir = path_obj / "ensemble"
        if ensemble_dir.exists() and ensemble_dir.is_dir():
            return load_ensemble_bundle(ensemble_dir, fallback_name)
        
        # Check training_summary.json for ensemble info
        summary_path = path_obj / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            artifacts = summary.get("artifacts", {})
            
            # Determine which ensemble to load based on fallback_name
            ensemble_paths = None
            if "low" in fallback_name.lower() and "quantile_low_ensemble" in artifacts:
                ensemble_paths = artifacts["quantile_low_ensemble"]
            elif "high" in fallback_name.lower() and "quantile_high_ensemble" in artifacts:
                ensemble_paths = artifacts["quantile_high_ensemble"]
            elif "regressor_ensemble" in artifacts:
                ensemble_paths = artifacts["regressor_ensemble"]
            
            if ensemble_paths:
                # Load ensemble from first path's directory
                first_path = Path(ensemble_paths[0])
                ensemble_dir = first_path.parent
                return load_ensemble_bundle(ensemble_dir, fallback_name)
            
            # Fallback: try to find single model in artifacts
            model_key = None
            if "low" in fallback_name.lower() and "quantile_low_model" in artifacts:
                model_key = "quantile_low_model"
            elif "high" in fallback_name.lower() and "quantile_high_model" in artifacts:
                model_key = "quantile_high_model"
            elif "regressor_model" in artifacts:
                model_key = "regressor_model"
            
            if model_key:
                model_path_str = artifacts[model_key]
                # If relative path, resolve relative to summary directory
                model_path = Path(model_path_str)
                if not model_path.is_absolute():
                    model_path = path_obj / model_path.name
                if model_path.exists():
                    path = str(model_path)
        
        # If still a directory and no model found, try common model filenames
        if path_obj.is_dir():
            common_names = [
                f"{fallback_name}_model.joblib",
                f"{fallback_name}.joblib",
                "model.joblib",
            ]
            for name in common_names:
                candidate = path_obj / name
                if candidate.exists():
                    path = str(candidate)
                    break
    
    # Load single model (or fall through if path is a file)
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model = joblib.load(path)
    name = Path(path).stem or fallback_name
    return ModelBundle(name=name, model=model, predict=lambda X: model.predict(X))

