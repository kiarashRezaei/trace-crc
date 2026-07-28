"""
Grouping strategies used to approximate conditional guarantees.

Each function returns a list of NumPy index arrays referencing calibration
samples that belong to the group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def quantile_bins(X: np.ndarray, K: int) -> List[np.ndarray]:
    """
    Partition the scalar covariate X into K bins with roughly equal counts.
    """
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Quantile bins expect a single feature.")
    edges = np.quantile(X.ravel(), np.linspace(0, 1, K + 1))
    bins = []
    for k in range(K):
        left, right = edges[k], edges[k + 1]
        mask = (X.ravel() >= left) & (X.ravel() <= right if k == K - 1 else X.ravel() < right)
        bins.append(np.where(mask)[0])
    return bins


def adaptive_binning(X: np.ndarray, min_points: int = 100) -> List[np.ndarray]:
    """
    Start with quantile bins and iteratively merge sparse ones to keep sample
    sizes above `min_points`.
    """
    K = max(2, len(X) // min_points)
    bins = quantile_bins(X, max(K, 2))
    merged = []
    buffer = np.array([], dtype=int)
    for idx in bins:
        buffer = np.concatenate([buffer, idx])
        if len(buffer) >= min_points:
            merged.append(buffer)
            buffer = np.array([], dtype=int)
    if buffer.size > 0:
        if merged:
            merged[-1] = np.concatenate([merged[-1], buffer])
        else:
            merged.append(buffer)
    return merged


def tree_based_groups(
    X: np.ndarray,
    y: np.ndarray,
    max_leaf_nodes: int = 5,
    min_samples_leaf: int = 100,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Fit a shallow decision tree to identify regions with different residual
    behavior. Leaves correspond to groups.
    """
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    tree.fit(X, y)
    leaves = tree.apply(X)
    bins = []
    for leaf_id in np.unique(leaves):
        bins.append(np.where(leaves == leaf_id)[0])
    return bins


def kmeans_groups(X: np.ndarray, n_clusters: int = 5, min_size: int = 50, seed: int = 0) -> List[np.ndarray]:
    """
    Use 1D or multidimensional clustering to define groups. Small clusters are
    merged into their nearest neighbor.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = km.fit_predict(X)
    bins = [np.where(labels == c)[0] for c in np.unique(labels)]
    # Merge tiny clusters with the closest centroid
    centroids = km.cluster_centers_
    final_bins: List[np.ndarray] = []
    for idx, centroid in zip(bins, centroids):
        if len(idx) >= min_size or not final_bins:
            final_bins.append(idx)
            continue
        # merge with previous bin by default
        final_bins[-1] = np.concatenate([final_bins[-1], idx])
    return final_bins


def tree_groups_multi(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    max_leaf_nodes: int = 5,
    min_samples_leaf: int = 100,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Multi-dimensional tree-based grouping for calibration and test sets.

    A single DecisionTreeRegressor is fitted on (X_cal, y_cal) and then applied
    to both X_cal and X_test. For each leaf, we return the indices of cal/test
    samples that fall into that leaf.
    """
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    tree.fit(X_cal, y_cal)
    leaves_cal = tree.apply(X_cal)
    leaves_test = tree.apply(X_test)

    leaf_ids = np.unique(leaves_cal)
    cal_groups: List[np.ndarray] = []
    test_groups: List[np.ndarray] = []
    for leaf_id in leaf_ids:
        cal_idx = np.where(leaves_cal == leaf_id)[0]
        test_idx = np.where(leaves_test == leaf_id)[0]
        cal_groups.append(cal_idx)
        test_groups.append(test_idx)
    return cal_groups, test_groups


def kmeans_groups_multi(
    X_cal: np.ndarray,
    X_test: np.ndarray,
    n_clusters: int = 5,
    min_size: int = 50,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Multi-dimensional k-means grouping for calibration and test sets.

    KMeans is fitted on X_cal only, and both cal/test samples are assigned to
    the resulting clusters. Very small clusters (in the calibration set) are
    merged with their predecessor, and test indices are merged accordingly.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels_cal = km.fit_predict(X_cal)
    labels_test = km.predict(X_test)

    unique_labels = np.unique(labels_cal)
    cal_bins: List[np.ndarray] = []
    test_bins: List[np.ndarray] = []
    for c in unique_labels:
        cal_bins.append(np.where(labels_cal == c)[0])
        test_bins.append(np.where(labels_test == c)[0])

    final_cal: List[np.ndarray] = []
    final_test: List[np.ndarray] = []
    for cal_idx, test_idx in zip(cal_bins, test_bins):
        if len(cal_idx) >= min_size or not final_cal:
            final_cal.append(cal_idx)
            final_test.append(test_idx)
            continue
        # merge with previous bin by default
        final_cal[-1] = np.concatenate([final_cal[-1], cal_idx])
        final_test[-1] = np.concatenate([final_test[-1], test_idx])

    return final_cal, final_test

