"""Metrics and cluster-level bootstrap.

``average precision`` is the primary metric because positives are rare (4.55%
of pairs in the shipped dataset) and because it is what the regulatory-genomics
benchmarks in this project report.  Every headline number is accompanied by a
*cluster-level* bootstrap interval: pairs inside one gene are correlated (4.9
pairs per gene), so resampling rows would understate the error.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(y), np.asarray(score)))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score)))


def bootstrap_ap_ci(
    y: np.ndarray,
    score: np.ndarray,
    cluster_id: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, float]:
    """Percentile CI for AP, resampling **clusters** (genes) with replacement.

    Resamples that end up with a single class are skipped; the number of usable
    resamples is reported so a silently degenerate interval is visible.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    cluster_id = np.asarray(cluster_id)

    _, inv = np.unique(cluster_id, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(n_clusters + 1))
    rows_by_cluster: List[np.ndarray] = [order[bounds[i]:bounds[i + 1]] for i in range(n_clusters)]

    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([rows_by_cluster[i] for i in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        aps.append(average_precision(yy, score[rows]))

    if len(aps) < 0.5 * n_boot:
        raise RuntimeError(
            f"only {len(aps)}/{n_boot} bootstrap resamples had both classes; "
            "the positives are too sparse for a stable interval"
        )
    aps = np.asarray(aps)
    return {
        "point": average_precision(y, score),
        "ci_low": float(np.percentile(aps, 100 * alpha / 2)),
        "ci_high": float(np.percentile(aps, 100 * (1 - alpha / 2))),
        "n_valid": int(len(aps)),
    }
