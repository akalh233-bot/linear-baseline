"""Metrics and locus-level bootstrap.

``average precision`` is the primary metric because positives are rare and
because it is what the regulatory-genomics benchmarks in this project report.
Every headline number is accompanied by a *locus-level* bootstrap interval: rows
inside one locus are correlated, so resampling rows would understate the error.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss as sk_log_loss,
    roc_auc_score,
)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(y), np.asarray(score)))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score)))


def log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(sk_log_loss(np.asarray(y), p, labels=[0, 1]))


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y), np.asarray(prob)))


def bootstrap_ap_ci(
    y: np.ndarray,
    score: np.ndarray,
    locus_id: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, float]:
    """Percentile CI for AP, resampling **loci** with replacement.

    Loci that end up with a single class are skipped; the number of usable
    resamples is reported so a silently degenerate interval is visible.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    locus_id = np.asarray(locus_id)

    _, inv = np.unique(locus_id, return_inverse=True)
    n_loci = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(n_loci + 1))
    rows_by_locus: List[np.ndarray] = [order[bounds[i]:bounds[i + 1]] for i in range(n_loci)]

    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_loci, size=n_loci)
        rows = np.concatenate([rows_by_locus[i] for i in pick])
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
