"""Metrics, cluster-level bootstrap, and operating-point tables.

``average precision`` is the primary metric because positives are rare (4.55%
of pairs in the shipped dataset) and because it is what the regulatory-genomics
benchmarks in this project report.  Every headline number is accompanied by a
*cluster-level* bootstrap interval: pairs inside one gene are correlated (4.9
pairs per gene), so resampling rows would understate the error.

Ranking metrics are threshold-free; a deployment is not.  ``threshold_table``
turns a probability into the quantities a call-set user needs (how many pairs
are flagged, how many of those are true, what recall is given up), and
``calibration`` checks whether "p >= 0.90" can be read as "90% of these calls
are real".  Both are reported for every arm, so the operating point is part of
the result and not an afterthought.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

#: Thresholds reported for every arm.  They span "call most of the obvious
#: positives" (0.50) to "only the very confident tail" (0.99); 0.90/0.95 are the
#: two operating points used in the repository's documentation.
DEFAULT_THRESHOLDS: Sequence[float] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)


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


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio with a defined-at-zero convention (0.0 when the denominator is 0)."""
    return float(numerator / denominator) if denominator else 0.0


def threshold_table(
    y: np.ndarray,
    score: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> List[Dict[str, float]]:
    """Hard-classification metrics for each ``score >= threshold`` call set.

    A pair is called positive iff its predicted probability is at least the
    threshold, so this is the table a user of the model actually faces.  Each row
    reports the confusion matrix (``tp/fp/fn/tn``), ``flagged`` (the size of the
    call set), precision, recall, specificity, F1, MCC and ``lift`` (precision
    divided by the base positive rate).  Precision, F1, MCC and lift are 0.0
    when undefined (no calls, or a degenerate one-class table); ``flagged`` tells
    the two cases apart.

    ``specificity`` is the true-negative rate, i.e. the share of all negatives
    that the call set correctly leaves alone; it matters here because 95% of the
    pairs are negatives and a "call everything" rule would look perfect on
    recall alone.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError(f"y and score disagree: {y.shape} vs {score.shape}")

    n = int(len(y))
    pos = int(y.sum())
    neg = n - pos
    prevalence = _ratio(pos, n)

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        call = score >= threshold
        tp = int(np.sum(call & y))
        fp = int(np.sum(call & ~y))
        fn = pos - tp
        tn = neg - fp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, pos)
        denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rows.append(
            {
                "threshold": threshold,
                "flagged": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "specificity": _ratio(tn, neg),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn),
                "mcc": _ratio(tp * tn - fp * fn, denominator),
                "lift": _ratio(precision, prevalence),
            }
        )
    return rows


def calibration(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """Brier score, ECE and the reliability bins behind them.

    Equal-width bins over [0, 1].  ``ece`` is the sample-weighted mean absolute
    gap between predicted probability and observed frequency; it is the number
    that justifies (or refutes) reading a probability as a confidence.  Bins with
    no rows are omitted rather than reported as empty gaps.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape:
        raise ValueError(f"y and score disagree: {y.shape} vs {score.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # include the right edge in the last bin so score == 1.0 is counted
    idx = np.clip(np.digitize(score, edges[1:-1], right=False), 0, n_bins - 1)
    bins: List[Dict[str, float]] = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mean_pred = float(score[m].mean())
        frac_pos = float(y[m].mean())
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
            }
        )
        ece += float(m.sum()) / len(y) * abs(frac_pos - mean_pred)

    return {
        "brier": float(np.mean((score - y) ** 2)),
        "ece": float(ece),
        "n_bins": int(n_bins),
        "bins": bins,
    }


def operating_point(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    """One row of :func:`threshold_table`, for callers that need a single cut."""
    return threshold_table(y, score, thresholds=(float(threshold),))[0]


def paired_bootstrap_delta_ap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cluster_id: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, float]:
    """Paired percentile CI for AP(score_a) - AP(score_b), resampling clusters."""
    y = np.asarray(y)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    cluster_id = np.asarray(cluster_id)

    _, inv = np.unique(cluster_id, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(n_clusters + 1))
    rows_by_cluster: List[np.ndarray] = [order[bounds[i]:bounds[i + 1]] for i in range(n_clusters)]

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([rows_by_cluster[i] for i in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        deltas.append(average_precision(yy, score_a[rows]) - average_precision(yy, score_b[rows]))

    if len(deltas) < 0.5 * n_boot:
        raise RuntimeError(
            f"only {len(deltas)}/{n_boot} paired bootstrap resamples had both classes"
        )
    deltas_arr = np.asarray(deltas)
    return {
        "delta": float(average_precision(y, score_a) - average_precision(y, score_b)),
        "ci_low": float(np.percentile(deltas_arr, 100 * alpha / 2)),
        "ci_high": float(np.percentile(deltas_arr, 100 * (1 - alpha / 2))),
        "p_gt0": float(np.mean(deltas_arr > 0.0)),
        "n_valid": int(len(deltas_arr)),
    }
