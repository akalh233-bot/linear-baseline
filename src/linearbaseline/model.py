"""Logistic regression + grouped cross-validation.

The model is a ``StandardScaler`` followed by L2-penalised logistic regression.
That is deliberately the weakest reasonable learner: if a sequence model cannot
beat it on held-out loci, the sequence model is not adding regulatory knowledge.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import label_balanced_folds
from .metrics import average_precision


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
    class_weight: str | None = None,
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def select_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    Cs: Sequence[float],
    n_splits: int = 5,
    max_iter: int = 2000,
    verbose: bool = False,
) -> Tuple[float, Dict[float, float]]:
    """Pick ``C`` by mean grouped-CV AP. Groups (loci) never straddle folds."""
    folds = label_balanced_folds(groups, y, n_splits=n_splits)
    scores: Dict[float, float] = {}

    for C in Cs:
        aps: List[float] = []
        for train_idx, valid_idx in folds:
            if np.unique(y[train_idx]).size < 2:
                continue
            model = fit_logistic(X[train_idx], y[train_idx], C=C, max_iter=max_iter)
            prob = model.predict_proba(X[valid_idx])[:, 1]
            aps.append(average_precision(y[valid_idx], prob))
        scores[float(C)] = float(np.mean(aps)) if aps else float("nan")
        if verbose:
            print(f"  C={C:<8g} cv_ap={scores[float(C)]:.4f}")

    usable = {c: s for c, s in scores.items() if np.isfinite(s)}
    if not usable:
        raise RuntimeError("every C produced a degenerate CV fold")
    best = max(usable, key=usable.get)
    return float(best), scores


def top_coefficients(model: Pipeline, names: Sequence[str], k: int = 20) -> List[Tuple[str, float]]:
    coefs = model.named_steps["lr"].coef_.ravel()
    order = np.argsort(-np.abs(coefs))[:k]
    return [(names[i], float(coefs[i])) for i in order]
