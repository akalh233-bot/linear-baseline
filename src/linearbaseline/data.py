"""Data: synthetic generator for smoke tests, loader for real npz bundles, splits.

Real-data contract (``.npz``), all arrays share the first axis:

==============  ==========================  ==================================
key             dtype                       meaning
==============  ==========================  ==================================
``seq``         uint8 ``(N, L)``            A=0, C=1, G=2, T=3
``y``           int8 ``(N,)``               binary label (1 = functional)
``locus_id``    ``<U*`` or int64 ``(N,)``   genomic locus; the unit of splitting
``tf_id``       ``<U*`` ``(N,)``            optional, used as a context feature
``cell_state``  ``<U*`` ``(N,)``            optional, used as a context feature
``distance_bp`` float32 ``(N,)``            optional, CRE -> TSS distance
``atac``        float32 ``(N,)``            optional, accessibility summary
==============  ==========================  ==================================

Rows that belong to one locus must never be split across train and test: the
whole point of this baseline is to be honest about cross-locus generalisation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

ALPHABET = "ACGT"


# --------------------------------------------------------------------------
# synthetic world
# --------------------------------------------------------------------------

def _random_pwms(rng: np.random.Generator, n_motifs: int, length: int) -> np.ndarray:
    """Small log-odds bank; not JASPAR, just enough structure to learn."""
    return rng.uniform(-1.6, 1.6, size=(n_motifs, length, 4)).astype(np.float32)


def motif_strength(seq: np.ndarray, pwms: np.ndarray) -> np.ndarray:
    """Max log-odds score of any motif at any position, per sequence."""
    n, seq_len = seq.shape
    k = pwms.shape[1]
    best = np.full(n, -np.inf, dtype=np.float32)
    for pwm in pwms:
        scores = np.zeros((n, seq_len - k + 1), dtype=np.float32)
        for j in range(k):
            scores += pwm[j][seq[:, j:seq_len - k + 1 + j]]
        best = np.maximum(best, scores.max(axis=1))
    return best


def make_synthetic(
    n_loci: int = 4000,
    rows_per_locus: int = 2,
    seq_len: int = 200,
    n_motifs: int = 6,
    n_tfs: int = 8,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """A toy world where binding depends on motif dosage, accessibility and distance.

    Each locus carries 1-9 planted motifs, which gives a continuous *dosage* the
    model has to infer from 3-mer content.  The label is

    ``z = 2.8 * dosage + 0.9 * (atac - 0.4) - 0.7 * log10(d / 1e4) + 0.5 * tf_effect``

    with ``y ~ Bernoulli(sigmoid(z - 1.3))``.  Prevalence lands around 30-40 %
    and a 3-mer linear model reaches AP around 0.65, so a correct pipeline shows
    a real but not suspicious signal.
    """
    rng = np.random.default_rng(seed)
    pwms = _random_pwms(rng, n_motifs, 8)

    locus_seq = rng.integers(0, 4, size=(n_loci, seq_len), dtype=np.uint8)
    n_planted = rng.integers(1, 10, size=n_loci)          # motif dosage 1..9
    for i in range(n_loci):
        for _ in range(n_planted[i]):
            pwm = pwms[rng.integers(0, n_motifs)]
            pos = rng.integers(0, seq_len - pwm.shape[0] + 1)
            for j in range(pwm.shape[0]):
                locus_seq[i, pos + j] = int(np.argmax(pwm[j]))

    tf_pool = np.array([f"TF{i:02d}" for i in range(n_tfs)])
    tf_effect = rng.normal(0.0, 0.6, size=n_tfs).astype(np.float32)

    locus_id = np.repeat(np.arange(n_loci), rows_per_locus)
    seq = np.repeat(locus_seq, rows_per_locus, axis=0)
    tf_idx = rng.integers(0, n_tfs, size=n_loci)
    tf_id = np.repeat(tf_pool[tf_idx], rows_per_locus)
    cell_state = np.repeat(rng.choice(["K562", "GM12878", "HepG2"], size=n_loci), rows_per_locus)
    distance_bp = np.exp(rng.normal(np.log(1.2e4), 1.1, size=len(locus_id))).astype(np.float32)
    atac = rng.beta(2.0, 3.0, size=len(locus_id)).astype(np.float32)

    dosage = (n_planted[locus_id] - 5.0) / 2.0
    z = (
        2.8 * dosage
        + 0.9 * (atac - 0.4)
        - 0.7 * np.log10(distance_bp / 1e4)
        + 0.5 * tf_effect[tf_idx][locus_id]
    )
    p = 1.0 / (1.0 + np.exp(-(z - 1.3)))
    y = (rng.random(len(p)) < p).astype(np.int8)

    return {
        "seq": seq,
        "y": y,
        "locus_id": locus_id,
        "tf_id": tf_id,
        "cell_state": cell_state,
        "distance_bp": distance_bp,
        "atac": atac,
        "pwms": pwms,
    }


# --------------------------------------------------------------------------
# real bundles
# --------------------------------------------------------------------------

def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load an npz bundle. ``allow_pickle`` is needed for string ``locus_id``."""
    with np.load(path, allow_pickle=True) as bundle:
        data = {k: bundle[k] for k in bundle.files}
    required = ("y", "locus_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    if "seq" not in data:
        raise ValueError(f"{path} is missing 'seq'; k-mer features need it")
    return data


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def label_balanced_folds(groups: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Group folds assigned round-robin after sorting groups by positive rate.

    One fold per group, so no group can straddle folds.  ``StratifiedGroupKFold``
    does the same thing but, at 5e4 groups, costs ~11 s to build; this is O(n log n)
    and takes milliseconds (see ``docs/DESIGN.md``).
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    _, g_idx = np.unique(groups, return_inverse=True)
    n_g = int(g_idx.max()) + 1
    if n_g < n_splits:
        raise ValueError(f"need >= {n_splits} groups, got {n_g}")

    pos = np.bincount(g_idx, weights=(y == 1).astype(np.float64), minlength=n_g)
    cnt = np.bincount(g_idx, minlength=n_g).astype(np.float64)
    rate = pos / np.maximum(cnt, 1.0)

    order = np.argsort(rate, kind="stable")
    fold_of_g = np.empty(n_g, dtype=np.int64)
    fold_of_g[order] = np.arange(n_g) % n_splits

    row_fold = fold_of_g[g_idx]
    return [(np.where(row_fold != k)[0], np.where(row_fold == k)[0]) for k in range(n_splits)]


def train_test_split_by_locus(
    locus_id: np.ndarray, test_frac: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Split rows by locus, never inside one."""
    groups = np.unique(np.asarray(locus_id))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_frac)))
    test_groups = set(groups[:n_test].tolist())
    is_test = np.array([g in test_groups for g in np.asarray(locus_id)])
    return np.where(~is_test)[0], np.where(is_test)[0]

