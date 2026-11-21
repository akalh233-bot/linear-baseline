"""Feature builders. Everything here is linear-model friendly and cheap.

Two rules:

* features are built from sequence plus (optionally) covariates, never from the
  label;
* every feature group has a name, so coefficients stay interpretable and
  ablations can drop whole groups.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

ALPHABET = "ACGT"


def _kmer_names(k: int) -> List[str]:
    names = [""]
    for _ in range(k):
        names = [prefix + base for prefix in names for base in ALPHABET]
    return [f"kmer:{name}" for name in names]


def kmer_features(seq: np.ndarray, k: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Normalised k-mer frequencies, shape ``(N, 4**k)``."""
    seq = np.asarray(seq)
    n, length = seq.shape
    if length < k:
        raise ValueError(f"sequence length {length} < k={k}")

    n_windows = length - k + 1
    # encode each window as a base-4 integer
    codes = np.zeros((n, n_windows), dtype=np.int64)
    for j in range(k):
        codes = codes * 4 + seq[:, j:n_windows + j]

    n_kmers = 4 ** k
    counts = np.zeros((n, n_kmers), dtype=np.float32)
    rows = np.repeat(np.arange(n), n_windows)
    np.add.at(counts, (rows, codes.ravel()), 1.0)
    counts /= float(n_windows)
    return counts, _kmer_names(k)


def distance_features(distance_bp: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Three smooth transforms of CRE-TSS distance, clipped to sane bounds."""
    d = np.clip(np.asarray(distance_bp, dtype=np.float64), 1.0, 5e6)
    feats = np.column_stack(
        [
            np.log10(d + 1.0),
            1.0 / (1.0 + d / 1e4),
            (d < 1e4).astype(np.float64),
            (d > 1e5).astype(np.float64),
        ]
    )
    return feats.astype(np.float32), ["dist:log10", "dist:inv10kb", "dist:lt10kb", "dist:gt100kb"]


def atac_features(atac: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    a = np.asarray(atac, dtype=np.float64)
    return a.reshape(-1, 1).astype(np.float32), ["atac:raw"]


def context_features(values: np.ndarray, prefix: str) -> Tuple[np.ndarray, List[str]]:
    """One-hot encoding over the observed vocabulary (sorted for determinism)."""
    values = np.asarray(values).astype(str)
    vocab = np.unique(values)
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    out[np.arange(len(values)), [index[v] for v in values]] = 1.0
    return out, [f"{prefix}:{v}" for v in vocab]


def build_features(
    data: Dict[str, np.ndarray],
    spec: Dict,
    only: Tuple[str, ...] = ("kmer", "distance", "atac", "context"),
) -> Tuple[np.ndarray, List[str]]:
    """Concatenate the requested feature groups.

    ``only`` selects groups, which is what the ablation ladder uses.
    """
    blocks: List[np.ndarray] = []
    names: List[str] = []

    if "kmer" in only:
        block, block_names = kmer_features(data["seq"], k=int(spec.get("kmer_k", 3)))
        blocks.append(block)
        names.extend(block_names)

    if "distance" in only:
        if "distance_bp" not in data:
            raise ValueError("config asks for distance features but data has no 'distance_bp'")
        block, block_names = distance_features(data["distance_bp"])
        blocks.append(block)
        names.extend(block_names)

    if "atac" in only:
        if "atac" not in data:
            raise ValueError("config asks for atac features but data has no 'atac'")
        block, block_names = atac_features(data["atac"])
        blocks.append(block)
        names.extend(block_names)

    if "context" in only:
        for key in ("tf_id", "cell_state"):
            if key not in data:
                continue
            block, block_names = context_features(data[key], prefix=key)
            blocks.append(block)
            names.extend(block_names)

    if not blocks:
        raise ValueError("no feature groups selected")
    return np.concatenate(blocks, axis=1), names
