import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from linearbaseline.data import label_balanced_folds, make_synthetic, train_test_split_by_locus
from linearbaseline.features import build_features, kmer_features
from linearbaseline.metrics import average_precision, bootstrap_ap_ci, log_loss
from linearbaseline.model import fit_logistic, select_C


def test_synthetic_world_has_learnable_signal():
    data = make_synthetic(n_loci=400, rows_per_locus=2, seq_len=160, seed=1)
    assert data["seq"].shape == (800, 160)
    assert data["y"].shape == (800,)
    assert 0.05 < data["y"].mean() < 0.6

    X, names = build_features(data, {"kmer_k": 3})
    assert X.shape == (800, 64 + 4 + 1 + 8 + 3)
    assert len(names) == X.shape[1]
    assert np.isfinite(X).all()

    model = fit_logistic(X[:600], data["y"][:600], C=1.0)
    prob = model.predict_proba(X[600:])[:, 1]
    # the synthetic world is designed to be learnable, but not trivially so
    assert average_precision(data["y"][600:], prob) > 0.6


def test_folds_never_split_a_locus():
    data = make_synthetic(n_loci=500, rows_per_locus=3, seq_len=100, seed=2)
    locus, y = data["locus_id"], data["y"]
    folds = label_balanced_folds(locus, y, n_splits=5)

    assert len(folds) == 5
    seen = []
    for train_idx, valid_idx in folds:
        train_loci = set(locus[train_idx].tolist())
        valid_loci = set(locus[valid_idx].tolist())
        assert train_loci.isdisjoint(valid_loci)
        assert len(train_idx) + len(valid_idx) == len(locus)
        seen.append(valid_idx)
    assert len(np.concatenate(seen)) == len(locus)

    rates = [y[v].mean() for _, v in folds]
    assert max(rates) - min(rates) < 0.05


def test_split_by_locus_is_disjoint():
    rng = np.random.default_rng(0)
    locus = rng.integers(0, 200, size=600)
    tr, te = train_test_split_by_locus(locus, test_frac=0.2, seed=3)
    assert set(locus[tr]).isdisjoint(set(locus[te]))
    assert len(tr) + len(te) == len(locus)


def test_ap_matches_sklearn_and_handles_rare_positives():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    score = rng.random(500) + 0.5 * y
    assert average_precision(y, score) == pytest.approx(average_precision_score(y, score))
    assert 0.0 <= log_loss(y, np.clip(score, 0, 1)) < 5.0


def test_bootstrap_ci_is_locus_level_and_stable():
    data = make_synthetic(n_loci=300, rows_per_locus=2, seq_len=100, seed=4)
    X, _ = build_features(data, {"kmer_k": 3})
    model = fit_logistic(X, data["y"], C=1.0)
    prob = model.predict_proba(X)[:, 1]

    ci = bootstrap_ap_ci(data["y"], prob, data["locus_id"], n_boot=200, seed=0)
    assert ci["ci_low"] <= ci["ci_high"]
    assert ci["n_valid"] >= 100
    assert ci["ci_low"] - 1e-9 <= ci["point"] <= ci["ci_high"] + 1e-9


def test_select_C_returns_a_value_from_the_grid():
    data = make_synthetic(n_loci=250, rows_per_locus=2, seq_len=100, seed=5)
    X, _ = build_features(data, {"kmer_k": 3})
    C, scores = select_C(X, data["y"], data["locus_id"], [0.1, 1.0], n_splits=3, max_iter=500)
    assert C in (0.1, 1.0)
    assert set(scores) == {0.1, 1.0}


def test_kmer_features_are_normalised():
    seq = np.zeros((3, 20), dtype=np.uint8)  # all A
    counts, names = kmer_features(seq, k=3)
    assert counts.shape == (3, 64)
    assert np.allclose(counts.sum(axis=1), 1.0)
    assert counts[0, 0] == pytest.approx(1.0)  # AAA is index 0
    assert names[0] == "kmer:AAA"
