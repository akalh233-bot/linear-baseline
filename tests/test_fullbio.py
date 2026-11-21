"""Tests for the real-data full-bio arm."""

from __future__ import annotations

import numpy as np
import pytest

from linearbaseline.fullbio import (
    MARK_ALL,
    feature_names,
    load_fullbio,
    loco_oof,
    summarize,
)


def _toy_table(seed: int = 0, n_per_chrom: int = 120, n_chrom: int = 5):
    """A tiny table with the same keys/dtypes as the packaged real data."""
    rng = np.random.default_rng(seed)
    n = n_per_chrom * n_chrom
    chrom = np.repeat([f"chr{i + 1}" for i in range(n_chrom)], n_per_chrom).astype("U")
    gene = np.repeat([f"g{i}" for i in range(n // 2)], 2).astype("U")

    dist = 10 ** rng.uniform(3.0, 6.0, size=n)
    logd = np.log1p(dist) / 10.0

    # Low-rank motif block: the first two columns carry the signal, the rest is
    # near-isotropic noise.  PCA-4 then keeps the signal, as it does on the real
    # JASPAR matrix, instead of rotating it away.
    motif = rng.normal(size=(n, 12))
    motif[:, :2] *= 3.0
    motif[:, 2:] *= 0.3

    bio = rng.normal(size=(n, 4))
    gene_ctx = rng.normal(size=(n, 5))
    atac = rng.gamma(2.0, 0.5, size=n)
    expr = rng.normal(size=n)

    base12 = np.zeros((n, 12))
    base12[:, 0] = logd
    base12[:, 1] = logd**2
    base12[:, MARK_ALL] = rng.normal(size=(n, len(MARK_ALL)))

    z = 2.5 * motif[:, 0] + 1.5 * motif[:, 1] - 0.8 * logd + 0.5 * np.log1p(atac)
    y = (z + rng.normal(scale=0.3, size=n) > np.quantile(z, 0.8)).astype(int)
    if y.sum() == 0 or y.sum() == n:
        raise AssertionError("toy table degenerated")

    return {
        "base12": base12,
        "motif": motif,
        "bio": bio,
        "gene_ctx": gene_ctx,
        "expr": expr,
        "atac": atac,
        "dist": dist,
        "y": y,
        "chrom": chrom,
        "gene": gene,
        "exprz": (expr - expr.mean()) / (expr.std() + 1e-9),
    }


def test_mark_indices_and_feature_names_agree():
    assert MARK_ALL == (2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert len(feature_names(with_marks=True)) == 26
    assert len(feature_names(with_marks=False)) == 17


def test_loco_oof_is_complete_and_predictive():
    d = _toy_table()
    preds = loco_oof(d, C=1.0, n_pca=2, max_iter=200)
    assert preds.shape == d["y"].shape
    assert np.isfinite(preds).all(), "every row must be held out exactly once"
    assert ((preds >= 0) & (preds <= 1)).all()

    overall = summarize(d["y"], preds, d["dist"])
    assert overall["all"]["ap"] > 0.6
    assert overall["all"]["n"] == len(d["y"])


def test_marks_ablation_is_a_real_arm():
    d = _toy_table(seed=3)
    with_marks = loco_oof(d, C=1.0, n_pca=2, max_iter=200, with_marks=True)
    no_marks = loco_oof(d, C=1.0, n_pca=2, max_iter=200, with_marks=False)
    assert not np.allclose(with_marks, no_marks)


def test_load_fullbio_roundtrip(tmp_path):
    d = _toy_table(seed=5)
    path = tmp_path / "features.npz"
    np.savez_compressed(path, **d)
    loaded = load_fullbio(str(path))
    assert set(loaded) >= {"base12", "motif", "y", "chrom", "gene", "exprz"}
    assert loaded["chrom"].dtype.kind == "U"
    assert loaded["exprz"].shape == loaded["y"].shape
    np.testing.assert_allclose(loaded["exprz"].mean(), 0.0, atol=1e-9)


def test_summarize_reports_distance_bins():
    d = _toy_table(seed=7)
    preds = loco_oof(d, C=1.0, n_pca=2, max_iter=200)
    out = summarize(d["y"], preds, d["dist"])
    assert "all" in out
    assert any(k.endswith("kb") for k in out)


def test_readme_matches_committed_results():
    """The headline numbers in README.md must equal the committed JSON."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "results" / "fullbio_loco.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    for arm in ("P0_full_bio", "P1_no_marks", "D0_distance_only"):
        ap = payload["arms"][arm]["all"]["ap"]
        assert f"{ap:.4f}" in readme, f"{arm} AP {ap:.4f} missing from README"
    for label in ("<20kb", "20-100kb", "100kb-1Mb"):
        ap = payload["arms"]["P0_full_bio"][label]["ap"]
        assert f"{ap:.4f}" in readme, f"P0 {label} AP {ap:.4f} missing from README"
