"""Tests for the real-data full-bio arm (v0.4.0: P0 fixes + P1 genomic + P2 3D)."""

from __future__ import annotations

import numpy as np
import pytest

from linearbaseline.fullbio import (
    CHIP_FEATURE_NAMES,
    MARK_ALL,
    feature_names,
    fit_transfer,
    load_fullbio,
    loco_oof,
    run_transfer,
    summarize,
)
from linearbaseline.metrics import calibration, paired_bootstrap_delta_ap, threshold_table


def _toy_table(seed: int = 0, n_per_chrom: int = 120, n_chrom: int = 5):
    """A tiny table with the same keys/dtypes as the packaged v0.4.0 real data."""
    rng = np.random.default_rng(seed)
    n = n_per_chrom * n_chrom
    chrom = np.repeat([f"chr{i + 1}" for i in range(n_chrom)], n_per_chrom).astype("U")
    gene = np.repeat([f"g{i}" for i in range(n // 2)], 2).astype("U")

    dist = 10 ** rng.uniform(3.0, 6.0, size=n)
    logd = np.log1p(dist) / 10.0

    # Low-rank correlated motif block (like real JASPAR families): two latent
    # factors shared across columns so scaled PCA-2 recovers the signal.
    lat = rng.normal(size=(n, 2))
    motif = lat @ rng.normal(size=(2, 12)) + rng.normal(scale=0.25, size=(n, 12))

    bio = rng.normal(size=(n, 4))
    gene_ctx = rng.normal(size=(n, 5))
    atac = np.log1p(rng.gamma(2.0, 0.5, size=n))  # already log1p(DHS.RPM)
    expr = rng.normal(size=n)
    raw_extra = rng.normal(size=(n, 5))
    genomic_ctx = rng.normal(size=(n, 7))
    contact_3d = rng.normal(size=(n, 4))

    base12 = np.zeros((n, 12))
    base12[:, 0] = logd
    base12[:, 1] = logd**2
    base12[:, MARK_ALL] = rng.normal(size=(n, len(MARK_ALL)))

    z = 2.5 * lat[:, 0] + 1.5 * lat[:, 1] - 0.8 * logd + 0.5 * atac + 0.4 * genomic_ctx[:, 0]
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
        "raw_extra": raw_extra,
        "genomic_ctx": genomic_ctx,
        "contact_3d": contact_3d,
        "y": y,
        "chrom": chrom,
        "gene": gene,
        "cell": np.where(np.arange(n) % 2 == 0, "K562", "WTC11").astype("U"),
        "exprz": (expr - expr.mean()) / (expr.std() + 1e-9),
    }


def test_mark_indices_and_feature_names_agree():
    assert MARK_ALL == (2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert len(feature_names(with_marks=True, with_bio=True, with_3d=True)) == 42
    assert len(feature_names(with_marks=True, with_bio=True, with_3d=False)) == 38
    assert len(feature_names(with_marks=True, with_bio=False, with_3d=False)) == 34
    assert len(feature_names(with_marks=False, with_bio=True, with_3d=False)) == 25
    assert len(feature_names(with_marks=False, with_bio=False, with_3d=False)) == 23
    # Zero-ChIP arms must never contain any ChIP-derived column
    p1_names = set(feature_names(with_marks=False, with_bio=True, with_3d=False))
    assert p1_names.isdisjoint(CHIP_FEATURE_NAMES)


def test_p1_no_marks_is_strictly_chip_free():
    """Perturbing ALL ChIP columns (base12 marks, bio[:,2:4], raw_extra[:,3:5])
    must leave P1_no_marks predictions bit-identical in both LOCO and transfer."""
    d = _toy_table(seed=41)
    d_corrupt = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    rng = np.random.default_rng(99)
    d_corrupt["base12"][:, MARK_ALL] = rng.normal(scale=100.0, size=(len(d["y"]), len(MARK_ALL)))
    d_corrupt["bio"][:, 2:4] = rng.normal(scale=100.0, size=(len(d["y"]), 2))
    d_corrupt["raw_extra"][:, 3:5] = rng.normal(scale=100.0, size=(len(d["y"]), 2))

    p_clean = loco_oof(d, C=1.0, n_pca=2, with_marks=False, max_iter=200)
    p_corrupt = loco_oof(d_corrupt, C=1.0, n_pca=2, with_marks=False, max_iter=200)
    np.testing.assert_allclose(p_clean, p_corrupt, rtol=0.0, atol=0.0)

    t_clean = fit_transfer(d, _toy_table(seed=42), C=1.0, n_pca=2, with_marks=False, max_iter=200)
    te_corrupt = _toy_table(seed=42)
    te_corrupt["base12"][:, MARK_ALL] = 999.0
    te_corrupt["bio"][:, 2:4] = -999.0
    te_corrupt["raw_extra"][:, 3:5] = 555.0
    t_corrupt = fit_transfer(d_corrupt, te_corrupt, C=1.0, n_pca=2, with_marks=False, max_iter=200)
    np.testing.assert_allclose(t_clean, t_corrupt, rtol=0.0, atol=0.0)


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
    assert set(loaded) >= {"base12", "motif", "raw_extra", "genomic_ctx", "contact_3d", "y", "chrom", "gene"}
    assert loaded["chrom"].dtype.kind == "U"
    assert loaded["exprz"].shape == loaded["y"].shape


def test_summarize_reports_distance_bins():
    d = _toy_table(seed=7)
    preds = loco_oof(d, C=1.0, n_pca=2, max_iter=200)
    out = summarize(d["y"], preds, d["dist"])
    assert "all" in out
    assert any(k.endswith("kb") for k in out)


def test_threshold_table_is_a_confusion_matrix():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=500)
    score = np.clip(0.7 * y + rng.normal(scale=0.3, size=500), 0.0, 1.0)
    rows = threshold_table(y, score, thresholds=(0.5, 0.8, 0.9, 0.95))
    flagged = []
    for row in rows:
        assert row["tp"] + row["fp"] == row["flagged"]
        assert row["tp"] + row["fn"] == int(y.sum())
        assert row["tn"] + row["fp"] == int((y == 0).sum())
        assert row["precision"] == pytest.approx(row["tp"] / max(row["flagged"], 1))
        assert row["recall"] == pytest.approx(row["tp"] / int(y.sum()))
        assert row["specificity"] == pytest.approx(row["tn"] / int((y == 0).sum()))
        flagged.append(row["flagged"])
    assert flagged == sorted(flagged, reverse=True), "a higher cut cannot flag more rows"


def test_calibration_bins_cover_every_row():
    rng = np.random.default_rng(13)
    y = rng.integers(0, 2, size=400)
    score = np.clip(0.6 * y + rng.normal(scale=0.35, size=400), 0.0, 1.0)
    out = calibration(y, score, n_bins=10)
    assert sum(b["n"] for b in out["bins"]) == len(y)
    assert 0.0 <= out["ece"] <= 1.0
    assert out["brier"] == pytest.approx(float(np.mean((score - y) ** 2)))
    assert all(b["lo"] <= b["mean_pred"] <= b["hi"] for b in out["bins"])


def test_paired_bootstrap_delta_ap():
    rng = np.random.default_rng(17)
    y = rng.integers(0, 2, size=300)
    g = np.repeat(np.arange(60), 5)
    s_good = np.clip(0.8 * y + rng.normal(scale=0.25, size=300), 0.0, 1.0)
    s_weak = np.clip(0.3 * y + rng.normal(scale=0.35, size=300), 0.0, 1.0)
    res = paired_bootstrap_delta_ap(y, s_good, s_weak, g, n_boot=200, seed=7)
    assert res["delta"] > 0
    assert res["ci_low"] <= res["delta"] <= res["ci_high"]
    assert res["n_valid"] > 100


def test_summarize_includes_thresholds_and_calibration():
    d = _toy_table(seed=19)
    preds = loco_oof(d, C=1.0, n_pca=2, max_iter=200)
    out = summarize(d["y"], preds, d["dist"])
    cuts = [r["threshold"] for r in out["all"]["thresholds"]]
    assert cuts == [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    assert "calibration" in out["all"]
    assert all("thresholds" in stratum for stratum in out.values())


def test_fit_transfer_ignores_target_labels():
    train = _toy_table(seed=23)
    test = _toy_table(seed=29)
    flipped = {**test, "y": 1 - test["y"]}
    a = fit_transfer(train, test, n_pca=2, max_iter=200)
    b = fit_transfer(train, flipped, n_pca=2, max_iter=200)
    np.testing.assert_allclose(a, b)
    assert ((a >= 0) & (a <= 1)).all()


def test_run_transfer_reports_each_cell_and_non_k562():
    train = _toy_table(seed=31)
    test = _toy_table(seed=37)
    arms = run_transfer(train, test, n_boot=200, verbose=False)
    assert set(arms["P0_full_bio"]["cells"]) == {"K562", "WTC11"}
    assert "non_k562" in arms["P0_full_bio"]
    for arm in ("P2_full_bio_3d", "P0_full_bio", "P0_nobio", "P1_no_marks", "P1_no_marks_nobio"):
        assert arms[arm]["all"]["n"] == len(test["y"])
        assert "thresholds" in arms[arm]["all"]
        for cell in arms[arm]["cells"].values():
            assert cell["all"]["n"] > 0
    assert "all_paired" in arms["delta_marks"]
    assert "all_paired" in arms["delta_3d"]


def test_readme_matches_committed_results():
    """The headline numbers in README.md must equal the committed JSON."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "results" / "fullbio_loco.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    for arm in ("P2_full_bio_3d", "P0_full_bio", "P0_nobio", "P1_no_marks", "P1_no_marks_nobio", "D0_distance_only"):
        ap = payload["arms"][arm]["all"]["ap"]
        assert f"{ap:.4f}" in readme, f"{arm} AP {ap:.4f} missing from README"
    for label in ("<20kb", "20-100kb", "100kb-1Mb", ">=1Mb"):
        ap = payload["arms"]["P0_full_bio"][label]["ap"]
        assert f"{ap:.4f}" in readme, f"P0 {label} AP {ap:.4f} missing from README"


def test_readme_matches_committed_transfer_results():
    """The transfer numbers in README.md must equal the committed JSON."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "results" / "heldout_transfer.json").read_text(encoding="utf-8"))
    readme = (root / "README.md").read_text(encoding="utf-8")

    p0 = payload["arms"]["P0_full_bio"]
    assert f"{p0['all']['ap']:.4f}" in readme, "pooled transfer AP missing from README"
    assert f"{p0['non_k562']['all']['ap']:.4f}" in readme, "4-non-K562 transfer AP missing from README"
    for label, cell in p0["cells"].items():
        assert f"{cell['all']['ap']:.4f}" in readme, f"{label} transfer AP missing from README"
    at_90 = next(r for r in p0["all"]["thresholds"] if r["threshold"] == 0.90)
    assert f"{at_90['precision']:.4f}" in readme, "pooled precision at 0.90 missing from README"

    loco = json.loads((root / "results" / "fullbio_loco.json").read_text(encoding="utf-8"))
    for arm in ("P2_full_bio_3d", "P0_full_bio", "P1_no_marks", "D0_distance_only"):
        rows = loco["arms"][arm]["all"]["thresholds"]
        assert [r["threshold"] for r in rows] == [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
        assert "calibration" in loco["arms"][arm]["all"]
