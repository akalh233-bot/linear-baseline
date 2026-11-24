"""Full-bio logistic baseline on EPCrisprBenchmark (K562, CRE->gene).

v0.4.0 incorporates the P0 correctness fixes (true zero-ChIP P1 ablation,
single log1p(DHS.RPM), strict in-fold expression standardization, scaled
in-fold motif PCA, complete distance bins including >=1 Mb, and explicit
4-non-K562 transfer reporting), P1 label-free genomic/epigenomic upgrades
(raw TSV percentiles/expandedRegion/orientation/width + pure hg38 pair-level
intervening TSS/cCRE and genome-wide power-law cCRE competition), and P2
K562 Micro-C 3D contact features (5/10/25 kb VC_SQRT-balanced + O/E).

All fitted quantities (motif scaler + PCA-4, gene-context medians, expression
median/mean/std, design-matrix standardization, logistic weights) are estimated
strictly on training chromosomes (LOCO) or K562 (zero-shot transfer).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import (
    DEFAULT_THRESHOLDS,
    bootstrap_ap_ci,
    calibration,
    paired_bootstrap_delta_ap,
    threshold_table,
)

# Column indices inside `base12` (the 12 raw EPCrispr design columns).
MARK_CONT: Tuple[int, ...] = (2, 4, 5, 6)  # h3k27ac, ctcf, h3k4me1, h3k27me3
MARK_PCT: int = 3  # h3k27ac percentile
MARK_PEAK: Tuple[int, ...] = (7, 8, 9, 10)  # peak-overlap flags
MARK_ALL: Tuple[int, ...] = tuple(sorted(MARK_CONT + (MARK_PCT,) + MARK_PEAK))

BASE12_NAMES: Tuple[str, ...] = (
    "logd", "logd2", "h3k27ac", "h3k27ac_pct", "ctcf", "h3k4me1",
    "h3k27me3", "h3k27ac_peak", "ctcf_peak", "h3k4me1_peak",
    "h3k27me3_peak", "ubiq_expr",
)
MARK_NAMES: Tuple[str, ...] = tuple(BASE12_NAMES[i] for i in MARK_ALL)

# `bio` columns: first 2 are DHS-only, last 2 are measured H3K27ac (ChIP!).
BIO_NAMES: Tuple[str, ...] = (
    "bio_dhs_mean",
    "bio_dhs_frac_gt1",
    "bio_h3k27ac_mean",
    "bio_h3k27ac_max_minus_own",
)
BIO_NOCHIP_IDX: Tuple[int, ...] = (0, 1)
BIO_ALL_IDX: Tuple[int, ...] = (0, 1, 2, 3)

# `raw_extra` columns: first 3 are ChIP-free, last 2 are expandedRegion H3K27ac (ChIP!).
RAW_EXTRA_NAMES: Tuple[str, ...] = (
    "dhs_pct",
    "is_upstream",
    "log_width",
    "h3k27ac_exp",
    "h3k27ac_exp_pct",
)
RAW_EXTRA_NOCHIP_IDX: Tuple[int, ...] = (0, 1, 2)
RAW_EXTRA_ALL_IDX: Tuple[int, ...] = (0, 1, 2, 3, 4)

# `genomic_ctx`: 7 pure hg38 genomic-topology features (zero ChIP, screen-independent).
GENOMIC_CTX_NAMES: Tuple[str, ...] = (
    "log_int_tss",
    "log_int_ccre",
    "log_ccre_5kb",
    "log_ccre_50kb",
    "log_tss_50kb",
    "log_gw_ccre_sum",
    "log_gw_ccre_share",
)

# `contact_3d`: 4 VC_SQRT-balanced K562 Micro-C 3D contact features (zero ChIP).
CONTACT_3D_NAMES: Tuple[str, ...] = (
    "mc_5kb",
    "mc_10kb",
    "mc_10kb_oe",
    "mc_25kb_oe",
)

CHIP_FEATURE_NAMES: frozenset[str] = frozenset(
    list(MARK_NAMES) + ["bio_h3k27ac_mean", "bio_h3k27ac_max_minus_own", "h3k27ac_exp", "h3k27ac_exp_pct"]
)

DISTANCE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("<20kb", 0.0, 20e3),
    ("20-100kb", 20e3, 100e3),
    ("100kb-1Mb", 100e3, 1e6),
    (">=1Mb", 1e6, float("inf")),
)


def feature_names(
    n_pca: int = 4,
    with_marks: bool = True,
    with_bio: bool = True,
    with_raw_extra: bool = True,
    with_genomic_ctx: bool = True,
    with_3d: bool = False,
) -> List[str]:
    names = ["logd", "logd2"]
    names += [f"motif_pca{i + 1}" for i in range(n_pca)]
    names += ["log_atac"]
    if with_bio:
        bio_idx = BIO_ALL_IDX if with_marks else BIO_NOCHIP_IDX
        names += [BIO_NAMES[i] for i in bio_idx]
    names += [f"gene_ctx{i + 1}" for i in range(5)]
    names += ["expr_prior"]
    if with_marks:
        names += list(MARK_NAMES)
    if with_raw_extra:
        rx_idx = RAW_EXTRA_ALL_IDX if with_marks else RAW_EXTRA_NOCHIP_IDX
        names += [RAW_EXTRA_NAMES[i] for i in rx_idx]
    if with_genomic_ctx:
        names += list(GENOMIC_CTX_NAMES)
    if with_3d:
        names += list(CONTACT_3D_NAMES)
    if not with_marks:
        leaked = CHIP_FEATURE_NAMES.intersection(names)
        assert not leaked, f"ChIP mark features leaked into zero-ChIP arm: {sorted(leaked)}"
    return names


def load_fullbio(path: str) -> Dict[str, np.ndarray]:
    """Load the packaged table into plain float64/int arrays."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for key in ("base12", "motif", "bio", "gene_ctx", "expr", "atac", "dist"):
        out[key] = out[key].astype(np.float64)
    for opt_key in ("raw_extra", "genomic_ctx", "contact_3d"):
        if opt_key in out:
            out[opt_key] = out[opt_key].astype(np.float64)
    out["y"] = out["y"].astype(int)
    out["chrom"] = out["chrom"].astype(str)
    out["gene"] = out["gene"].astype(str)
    # Retained for backwards compatibility only; loco_oof and fit_transfer
    # ALWAYS fit expression median/mean/std strictly on training rows.
    out["exprz"] = _expression_prior(out["expr"])
    return out


def _expression_prior(
    expr: np.ndarray,
    ref_mean: Optional[float] = None,
    ref_std: Optional[float] = None,
    fill: Optional[float] = None,
) -> np.ndarray:
    """Z-scored log-expression prior, imputed and standardized from reference stats."""
    ev = np.asarray(expr, dtype=np.float64).copy()
    fill_val = float(np.nanmedian(ev)) if fill is None else float(fill)
    if not np.isfinite(fill_val):
        fill_val = 0.0
    ev = np.where(np.isfinite(ev), ev, fill_val)
    if ref_mean is None or ref_std is None:
        return (ev - ev.mean()) / (ev.std() + 1e-9)
    return (ev - ref_mean) / (ref_std + 1e-9)


def _fill_gene_ctx(gene_ctx: np.ndarray, medians: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute the gene-side block; medians always come from training rows."""
    gf = np.asarray(gene_ctx, dtype=np.float64).copy()
    if medians is None:
        medians = np.array([np.nanmedian(gf[:, j]) for j in range(gf.shape[1])])
    medians = np.asarray(medians, dtype=np.float64).copy()
    if not np.isfinite(medians).all():
        medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(gf.shape[1]):
        gf[~np.isfinite(gf[:, j]), j] = medians[j]
    return gf, medians


def _fit_transform_pair(
    tr: Dict[str, np.ndarray],
    te: Dict[str, np.ndarray],
    *,
    n_pca: int = 4,
    with_marks: bool = True,
    with_bio: bool = True,
    with_raw_extra: bool = True,
    with_genomic_ctx: bool = True,
    with_3d: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit every preprocessing step on `tr` ONLY and transform both `tr` and `te`."""
    # Check optional keys availability cleanly
    use_rx = with_raw_extra and ("raw_extra" in tr) and ("raw_extra" in te)
    use_gc = with_genomic_ctx and ("genomic_ctx" in tr) and ("genomic_ctx" in te)
    use_3d = with_3d and ("contact_3d" in tr) and ("contact_3d" in te)

    # 1) Scaled motif PCA fitted strictly on `tr`
    m_mu = tr["motif"].mean(0)
    m_sd = tr["motif"].std(0) + 1e-9
    m_tr_s = (tr["motif"] - m_mu) / m_sd
    m_te_s = (te["motif"] - m_mu) / m_sd
    pca = PCA(n_components=n_pca, random_state=0).fit(m_tr_s)
    mot_tr = pca.transform(m_tr_s).astype(np.float64)
    mot_te = pca.transform(m_te_s).astype(np.float64)

    # 2) Gene-context median imputation fitted strictly on `tr`
    tr_med = np.array([np.nanmedian(tr["gene_ctx"][:, j]) for j in range(tr["gene_ctx"].shape[1])])
    gc_tr, _ = _fill_gene_ctx(tr["gene_ctx"], medians=tr_med)
    gc_te, _ = _fill_gene_ctx(te["gene_ctx"], medians=tr_med)

    # 3) Expression prior median/mean/std fitted strictly on `tr`
    ev_tr = np.asarray(tr["expr"], dtype=np.float64)
    med_e = float(np.nanmedian(ev_tr))
    if not np.isfinite(med_e):
        med_e = 0.0
    ev_tr_f = np.where(np.isfinite(ev_tr), ev_tr, med_e)
    e_mean, e_std = float(ev_tr_f.mean()), float(ev_tr_f.std())
    ez_tr = _expression_prior(tr["expr"], ref_mean=e_mean, ref_std=e_std, fill=med_e)
    ez_te = _expression_prior(te["expr"], ref_mean=e_mean, ref_std=e_std, fill=med_e)

    def _assemble_one(d: Dict[str, np.ndarray], mot: np.ndarray, gc: np.ndarray, ez: np.ndarray) -> np.ndarray:
        # `d["atac"]` is already log1p(DHS.RPM) from upstream build_epcrispr.py; do NOT double-log!
        cols = [d["base12"][:, 0:2], mot, d["atac"][:, None]]
        if with_bio:
            bio_idx = BIO_ALL_IDX if with_marks else BIO_NOCHIP_IDX
            cols.append(d["bio"][:, list(bio_idx)])
        cols.append(gc)
        cols.append(ez[:, None])
        if with_marks:
            cols.append(d["base12"][:, MARK_ALL])
        if use_rx:
            rx_idx = RAW_EXTRA_ALL_IDX if with_marks else RAW_EXTRA_NOCHIP_IDX
            cols.append(d["raw_extra"][:, list(rx_idx)])
        if use_gc:
            cols.append(d["genomic_ctx"])
        if use_3d:
            cols.append(d["contact_3d"])
        return np.column_stack(cols)

    expected_cols = feature_names(
        n_pca=n_pca,
        with_marks=with_marks,
        with_bio=with_bio,
        with_raw_extra=use_rx,
        with_genomic_ctx=use_gc,
        with_3d=use_3d,
    )
    F_tr = _assemble_one(tr, mot_tr, gc_tr, ez_tr)
    F_te = _assemble_one(te, mot_te, gc_te, ez_te)
    assert F_tr.shape[1] == len(expected_cols), f"column mismatch: {F_tr.shape[1]} vs {len(expected_cols)}"

    # 4) Design-matrix standardization fitted strictly on `tr`
    mu = F_tr.mean(0)
    sd = F_tr.std(0) + 1e-9
    return (F_tr - mu) / sd, (F_te - mu) / sd


def _subset(d: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(mask)
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in d.items()}


def loco_oof(
    d: Dict[str, np.ndarray],
    C: float = 1.0,
    n_pca: int = 4,
    with_marks: bool = True,
    with_bio: bool = True,
    with_raw_extra: bool = True,
    with_genomic_ctx: bool = True,
    with_3d: bool = False,
    max_iter: int = 400,
) -> np.ndarray:
    """Leave-one-chromosome-out out-of-fold probabilities.

    All fitted objects (motif scaler/PCA, gene-context medians, expression
    median/mean/std, design standardization, logistic weights) are estimated on
    the 22 training chromosomes only.
    """
    y = d["y"]
    chrom = d["chrom"]
    preds = np.full(len(y), np.nan, dtype=np.float64)
    for c in np.unique(chrom):
        te = chrom == c
        tr = ~te
        Z_tr, Z_te = _fit_transform_pair(
            _subset(d, tr),
            _subset(d, te),
            n_pca=n_pca,
            with_marks=with_marks,
            with_bio=with_bio,
            with_raw_extra=with_raw_extra,
            with_genomic_ctx=with_genomic_ctx,
            with_3d=with_3d,
        )
        lr = LogisticRegression(C=C, max_iter=max_iter).fit(Z_tr, y[tr])
        preds[te] = lr.predict_proba(Z_te)[:, 1]
    if not np.isfinite(preds).all():
        raise RuntimeError("some rows were never held out; check the chromosome column")
    return preds


def loco_oof_distance(d: Dict[str, np.ndarray], C: float = 1.0, max_iter: int = 400) -> np.ndarray:
    """The honest floor: distance to TSS only, nothing else."""
    y = d["y"]
    chrom = d["chrom"]
    F = d["base12"][:, 0:2]
    preds = np.full(len(y), np.nan, dtype=np.float64)
    for c in np.unique(chrom):
        te = chrom == c
        tr = ~te
        mu = F[tr].mean(0)
        sd = F[tr].std(0) + 1e-9
        Z = (F - mu) / sd
        lr = LogisticRegression(C=C, max_iter=max_iter).fit(Z[tr], y[tr])
        preds[te] = lr.predict_proba(Z[te])[:, 1]
    return preds


def fit_transfer(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    C: float = 1.0,
    n_pca: int = 4,
    with_marks: bool = True,
    with_bio: bool = True,
    with_raw_extra: bool = True,
    with_genomic_ctx: bool = True,
    with_3d: bool = False,
    max_iter: int = 400,
) -> np.ndarray:
    """Fit on `train` only and score `test` zero-shot (target labels untouched)."""
    Z_train, Z_test = _fit_transform_pair(
        train,
        test,
        n_pca=n_pca,
        with_marks=with_marks,
        with_bio=with_bio,
        with_raw_extra=with_raw_extra,
        with_genomic_ctx=with_genomic_ctx,
        with_3d=with_3d,
    )
    lr = LogisticRegression(C=C, max_iter=max_iter).fit(Z_train, train["y"])
    return lr.predict_proba(Z_test)[:, 1]


def _summarize_transfer_stratum(
    y: np.ndarray,
    score: np.ndarray,
    dist: np.ndarray,
    gene: np.ndarray,
    n_boot: int,
    seed: int,
    thresholds: Sequence[float],
    min_pos: int = 20,
) -> Dict:
    y = np.asarray(y)
    if int(y.sum()) >= min_pos:
        return summarize(y, score, dist, cluster=gene, n_boot=n_boot, seed=seed, thresholds=thresholds)
    return summarize(y, score, dist, cluster=None, n_boot=n_boot, seed=seed, thresholds=thresholds)


def run_transfer(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    C: float = 1.0,
    n_boot: int = 1000,
    seed: int = 7,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Zero-shot cross-cell-line arms on the five held-out cell types.

    Reports ``all`` (all 5 cell types), ``non_k562`` (the 4 genuinely held-out
    non-K562 cell types, excluding in-domain K562), and each cell type
    individually, plus paired gene-cluster bootstrap CIs for ``delta_marks``
    and ``delta_3d``.
    """
    cell = np.asarray(test["cell"]).astype(str)
    non_k562 = cell != "K562"
    has_3d = ("contact_3d" in train) and ("contact_3d" in test)

    arm_specs: List[Tuple[str, Dict]] = [
        ("P0_full_bio", dict(with_marks=True, with_bio=True, with_3d=False)),
        ("P0_nobio", dict(with_marks=True, with_bio=False, with_3d=False)),
        ("P1_no_marks", dict(with_marks=False, with_bio=True, with_3d=False)),
        ("P1_no_marks_nobio", dict(with_marks=False, with_bio=False, with_3d=False)),
    ]
    if has_3d:
        arm_specs.insert(0, ("P2_full_bio_3d", dict(with_marks=True, with_bio=True, with_3d=True)))

    arms: Dict[str, Dict] = {}
    pred_map: Dict[str, np.ndarray] = {}
    for name, kw in arm_specs:
        preds = fit_transfer(train, test, C=C, **kw)
        pred_map[name] = preds
        summary = _summarize_transfer_stratum(
            test["y"], preds, test["dist"], test["gene"], n_boot, seed, thresholds
        )
        if non_k562.any() and test["y"][non_k562].min() != test["y"][non_k562].max():
            summary["non_k562"] = _summarize_transfer_stratum(
                test["y"][non_k562],
                preds[non_k562],
                test["dist"][non_k562],
                test["gene"][non_k562],
                n_boot,
                seed,
                thresholds,
            )
        summary["cells"] = {}
        for label in sorted(np.unique(cell)):
            m = cell == label
            summary["cells"][label] = _summarize_transfer_stratum(
                test["y"][m], preds[m], test["dist"][m], test["gene"][m], n_boot, seed, thresholds
            )
        arms[name] = summary
        if verbose:
            a = summary["all"]
            nk = summary.get("non_k562", {}).get("all", {})
            nk_str = f"  4-non-K562 AP {nk['ap']:.4f}" if nk else ""
            print(f"{name:18s} transfer AP {a['ap']:.4f}{nk_str}  n={a['n']}  pos={a['pos']}")

    arms["delta_marks"] = {
        "all": arms["P0_full_bio"]["all"]["ap"] - arms["P1_no_marks"]["all"]["ap"],
        "all_paired": paired_bootstrap_delta_ap(
            test["y"], pred_map["P0_full_bio"], pred_map["P1_no_marks"], test["gene"], n_boot=n_boot, seed=seed
        ),
        "cells": {
            label: arms["P0_full_bio"]["cells"][label]["all"]["ap"]
            - arms["P1_no_marks"]["cells"][label]["all"]["ap"]
            for label in arms["P0_full_bio"]["cells"]
        },
    }
    if non_k562.any() and int(test["y"][non_k562].sum()) >= 20:
        arms["delta_marks"]["non_k562"] = (
            arms["P0_full_bio"]["non_k562"]["all"]["ap"] - arms["P1_no_marks"]["non_k562"]["all"]["ap"]
        )
        arms["delta_marks"]["non_k562_paired"] = paired_bootstrap_delta_ap(
            test["y"][non_k562],
            pred_map["P0_full_bio"][non_k562],
            pred_map["P1_no_marks"][non_k562],
            test["gene"][non_k562],
            n_boot=n_boot,
            seed=seed,
        )
    if has_3d:
        arms["delta_3d"] = {
            "all": arms["P2_full_bio_3d"]["all"]["ap"] - arms["P0_full_bio"]["all"]["ap"],
            "all_paired": paired_bootstrap_delta_ap(
                test["y"], pred_map["P2_full_bio_3d"], pred_map["P0_full_bio"], test["gene"], n_boot=n_boot, seed=seed
            ),
        }
        if non_k562.any() and int(test["y"][non_k562].sum()) >= 20:
            arms["delta_3d"]["non_k562"] = (
                arms["P2_full_bio_3d"]["non_k562"]["all"]["ap"] - arms["P0_full_bio"]["non_k562"]["all"]["ap"]
            )
            arms["delta_3d"]["non_k562_paired"] = paired_bootstrap_delta_ap(
                test["y"][non_k562],
                pred_map["P2_full_bio_3d"][non_k562],
                pred_map["P0_full_bio"][non_k562],
                test["gene"][non_k562],
                n_boot=n_boot,
                seed=seed,
            )
    return arms


def distance_mask(dist: np.ndarray, lo: float, hi: float) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    return (dist >= lo) & (dist < hi)


def summarize(
    y: np.ndarray,
    score: np.ndarray,
    dist: np.ndarray,
    cluster: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 7,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict:
    """Overall + distance-binned AP, threshold call sets, and calibration."""
    y = np.asarray(y)
    score = np.asarray(score)
    out: Dict = {
        "all": {
            "ap": float(average_precision_score(y, score)),
            "auc": float(roc_auc_score(y, score)),
            "n": int(len(y)),
            "pos": int(y.sum()),
            "calibration": calibration(y, score),
            "thresholds": threshold_table(y, score, thresholds),
        }
    }
    if cluster is not None:
        ci = bootstrap_ap_ci(y, score, cluster, n_boot=n_boot, seed=seed)
        out["all"].update({"ap_ci_low": ci["ci_low"], "ap_ci_high": ci["ci_high"], "n_boot_valid": ci["n_valid"]})
    for label, lo, hi in DISTANCE_BINS:
        m = distance_mask(dist, lo, hi)
        if m.sum() == 0 or y[m].min() == y[m].max():
            continue
        out[label] = {
            "ap": float(average_precision_score(y[m], score[m])),
            "auc": float(roc_auc_score(y[m], score[m])),
            "n": int(m.sum()),
            "pos": int(y[m].sum()),
            "thresholds": threshold_table(y[m], score[m], thresholds),
        }
    return out


def run_arms(
    d: Dict[str, np.ndarray],
    C: float = 1.0,
    n_boot: int = 1000,
    seed: int = 7,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run nested LOCO arms: P2 (+3D), P0 (full bio), P0_nobio, P1 (zero ChIP), P1_nobio, D0."""
    has_3d = "contact_3d" in d
    specs: List[Tuple[str, callable]] = [
        ("P0_full_bio", lambda: loco_oof(d, C=C, with_marks=True, with_bio=True, with_3d=False)),
        ("P0_nobio", lambda: loco_oof(d, C=C, with_marks=True, with_bio=False, with_3d=False)),
        ("P1_no_marks", lambda: loco_oof(d, C=C, with_marks=False, with_bio=True, with_3d=False)),
        ("P1_no_marks_nobio", lambda: loco_oof(d, C=C, with_marks=False, with_bio=False, with_3d=False)),
        ("D0_distance_only", lambda: loco_oof_distance(d, C=C)),
    ]
    if has_3d:
        specs.insert(0, ("P2_full_bio_3d", lambda: loco_oof(d, C=C, with_marks=True, with_bio=True, with_3d=True)))

    arms: Dict[str, Dict] = {}
    pred_map: Dict[str, np.ndarray] = {}
    for name, fn in specs:
        preds = fn()
        pred_map[name] = preds
        arms[name] = summarize(d["y"], preds, d["dist"], d["gene"], n_boot=n_boot, seed=seed)
        if verbose:
            a = arms[name]["all"]
            print(f"{name:18s} AP {a['ap']:.4f} [{a.get('ap_ci_low', float('nan')):.4f}, {a.get('ap_ci_high', float('nan')):.4f}]")
            for label, _, _ in DISTANCE_BINS:
                if label in arms[name]:
                    print(f"    {label:10s} AP {arms[name][label]['ap']:.4f}  n={arms[name][label]['n']}")

    arms["delta_marks"] = {
        "all": arms["P0_full_bio"]["all"]["ap"] - arms["P1_no_marks"]["all"]["ap"],
        "all_paired": paired_bootstrap_delta_ap(
            d["y"], pred_map["P0_full_bio"], pred_map["P1_no_marks"], d["gene"], n_boot=n_boot, seed=seed
        ),
        "note": "AP lost when all measured ChIP marks (base12 marks, bio H3K27ac, expandedRegion H3K27ac) are removed",
    }
    for label, _, _ in DISTANCE_BINS:
        if label in arms["P0_full_bio"] and label in arms["P1_no_marks"]:
            arms["delta_marks"][label] = arms["P0_full_bio"][label]["ap"] - arms["P1_no_marks"][label]["ap"]

    if has_3d:
        arms["delta_3d"] = {
            "all": arms["P2_full_bio_3d"]["all"]["ap"] - arms["P0_full_bio"]["all"]["ap"],
            "all_paired": paired_bootstrap_delta_ap(
                d["y"], pred_map["P2_full_bio_3d"], pred_map["P0_full_bio"], d["gene"], n_boot=n_boot, seed=seed
            ),
            "note": "AP gained on K562 LOCO by adding 4 VC_SQRT-balanced K562 Micro-C 3D contact features",
        }
        for label, _, _ in DISTANCE_BINS:
            if label in arms["P2_full_bio_3d"] and label in arms["P0_full_bio"]:
                arms["delta_3d"][label] = arms["P2_full_bio_3d"][label]["ap"] - arms["P0_full_bio"][label]["ap"]
    return arms
