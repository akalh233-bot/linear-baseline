"""Full-bio logistic baseline on EPCrisprBenchmark (K562, CRE->gene).

This is the repository's arm: a 26-column logistic regression that is the
honest floor for CRE->gene prediction on the Gasperini/Fulco EPCrispr
benchmark.  Everything about it is deliberately plain -- standardised features,
in-fold PCA for the motif block, leave-one-chromosome-out CV -- so that the
number it produces (AP 0.6715) can be audited column by column.

Design matrix, in order::

    logd, logd2                                  distance to TSS, scaled
    motif_pca1..4                                in-fold PCA of 128 JASPAR scores
    log_atac                                     log1p(DHS RPM)
    bio1..4                                      neighbourhood chromatin summary
    gene_ctx1..5                                 gene-side context (rE2G bed)
    expr_prior                                   log expression prior (Gasperini GLM)
    h3k27ac, h3k27ac_pct, ctcf, h3k4me1, h3k27me3,
    h3k27ac_peak, ctcf_peak, h3k4me1_peak, h3k27me3_peak

The last nine columns are *measured* marks: dropping them ("no marks" arm) costs
0.0208 AP, which is the deployability price of not having ChIP data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import DEFAULT_THRESHOLDS, bootstrap_ap_ci, calibration, threshold_table

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

DISTANCE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("<20kb", 0.0, 20e3),
    ("20-100kb", 20e3, 100e3),
    ("100kb-1Mb", 100e3, 1e6),
)


def feature_names(n_pca: int = 4, with_marks: bool = True) -> List[str]:
    names = ["logd", "logd2"]
    names += [f"motif_pca{i + 1}" for i in range(n_pca)]
    names += ["log_atac"]
    names += [f"bio{i + 1}" for i in range(4)]
    names += [f"gene_ctx{i + 1}" for i in range(5)]
    names += ["expr_prior"]
    if with_marks:
        names += list(MARK_NAMES)
    return names


def load_fullbio(path: str) -> Dict[str, np.ndarray]:
    """Load the packaged table into plain float64/int arrays."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for key in ("base12", "motif", "bio", "gene_ctx", "expr", "atac", "dist"):
        out[key] = out[key].astype(np.float64)
    out["y"] = out["y"].astype(int)
    out["chrom"] = out["chrom"].astype(str)
    out["gene"] = out["gene"].astype(str)
    out["exprz"] = _expression_prior(out["expr"])
    return out


def _expression_prior(
    expr: np.ndarray,
    ref_mean: Optional[float] = None,
    ref_std: Optional[float] = None,
    fill: Optional[float] = None,
) -> np.ndarray:
    """Z-scored log-expression prior.

    When ``ref_mean``/``ref_std`` are given (cross-cell-line transfer) the
    reference statistics of the *training* cell line are applied, so the test
    cell line's own expression mean never leaks into the features.  With no
    reference the array is standardised on its own rows, which is what
    :func:`load_fullbio` does for the packaged K562 table.  ``fill`` imputes
    missing genes (the training median is the right choice in transfer).
    """
    ev = np.asarray(expr, dtype=np.float64).copy()
    ev = np.where(np.isfinite(ev), ev, np.nanmedian(ev) if fill is None else fill)
    if ref_mean is None or ref_std is None:
        return (ev - ev.mean()) / (ev.std() + 1e-9)
    return (ev - ref_mean) / (ref_std + 1e-9)


def _fill_gene_ctx(gene_ctx: np.ndarray, medians: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute the gene-side block; medians can come from a training set."""
    gf = np.asarray(gene_ctx, dtype=np.float64).copy()
    if medians is None:
        medians = np.array([np.nanmedian(gf[:, j]) for j in range(gf.shape[1])])
    medians = np.asarray(medians, dtype=np.float64).copy()
    if not np.isfinite(medians).all():
        # A column that is entirely missing in the training rows cannot be
        # imputed from them; fall back to 0.0, the mean of the standardised
        # training block, and let the logistic weight decide what it is worth.
        medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(gf.shape[1]):
        gf[~np.isfinite(gf[:, j]), j] = medians[j]
    return gf, medians


def _assemble(
    d: Dict[str, np.ndarray],
    motif_pca: np.ndarray,
    gene_ctx: np.ndarray,
    with_marks: bool,
) -> np.ndarray:
    cols = [
        d["base12"][:, 0:2],
        motif_pca,
        np.log1p(d["atac"])[:, None],
        d["bio"],
        gene_ctx,
        d["exprz"][:, None],
    ]
    if with_marks:
        cols.append(d["base12"][:, MARK_ALL])
    return np.column_stack(cols)


def loco_oof(
    d: Dict[str, np.ndarray],
    C: float = 1.0,
    n_pca: int = 4,
    with_marks: bool = True,
    max_iter: int = 400,
) -> np.ndarray:
    """Leave-one-chromosome-out out-of-fold probabilities.

    All fitted objects (PCA, gene-context medians, mean/std, logistic weights)
    are estimated on the training chromosomes only, so the held-out chromosome
    never touches a fitted statistic.
    """
    y = d["y"]
    chrom = d["chrom"]
    motif = d["motif"]
    gene_ctx = d["gene_ctx"].astype(np.float64).copy()

    preds = np.full(len(y), np.nan, dtype=np.float64)
    for c in np.unique(chrom):
        te = chrom == c
        tr = ~te

        pca = PCA(n_components=n_pca, random_state=0).fit(motif[tr])
        motif_pca = pca.transform(motif).astype(np.float64)

        medians = np.array([np.nanmedian(gene_ctx[tr, j]) for j in range(gene_ctx.shape[1])])
        gf, _ = _fill_gene_ctx(gene_ctx, medians=medians)

        F = _assemble(d, motif_pca, gf, with_marks)
        mu = F[tr].mean(0)
        sd = F[tr].std(0) + 1e-9
        Z = (F - mu) / sd
        lr = LogisticRegression(C=C, max_iter=max_iter).fit(Z[tr], y[tr])
        preds[te] = lr.predict_proba(Z[te])[:, 1]
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
    max_iter: int = 400,
) -> np.ndarray:
    """Fit the full-bio model on one cell line, score a *different* cell line.

    Everything fitted -- PCA, gene-context medians, expression centring,
    standardisation, logistic weights -- is estimated on ``train`` only; the
    test cell line contributes raw columns and nothing else.  This is the
    zero-shot transfer protocol: the labels of the target cell type are never
    touched, so the resulting probabilities can be thresholded and scored.

    The test table must carry the same keys as :func:`load_fullbio` (``base12``,
    ``motif``, ``bio``, ``gene_ctx``, ``expr``, ``atac``, ``dist``, ``y``,
    ``chrom``, ``gene``); ``cell`` is used only by :func:`run_transfer` and is
    not needed here.
    """
    motif_pca_train = PCA(n_components=n_pca, random_state=0).fit(train["motif"])
    motif_train = motif_pca_train.transform(train["motif"]).astype(np.float64)
    motif_test = motif_pca_train.transform(test["motif"]).astype(np.float64)

    train_medians = np.array([np.nanmedian(train["gene_ctx"][:, j]) for j in range(train["gene_ctx"].shape[1])])
    train_gc, _ = _fill_gene_ctx(train["gene_ctx"], medians=train_medians)
    test_gc, _ = _fill_gene_ctx(test["gene_ctx"], medians=train_medians)

    train_expr = np.asarray(train["expr"], dtype=np.float64)
    train_expr = np.where(np.isfinite(train_expr), train_expr, np.nanmedian(train_expr))
    expr_mean, expr_std = float(train_expr.mean()), float(train_expr.std())
    test_exprz = _expression_prior(test["expr"], ref_mean=expr_mean, ref_std=expr_std, fill=float(np.nanmedian(train_expr)))

    F_train = _assemble(train, motif_train, train_gc, with_marks)
    test_view = {**test, "exprz": test_exprz}
    F_test = _assemble(test_view, motif_test, test_gc, with_marks)

    mu = F_train.mean(0)
    sd = F_train.std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=max_iter).fit((F_train - mu) / sd, train["y"])
    return lr.predict_proba((F_test - mu) / sd)[:, 1]


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
    """Summarize one stratum, with a gene bootstrap only if it can be trusted.

    Small target cell types (7-16 positives) cannot support a 1,000-resample
    gene bootstrap -- resamples without both classes are skipped and the
    interval degenerates.  Below ``min_pos`` positives the point AP is reported
    without an interval rather than a misleading one.
    """
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
    """Zero-shot cross-cell-line arms: K562-trained P0 and P1 on held-out types.

    Returns the same shape as :func:`run_arms` (``P0_full_bio``, ``P1_no_marks``,
    ``delta_marks``) plus a ``cells`` mapping inside each arm.  ``test["cell"]``
    is required and is the cell line of each held-out pair.
    """
    cell = np.asarray(test["cell"]).astype(str)
    arms: Dict[str, Dict] = {}
    for name, with_marks in (("P0_full_bio", True), ("P1_no_marks", False)):
        preds = fit_transfer(train, test, C=C, with_marks=with_marks)
        summary = _summarize_transfer_stratum(
            test["y"], preds, test["dist"], test["gene"], n_boot, seed, thresholds
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
            print(f"{name:16s} transfer AP {a['ap']:.4f}  n={a['n']}  pos={a['pos']}")
            for label, cell_summary in summary["cells"].items():
                c = cell_summary["all"]
                at_90 = min(c["thresholds"], key=lambda r: abs(r["threshold"] - 0.90))
                print(
                    f"    {label:8s} AP {c['ap']:.4f}  n={c['n']:5d} pos={c['pos']:4d}"
                    f"  calls@0.90={at_90['flagged']:4d} prec={at_90['precision']:.3f}"
                )
    arms["delta_marks"] = {"all": arms["P0_full_bio"]["all"]["ap"] - arms["P1_no_marks"]["all"]["ap"]}
    arms["delta_marks"]["cells"] = {
        label: arms["P0_full_bio"]["cells"][label]["all"]["ap"] - arms["P1_no_marks"]["cells"][label]["all"]["ap"]
        for label in arms["P0_full_bio"]["cells"]
    }
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
    """Overall + distance-binned AP, threshold call sets, and calibration.

    ``thresholds`` is carried into every stratum so the near-field and distal
    bins answer the same deployment question ("how many calls, how many true")
    as the pooled arm.  Calibration (Brier, ECE, reliability bins) is reported
    once, on the pooled rows, because the bins are too sparse per distance bin
    to mean anything.
    """
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
    """P0 (full bio) vs P1 (no marks) vs D0 (distance only)."""
    arms: Dict[str, Dict] = {}
    specs = (
        ("P0_full_bio", lambda: loco_oof(d, C=C, with_marks=True)),
        ("P1_no_marks", lambda: loco_oof(d, C=C, with_marks=False)),
        ("D0_distance_only", lambda: loco_oof_distance(d, C=C)),
    )
    for name, fn in specs:
        preds = fn()
        arms[name] = summarize(d["y"], preds, d["dist"], d["gene"], n_boot=n_boot, seed=seed)
        if verbose:
            a = arms[name]["all"]
            print(f"{name:16s} AP {a['ap']:.4f} [{a.get('ap_ci_low', float('nan')):.4f}, {a.get('ap_ci_high', float('nan')):.4f}]")
            for label, _, _ in DISTANCE_BINS:
                if label in arms[name]:
                    print(f"    {label:10s} AP {arms[name][label]['ap']:.4f}  n={arms[name][label]['n']}")
    arms["delta_marks"] = {
        "all": arms["P0_full_bio"]["all"]["ap"] - arms["P1_no_marks"]["all"]["ap"],
        "note": "AP lost when the nine measured mark columns are removed",
    }
    for label, _, _ in DISTANCE_BINS:
        if label in arms["P0_full_bio"] and label in arms["P1_no_marks"]:
            arms["delta_marks"][label] = arms["P0_full_bio"][label]["ap"] - arms["P1_no_marks"][label]["ap"]
    return arms
