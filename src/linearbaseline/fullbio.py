"""Full-bio logistic baseline on EPCrisprBenchmark (K562, CRE->gene).

This is the *real-data* counterpart of :mod:`linearbaseline.models`: a 26-column
logistic regression that is the honest floor for CRE->gene prediction on the
Gasperini/Fulco EPCrispr benchmark.  Everything about it is deliberately plain --
standardised features, in-fold PCA for the motif block, leave-one-chromosome-out
CV -- so that the number it produces (AP 0.6715) can be audited column by
column.

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

from .metrics import bootstrap_ap_ci

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


def _expression_prior(expr: np.ndarray) -> np.ndarray:
    ev = np.asarray(expr, dtype=np.float64).copy()
    ev = np.where(np.isfinite(ev), ev, np.nanmedian(ev))
    return (ev - ev.mean()) / (ev.std() + 1e-9)


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

        gf = gene_ctx.copy()
        for j in range(gf.shape[1]):
            med = np.nanmedian(gf[tr, j])
            gf[~np.isfinite(gf[:, j]), j] = med

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
) -> Dict:
    """Overall + distance-binned AP, with a gene-level bootstrap interval."""
    y = np.asarray(y)
    score = np.asarray(score)
    out: Dict = {
        "all": {
            "ap": float(average_precision_score(y, score)),
            "auc": float(roc_auc_score(y, score)),
            "n": int(len(y)),
            "pos": int(y.sum()),
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
