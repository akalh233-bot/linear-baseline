# -*- coding: utf-8 -*-
"""Publication experiment suite for linear-baseline v0.4.0.

Runs strictly leak-free analyses on local data and committed results:
  1. Nested-CV Platt & Isotonic recalibration (K562 LOCO + heldout_5ct transfer)
  2. Exact head-to-head comparison on the 10,356 K562 23-fold LOCO split against
     1 kb CNN surrogate marks, Borzoi (524 kb), and Enformer (196 kb)
  3. ENCODE-rE2G `direct_vs_indirect_negative` weighted AP & quantitative `EffectSize`
     correlation (Spearman / Pearson)
  4. Independent Fulco2019 (5,091 pairs) & Astrocyte screen check (overlap-split)
  5. Biological mechanism quantification:
       - Housekeeping / ubiquitous gene buffering (Fulco 2019 / Bergman 2022)
       - Intervening promoter competition & CTCF/H3K27me3 insulation
       - Classic loci case studies: MYC (>1 Mb), GATA1/HDAC6, HBE1 LCR, CCND1 (HCT116)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linearbaseline.fullbio import (  # noqa: E402
    DISTANCE_BINS,
    feature_names,
    fit_transfer,
    load_fullbio,
    loco_oof,
    _fit_transform_pair,
    _subset,
)
from linearbaseline.metrics import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    calibration,
    operating_point,
    paired_bootstrap_delta_ap,
    threshold_table,
)

RAW_TR = Path(r"E:\triomics\data\raw\rE2G\training_K562\EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz")
RAW_HE = Path(r"E:\triomics\data\raw\rE2G\heldout_5_cell_types\EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz")
ENF_OOF = Path(r"E:\triomics\phase3\realdata\results\enformer_final_oof.npz")
CNN_OOF = Path(r"E:\triomics\phase3\realdata\results\cnn_marks_oof.npz")
FULCO_PATH = Path(r"E:\triomics\data\raw\fulco2019\EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv.gz")


def _fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def nll(p):
        z = np.clip(p[0] * logits + p[1], -30.0, 30.0)
        pr = 1.0 / (1.0 + np.exp(-z))
        return -float(np.mean(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)))
    res = minimize(nll, [1.0, 0.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])


def exp_recalibration(d_tr: dict, d_he: dict) -> dict:
    """Strictly nested-CV Platt & Isotonic recalibration on P0_full_bio & P1_no_marks."""
    out = {}
    chrom = d_tr["chrom"]
    y = d_tr["y"]
    non_k562 = d_he["cell"] != "K562"

    for arm, wm in [("P0_full_bio", True), ("P1_no_marks", False), ("P2_full_bio_3d", True)]:
        w3d = (arm == "P2_full_bio_3d")
        raw_oof = np.full(len(y), np.nan)
        platt_oof = np.full(len(y), np.nan)
        iso_oof = np.full(len(y), np.nan)

        for c in np.unique(chrom):
            te = chrom == c
            tr = ~te
            sub_tr, sub_te = _subset(d_tr, tr), _subset(d_tr, te)
            Z_tr, Z_te = _fit_transform_pair(sub_tr, sub_te, with_marks=wm, with_3d=w3d)
            clf = LogisticRegression(C=1.0, max_iter=400).fit(Z_tr, sub_tr["y"])
            te_logits = clf.decision_function(Z_te)
            raw_oof[te] = clf.predict_proba(Z_te)[:, 1]

            # Inner 5-fold chrom-grouped OOF logits for calibration fitting
            uc = np.unique(sub_tr["chrom"])
            in_logits = np.zeros(len(sub_tr["y"]))
            for fi in range(5):
                vm = np.isin(sub_tr["chrom"], uc[fi::5])
                tm = ~vm
                c_in = LogisticRegression(C=1.0, max_iter=400).fit(Z_tr[tm], sub_tr["y"][tm])
                in_logits[vm] = c_in.decision_function(Z_tr[vm])
            a, b = _fit_platt(in_logits, sub_tr["y"])
            platt_oof[te] = 1.0 / (1.0 + np.exp(-np.clip(a * te_logits + b, -30.0, 30.0)))
            iso = IsotonicRegression(out_of_bounds="clip").fit(1.0 / (1.0 + np.exp(-in_logits)), sub_tr["y"])
            iso_oof[te] = iso.predict(raw_oof[te])

        # Transfer recalibration fitted on full K562 5-fold OOF logits
        Z_tr_all, Z_he_all = _fit_transform_pair(d_tr, d_he, with_marks=wm, with_3d=w3d)
        clf_all = LogisticRegression(C=1.0, max_iter=400).fit(Z_tr_all, y)
        he_raw = clf_all.predict_proba(Z_he_all)[:, 1]
        he_logits = clf_all.decision_function(Z_he_all)

        uc = np.unique(chrom)
        in_logits = np.zeros(len(y))
        for fi in range(5):
            vm = np.isin(chrom, uc[fi::5])
            tm = ~vm
            c_in = LogisticRegression(C=1.0, max_iter=400).fit(Z_tr_all[tm], y[tm])
            in_logits[vm] = c_in.decision_function(Z_tr_all[vm])
        a, b = _fit_platt(in_logits, y)
        he_platt = 1.0 / (1.0 + np.exp(-np.clip(a * he_logits + b, -30.0, 30.0)))
        iso_all = IsotonicRegression(out_of_bounds="clip").fit(1.0 / (1.0 + np.exp(-in_logits)), y)
        he_iso = iso_all.predict(he_raw)

        arm_res = {}
        for tag, pred_l, pred_h in [("raw", raw_oof, he_raw), ("platt", platt_oof, he_platt), ("isotonic", iso_oof, he_iso)]:
            cal = calibration(y, pred_l)
            op90 = operating_point(y, pred_l, 0.90)
            op90_5 = operating_point(d_he["y"], pred_h, 0.90)
            op90_4 = operating_point(d_he["y"][non_k562], pred_h[non_k562], 0.90)
            arm_res[tag] = {
                "loco_ap": float(average_precision_score(y, pred_l)),
                "brier": cal["brier"],
                "ece": cal["ece"],
                "top_pred": cal["bins"][-1]["mean_pred"],
                "top_obs": cal["bins"][-1]["frac_pos"],
                "loco_90": op90,
                "tr5_90": op90_5,
                "tr4_90": op90_4,
            }
        out[arm] = arm_res
    return out


def exp_head_to_head(d_tr: dict, d_he: dict) -> dict:
    """Head-to-head on exact 10,356 K562 LOCO split vs 1kb CNN, Borzoi, Enformer."""
    y = d_tr["y"]
    gene = d_tr["gene"]
    dist = d_tr["dist"]
    enf = np.load(ENF_OOF, allow_pickle=True)
    assert np.array_equal(enf["y"].astype(int), y), "enformer_final_oof.npz y mismatch!"

    cnn = np.load(CNN_OOF, allow_pickle=True)
    # Build P1_no_marks + 1kb CNN surrogate marks (strictly zero-ChIP!)
    # Replace base12[:, MARK_ALL] with cnn['oof'] (for train LOCO) and cnn['held'] (for transfer)
    d_tr_cnn = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d_tr.items()}
    d_he_cnn = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d_he.items()}
    from linearbaseline.fullbio import MARK_ALL
    d_tr_cnn["base12"][:, MARK_ALL] = cnn["oof"]
    d_he_cnn["base12"][:, MARK_ALL] = cnn["held"]
    # Keep bio[:, 2:4] and raw_extra[:, 3:5] zeroed or use with_marks=True ONLY on CNN base12 marks:
    # We replace bio[:, 2:4] with 0 and raw_extra[:, 3:5] with 0 so ZERO measured ChIP enters!
    d_tr_cnn["bio"][:, 2:4] = 0.0
    d_tr_cnn["raw_extra"][:, 3:5] = 0.0
    d_he_cnn["bio"][:, 2:4] = 0.0
    d_he_cnn["raw_extra"][:, 3:5] = 0.0

    p1_cnn_oof = loco_oof(d_tr_cnn, C=1.0, with_marks=True, with_bio=True, with_3d=False)
    p1_cnn_tr = fit_transfer(d_tr_cnn, d_he_cnn, C=1.0, with_marks=True, with_bio=True, with_3d=False)

    p1_oof = loco_oof(d_tr, C=1.0, with_marks=False, with_bio=True, with_3d=False)
    p0_oof = loco_oof(d_tr, C=1.0, with_marks=True, with_bio=True, with_3d=False)
    p2_oof = loco_oof(d_tr, C=1.0, with_marks=True, with_bio=True, with_3d=True)

    models = [
        ("Enformer_only_A_old (v0.2)", enf["A"], "no ChIP (old 12-col)"),
        ("Borzoi_524kb_D (v0.2+Borzoi)", enf["D"], "no ChIP + Borzoi"),
        ("Enformer_196kb_G (v0.2+Enf)", enf["G"], "no ChIP + Enformer"),
        ("Measured_ChIP_C_old (v0.2+ChIP)", enf["C"], "measured ChIP (old)"),
        ("Enformer+ChIP_H (v0.2+Enf+ChIP)", enf["H"], "measured ChIP + Enformer"),
        ("P1_no_marks (v0.4.0, 25f)", p1_oof, "strictly zero-ChIP"),
        ("P1_no_marks+1kbCNN (v0.4.0)", p1_cnn_oof, "zero-ChIP + 1kb CNN"),
        ("P0_full_bio (v0.4.0, 38f)", p0_oof, "measured ChIP"),
        ("P2_full_bio_3d (v0.4.0, 42f)", p2_oof, "measured ChIP + Micro-C"),
    ]
    rows = []
    for name, score, cat in models:
        b = {bl: float(average_precision_score(y[(dist >= lo) & (dist < hi)], score[(dist >= lo) & (dist < hi)]))
             for bl, lo, hi in DISTANCE_BINS}
        d_vs_g = paired_bootstrap_delta_ap(y, score, enf["G"], gene, n_boot=1000, seed=7)
        rows.append({
            "name": name,
            "category": cat,
            "ap": float(average_precision_score(y, score)),
            "auc": float(roc_auc_score(y, score)),
            "bins": b,
            "delta_vs_enformer_G": d_vs_g,
        })

    non_k562 = d_he["cell"] != "K562"
    return {
        "k562_loco": rows,
        "p1_cnn_transfer": {
            "all5_ap": float(average_precision_score(d_he["y"], p1_cnn_tr)),
            "4non_ap": float(average_precision_score(d_he["y"][non_k562], p1_cnn_tr[non_k562])),
        },
        "preds": {"p1": p1_oof, "p0": p0_oof, "p2": p2_oof, "p1_cnn": p1_cnn_oof},
    }


def exp_quantitative_and_weighted(d_tr: dict, d_he: dict, raw_tr: pd.DataFrame, raw_he: pd.DataFrame, preds: dict) -> dict:
    """Weighted AP (`direct_vs_indirect_negative`) and quantitative EffectSize correlations."""
    w_tr = pd.to_numeric(raw_tr["direct_vs_indirect_negative"], errors="coerce").fillna(1.0).to_numpy(float)
    w_he = pd.to_numeric(raw_he["direct_vs_indirect_negative"], errors="coerce").fillna(1.0).to_numpy(float)
    eff_tr = -pd.to_numeric(raw_tr["EffectSize"], errors="coerce").to_numpy(float)  # positive = stronger knockdown
    eff_he = -pd.to_numeric(raw_he["EffectSize"], errors="coerce").to_numpy(float)

    res = {"k562": {}, "heldout_all5": {}, "heldout_4non": {}}
    for arm, p in [("P1_no_marks", preds["p1"]), ("P0_full_bio", preds["p0"]), ("P2_full_bio_3d", preds["p2"])]:
        pos_m = d_tr["y"] == 1
        dist_m = d_tr["dist"] >= 20e3
        valid_e = np.isfinite(eff_tr)
        res["k562"][arm] = {
            "ap_unweighted": float(average_precision_score(d_tr["y"], p)),
            "ap_weighted": float(average_precision_score(d_tr["y"], p, sample_weight=w_tr)),
            "spearman_all": float(spearmanr(p[valid_e], eff_tr[valid_e]).statistic),
            "spearman_pos": float(spearmanr(p[pos_m & valid_e], eff_tr[pos_m & valid_e]).statistic),
            "pearson_pos_logit": float(pearsonr(np.log(np.clip(p[pos_m & valid_e], 1e-6, 1 - 1e-6) / (1 - np.clip(p[pos_m & valid_e], 1e-6, 1 - 1e-6))), eff_tr[pos_m & valid_e]).statistic),
            "spearman_distal20k": float(spearmanr(p[dist_m & valid_e], eff_tr[dist_m & valid_e]).statistic),
        }

    non_k562 = d_he["cell"] != "K562"
    for arm, wm, w3d in [("P1_no_marks", False, False), ("P0_full_bio", True, False)]:
        ph = fit_transfer(d_tr, d_he, C=1.0, with_marks=wm, with_3d=w3d)
        ve = np.isfinite(eff_he)
        res["heldout_all5"][arm] = {
            "ap_unweighted": float(average_precision_score(d_he["y"], ph)),
            "ap_weighted": float(average_precision_score(d_he["y"], ph, sample_weight=w_he)),
            "spearman_all": float(spearmanr(ph[ve], eff_he[ve]).statistic),
        }
        res["heldout_4non"][arm] = {
            "ap_unweighted": float(average_precision_score(d_he["y"][non_k562], ph[non_k562])),
            "ap_weighted": float(average_precision_score(d_he["y"][non_k562], ph[non_k562], sample_weight=w_he[non_k562])),
            "spearman_all": float(spearmanr(ph[non_k562 & ve], eff_he[non_k562 & ve]).statistic),
        }
    return res


def exp_biological_explanations(d_tr: dict, d_he: dict, raw_tr: pd.DataFrame, raw_he: pd.DataFrame, preds: dict) -> dict:
    """Quantify biological mechanisms + classic locus case studies."""
    y = d_tr["y"]
    dist = d_tr["dist"]
    ubiq = raw_tr["measuredGeneUbiquitousExpressed"].astype(bool).to_numpy()
    p0, p1, p2 = preds["p0"], preds["p1"], preds["p2"]

    # 1) Housekeeping / Ubiquitous gene buffering (Fulco 2019; Bergman 2022)
    housekeeping = {}
    for label, m in [("ubiquitous", ubiq), ("non_ubiquitous", ~ubiq)]:
        m_dist = m & (dist >= 20e3)
        housekeeping[label] = {
            "n": int(m.sum()),
            "pos": int(y[m].sum()),
            "pos_rate": float(y[m].mean()),
            "n_distal20k": int(m_dist.sum()),
            "pos_distal20k": int(y[m_dist].sum()),
            "pos_rate_distal20k": float(y[m_dist].mean()),
            "p0_ap": float(average_precision_score(y[m], p0[m])),
            "p0_distal20k_ap": float(average_precision_score(y[m_dist], p0[m_dist])),
            "p1_ap": float(average_precision_score(y[m], p1[m])),
        }

    # 2) Intervening TSS & genome-wide competition & CTCF/H3K27me3 insulation in >=20kb distal pairs
    m20 = dist >= 20e3
    int_tss = np.expm1(d_tr["genomic_ctx"][:, 0])
    gw_share = np.exp(d_tr["genomic_ctx"][:, 6])
    h3k27me3 = d_tr["base12"][:, 6]
    ctcf = d_tr["base12"][:, 4]
    mc10_oe = np.expm1(d_tr["contact_3d"][:, 2])
    distal_mech = {}
    for grp_name, gm in [("distal_pos", m20 & (y == 1)), ("distal_neg", m20 & (y == 0))]:
        distal_mech[grp_name] = {
            "n": int(gm.sum()),
            "median_int_tss": float(np.median(int_tss[gm])),
            "mean_int_tss": float(np.mean(int_tss[gm])),
            "zero_int_tss_frac": float(np.mean(int_tss[gm] == 0)),
            "mean_gw_share": float(np.mean(gw_share[gm])),
            "mean_h3k27me3": float(np.mean(h3k27me3[gm])),
            "mean_ctcf": float(np.mean(ctcf[gm])),
            "mean_mc10_oe": float(np.mean(mc10_oe[gm])),
        }

    # 3) Standardized weights of P0_full_bio across 23 folds (for interpretability)
    fnames = feature_names(with_marks=True, with_bio=True, with_3d=False)
    Z_all, _ = _fit_transform_pair(d_tr, d_tr, with_marks=True, with_bio=True, with_3d=False)
    clf_all = LogisticRegression(C=1.0, max_iter=400).fit(Z_all, y)
    weights = sorted(zip(fnames, clf_all.coef_[0].tolist()), key=lambda x: abs(x[1]), reverse=True)

    # 4) Classic loci case studies (K562 LOCO OOF + heldout transfer)
    loci_k562 = {}
    for g in ["MYC", "GATA1", "HDAC6", "HBE1", "KLF1"]:
        gm = d_tr["gene"] == g
        if gm.sum() == 0 or y[gm].sum() == 0:
            continue
        pos_idx = np.where(gm & (y == 1))[0]
        loci_k562[g] = {
            "n": int(gm.sum()),
            "pos": int(y[gm].sum()),
            "p0_ap": float(average_precision_score(y[gm], p0[gm])),
            "p1_ap": float(average_precision_score(y[gm], p1[gm])),
            "p2_ap": float(average_precision_score(y[gm], p2[gm])),
            "d0_ap": float(average_precision_score(y[gm], loco_oof_dist_single(d_tr)[gm])),
            "top_positives": [
                {
                    "dist_bp": int(dist[i]),
                    "int_tss": int(round(int_tss[i])),
                    "mc10_oe": round(float(mc10_oe[i]), 3),
                    "p1": round(float(p1[i]), 3),
                    "p0": round(float(p0[i]), 3),
                    "p2": round(float(p2[i]), 3),
                }
                for i in pos_idx[:5]
            ],
        }

    # Check MYC & CCND1 in heldout_5ct (HCT116)
    p0_he = fit_transfer(d_tr, d_he, C=1.0, with_marks=True, with_bio=True, with_3d=False)
    p1_he = fit_transfer(d_tr, d_he, C=1.0, with_marks=False, with_bio=True, with_3d=False)
    p2_he = fit_transfer(d_tr, d_he, C=1.0, with_marks=True, with_bio=True, with_3d=True)
    loci_held = {}
    for g, cl in [("MYC", "HCT116"), ("CCND1", "HCT116")]:
        gm = (d_he["gene"] == g) & (d_he["cell"] == cl)
        if gm.sum() > 0 and d_he["y"][gm].sum() > 0:
            loci_held[f"{g}_{cl}"] = {
                "n": int(gm.sum()),
                "pos": int(d_he["y"][gm].sum()),
                "p1_ap": float(average_precision_score(d_he["y"][gm], p1_he[gm])),
                "p0_ap": float(average_precision_score(d_he["y"][gm], p0_he[gm])),
                "p2_ap": float(average_precision_score(d_he["y"][gm], p2_he[gm])),
            }

    return {
        "housekeeping_buffering": housekeeping,
        "distal_mechanisms": distal_mech,
        "top_weights_p0": weights[:15],
        "loci_k562": loci_k562,
        "loci_heldout": loci_held,
    }


def loco_oof_dist_single(d_tr):
    from linearbaseline.fullbio import loco_oof_distance
    return loco_oof_distance(d_tr, C=1.0)


def print_threshold_tables(loco_json: dict, tr_json: dict) -> None:
    print("\n=== FULL 9-THRESHOLD TABLES (K562 LOCO) ===")
    for arm in ("P2_full_bio_3d", "P0_full_bio", "P1_no_marks"):
        print(f"\n[{arm}]")
        print(f"{'p>=':>5s} {'calls':>6s} {'TP':>4s} {'FP':>4s} {'prec':>7s} {'rec':>7s} {'spec':>7s} {'F1':>7s}")
        for r in loco_json["arms"][arm]["all"]["thresholds"]:
            print(f"{r['threshold']:5.2f} {r['flagged']:6d} {r['tp']:4d} {r['fp']:4d} {r['precision']:7.4f} {r['recall']:7.4f} {r['specificity']:7.4f} {r['f1']:7.4f}")

    print("\n=== FULL 9-THRESHOLD TABLES (TRANSFER: ALL-5 & 4-NON-K562) ===")
    for arm in ("P0_full_bio", "P1_no_marks"):
        for sname, rows in [("all5", tr_json["arms"][arm]["all"]["thresholds"]), ("4nonK562", tr_json["arms"][arm]["non_k562"]["all"]["thresholds"])]:
            print(f"\n[{arm} - {sname}]")
            print(f"{'p>=':>5s} {'calls':>6s} {'TP':>4s} {'FP':>4s} {'prec':>7s} {'rec':>7s} {'spec':>7s} {'F1':>7s}")
            for r in rows:
                print(f"{r['threshold']:5.2f} {r['flagged']:6d} {r['tp']:4d} {r['fp']:4d} {r['precision']:7.4f} {r['recall']:7.4f} {r['specificity']:7.4f} {r['f1']:7.4f}")


def main():
    d_tr = load_fullbio(str(ROOT / "data/fullbio/features.npz"))
    d_he = load_fullbio(str(ROOT / "data/fullbio/heldout_5ct.npz"))
    raw_tr = pd.read_csv(RAW_TR, sep="\t", low_memory=False)
    raw_tr = raw_tr[raw_tr["ValidConnection"].astype(bool)].reset_index(drop=True)
    raw_he = pd.read_csv(RAW_HE, sep="\t", low_memory=False)
    raw_he = raw_he[raw_he["ValidConnection"].astype(bool)].reset_index(drop=True)

    loco_json = json.loads((ROOT / "results/fullbio_loco.json").read_text(encoding="utf-8"))
    tr_json = json.loads((ROOT / "results/heldout_transfer.json").read_text(encoding="utf-8"))
    print_threshold_tables(loco_json, tr_json)

    print("\n=== EXP A: NESTED-CV PLATT & ISOTONIC RECALIBRATION ===")
    recal = exp_recalibration(d_tr, d_he)
    for arm, d in recal.items():
        print(f"\n[{arm}]")
        for tag, r in d.items():
            print(
                f"  {tag:9s}: AP={r['loco_ap']:.4f} Brier={r['brier']:.4f} ECE={r['ece']:.4f} "
                f"Top={r['top_pred']:.3f}/{r['top_obs']:.3f} | "
                f"LOCO@90={r['loco_90']['flagged']:2d}({r['loco_90']['precision']:.3f}/{r['loco_90']['recall']:.3f}) | "
                f"TR5@90={r['tr5_90']['flagged']:2d}({r['tr5_90']['precision']:.3f}) | "
                f"TR4@90={r['tr4_90']['flagged']:2d}({r['tr4_90']['precision']:.3f})"
            )

    print("\n=== EXP B: HEAD-TO-HEAD ON IDENTICAL 10,356 K562 23-FOLD LOCO SPLIT ===")
    h2h = exp_head_to_head(d_tr, d_he)
    for r in h2h["k562_loco"]:
        dv = r["delta_vs_enformer_G"]
        b = r["bins"]
        print(
            f"  {r['name']:34s} | AP={r['ap']:.4f} AUC={r['auc']:.4f} | "
            f"<20={b['<20kb']:.4f} 20-100={b['20-100kb']:.4f} 100-1M={b['100kb-1Mb']:.4f} >=1M={b['>=1Mb']:.4f} | "
            f"vs EnfG={dv['delta']:+.4f} [{dv['ci_low']:+.4f},{dv['ci_high']:+.4f}]"
        )
    print("  P1_no_marks+1kbCNN transfer:", h2h["p1_cnn_transfer"])

    print("\n=== EXP C: WEIGHTED AP & QUANTITATIVE EFFECT SIZE ===")
    quant = exp_quantitative_and_weighted(d_tr, d_he, raw_tr, raw_he, h2h["preds"])
    print(json.dumps(quant, indent=2))

    print("\n=== EXP D: BIOLOGICAL MECHANISMS & CLASSIC LOCI ===")
    bio = exp_biological_explanations(d_tr, d_he, raw_tr, raw_he, h2h["preds"])
    print("Housekeeping buffering:", json.dumps(bio["housekeeping_buffering"], indent=2))
    print("Distal (>=20kb) mechanisms:", json.dumps(bio["distal_mechanisms"], indent=2))
    print("Top-15 standardized weights (P0):", [(n, round(w, 4)) for n, w in bio["top_weights_p0"]])
    print("Classic K562 loci:", json.dumps(bio["loci_k562"], indent=2))
    print("Classic Heldout loci:", json.dumps(bio["loci_heldout"], indent=2))

    # Save summary JSON (compact, no large arrays)
    summary = {
        "recalibration": recal,
        "head_to_head": h2h["k562_loco"],
        "p1_cnn_transfer": h2h["p1_cnn_transfer"],
        "quantitative_and_weighted": quant,
        "biological_explanations": bio,
    }
    out_path = ROOT / "results/publication_suite.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
