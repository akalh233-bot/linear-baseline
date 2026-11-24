# -*- coding: utf-8 -*-
"""Repackage the held-out five-cell-type EPCrispr table into ``heldout_5ct.npz``.

Builds the exact same feature blocks as ``data/fullbio/features.npz`` for the
4,378-pair held-out five-cell-type table (K562, WTC11, HCT116, GM12878, Jurkat),
with strict row-by-row alignment assertions and zero target-label usage.

Example::

    python scripts/build_heldout_npz.py \
        --x       ../phase3/realdata/epcrispr_seq_held/X.npy \
        --meta    ../phase3/realdata/epcrispr_seq_held/meta.csv \
        --raw-tsv ../data/raw/rE2G/heldout_5_cell_types/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz \
        --ctx     ../phase3/realdata/results/e2g_gene_ctx.csv \
        --expr    ../phase3/realdata/results/gene_expr_sym.csv \
        --tss     ../tools/tss_prot.npz \
        --ccre    ../refs/cCRE_v3.bed.gz \
        --hic     ../data/raw/hic/K562_MicroC_hg38.hic \
        --out     data/fullbio/heldout_5ct.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _genomic_builders import (  # noqa: E402
    compute_genomic_ctx,
    compute_raw_extra,
    extract_microc_contacts,
    load_genomic_intervals,
)


def neighbourhood_bio(dhs: np.ndarray, k27: np.ndarray, gene: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Leave-one-out neighbourhood summary, computed per cell type.

    Columns match ``ctx_feats.npz['bio']``:
      0: mean accessibility (log1p(log1p(DHS.RPM)) legacy scale) of gene's other tested elements
      1: fraction of other tested elements with value > 1
      2: mean H3K27ac of gene's other tested elements (ChIP mark; excluded from P1_no_marks)
      3: max(H3K27ac across gene's tested elements) - own H3K27ac (ChIP mark; excluded from P1_no_marks)
    """
    dhs = np.asarray(dhs, dtype=np.float64)
    k27 = np.asarray(k27, dtype=np.float64)
    bio = np.zeros((len(dhs), 4), dtype=np.float64)
    for label in np.unique(cell):
        m = cell == label
        frame = pd.DataFrame({"gene": gene[m], "dhs": dhs[m], "k27": k27[m]})
        grouped = frame.groupby("gene", sort=False)
        n = grouped["dhs"].transform("size").to_numpy(float)
        sum_dhs = grouped["dhs"].transform("sum").to_numpy(float)
        sum_over = grouped["dhs"].transform(lambda s: float((s > 1.0).sum())).to_numpy(float)
        sum_k27 = grouped["k27"].transform("sum").to_numpy(float)
        max_k27 = grouped["k27"].transform("max").to_numpy(float)
        denom = np.maximum(n - 1, 1)
        bio[m] = np.column_stack(
            [
                (sum_dhs - dhs[m]) / denom,
                (sum_over - (dhs[m] > 1.0)) / denom,
                (sum_k27 - k27[m]) / denom,
                max_k27 - k27[m],
            ]
        )
    return bio


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--raw-tsv", required=True)
    p.add_argument("--ctx", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--tss", required=True)
    p.add_argument("--ccre", required=True)
    p.add_argument("--hic", required=True)
    p.add_argument("--out", default="data/fullbio/heldout_5ct.npz")
    args = p.parse_args(argv)

    X = np.load(args.x).astype(np.float32)
    meta = pd.read_csv(args.meta)
    raw = pd.read_csv(args.raw_tsv, sep="\t", low_memory=False)
    raw = raw[raw["ValidConnection"].astype(bool)].reset_index(drop=True)
    if X.shape[1] != 140:
        raise SystemExit(f"expected 140 columns (12 design + 128 motif), got {X.shape[1]}")
    if len(raw) != len(meta) or len(meta) != len(X):
        raise SystemExit(f"row mismatch: raw={len(raw)} meta={len(meta)} x={len(X)}")
    gene = meta["gene"].astype(str).to_numpy()
    if not np.array_equal(raw["measuredGeneSymbol"].astype(str).to_numpy(), gene):
        raise SystemExit("measuredGeneSymbol and meta.gene disagree; tables are not row-aligned")
    if not np.allclose(np.abs(raw["distanceToTSS"].to_numpy(float)), meta["d"].to_numpy(float)):
        raise SystemExit("distanceToTSS and meta.d disagree; tables are not row-aligned")
    y = meta["y"].to_numpy(np.int8)
    if not np.array_equal(raw["Regulated"].astype(bool).to_numpy(), y.astype(bool)):
        raise SystemExit("Regulated and meta.y disagree")

    expected_atac = np.log1p(np.clip(raw["DHS.RPM"].to_numpy(float), 0.0, None))
    if not np.allclose(meta["atac"].to_numpy(float), expected_atac, atol=1e-4):
        raise SystemExit("meta.atac does not equal log1p(DHS.RPM); check upstream builder")

    cell = np.asarray(raw["CellType"].astype(str).tolist(), dtype="U")
    dhs = np.log1p(meta["atac"].to_numpy(np.float64))
    k27 = X[:, 2].astype(np.float64)
    bio = neighbourhood_bio(dhs, k27, gene, cell)

    gc = pd.read_csv(args.ctx).set_index("gene")
    gc_cols = ["prom_dnase", "n_cand_elem", "n_tss", "sum_nearby_enh", "ubiq_frac"]
    gene_ctx = gc.reindex(gene)[gc_cols].to_numpy(np.float32)

    expr = pd.read_csv(args.expr).set_index("gene")["log_expr"]
    ev = expr.reindex(gene).to_numpy(np.float32)

    tss_map, ccre_map = load_genomic_intervals(args.tss, args.ccre)
    raw_extra = compute_raw_extra(raw)
    genomic_ctx = compute_genomic_ctx(raw, tss_map, ccre_map)
    contact_3d = extract_microc_contacts(raw, args.hic)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        base12=X[:, :12],
        motif=X[:, 12:],
        bio=bio.astype(np.float32),
        gene_ctx=gene_ctx,
        expr=ev,
        atac=meta["atac"].to_numpy(np.float32),
        dist=meta["d"].to_numpy(np.float64),
        raw_extra=raw_extra,
        genomic_ctx=genomic_ctx,
        contact_3d=contact_3d,
        y=y,
        chrom=np.asarray(raw["chrom"].astype(str).tolist(), dtype="U"),
        gene=np.asarray(gene.tolist(), dtype="U"),
        cell=cell,
    )
    run = pd.Series(cell).value_counts().to_dict()
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e6, 2), "MB")
    print(
        json.dumps(
            {
                "n": int(len(meta)),
                "pos": int(y.sum()),
                "cells": {k: int(v) for k, v in sorted(run.items())},
                "gene_ctx_missing": int(np.isnan(gene_ctx).any(1).sum()),
                "expr_missing": int(np.isnan(ev).sum()),
                "mc_10kb_nonzero": round(float((contact_3d[:, 1] > 0).mean()), 4),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
