# -*- coding: utf-8 -*-
"""Repackage the held-out five-cell-type EPCrispr table into ``heldout_5ct.npz``.

This is the target table for the zero-shot transfer arm: the same 26 columns as
``data/fullbio/features.npz``, built row by row from the held-out benchmark, with
the target cell line's *measured* marks and accessibility (label-free) and the
K562-derived gene-side / expression proxies that the model was trained with.
No label of the target cell line is used to build a column.

Inputs (none of them are redistributed here):

``--x``       ``epcrispr_seq_held/X.npy``  (n, 140) float32
              [12 design columns | 128 JASPAR motif scores]
``--meta``    ``epcrispr_seq_held/meta.csv``
              pair_id, state_idx, gene, d, contact, atac, mask, y, split
``--raw-tsv`` ``EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz``
              supplies ``chrom``, ``CellType`` and the label; filtered to
              ``ValidConnection == True`` and asserted row-aligned with meta
``--ctx``     ``results/e2g_gene_ctx.csv``  per-gene gene-side block (5 columns)
``--expr``    ``results/gene_expr_sym.csv``  ``gene``, ``log_expr``

The neighbourhood block (``bio``, 4 columns) is recomputed here, exactly as
``ctx_eval.py`` does for the training table but **within each target cell type**:
for each gene, the CRE's own row is removed from the mean/fraction/max of the
other candidate elements of that gene.  That keeps the column label-free while
letting the model see the target cell line's chromatin context, which is the
honest zero-shot deployment: marks and accessibility are measured, labels are
not.

Example::

    python scripts/build_heldout_npz.py \\
        --x      phase3/realdata/epcrispr_seq_held/X.npy \\
        --meta   phase3/realdata/epcrispr_seq_held/meta.csv \\
        --raw-tsv data/raw/rE2G/heldout_5_cell_types/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz \\
        --ctx    phase3/realdata/results/e2g_gene_ctx.csv \\
        --expr   phase3/realdata/results/gene_expr_sym.csv \\
        --out    data/fullbio/heldout_5ct.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def neighbourhood_bio(dhs: np.ndarray, k27: np.ndarray, gene: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Leave-one-out neighbourhood summary, computed per cell type.

    Columns match ``ctx_feats.npz['bio']``:
    mean accessibility of the gene's other elements, fraction of those with
    accessibility > 1, mean H3K27ac, and max H3K27ac minus the row's own value.
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
        raise SystemExit("measuredGeneSymbol and meta.gene disagree; the tables are not row-aligned")
    if not np.allclose(np.abs(raw["distanceToTSS"].to_numpy(float)), meta["d"].to_numpy(float)):
        raise SystemExit("distanceToTSS and meta.d disagree; the tables are not row-aligned")
    y = meta["y"].to_numpy(np.int8)
    if not np.array_equal(raw["Regulated"].astype(bool).to_numpy(), y.astype(bool)):
        raise SystemExit("Regulated and meta.y disagree")

    cell = np.asarray(raw["CellType"].astype(str).tolist(), dtype="U")
    dhs = np.log1p(meta["atac"].to_numpy(np.float64))
    k27 = X[:, 2].astype(np.float64)  # H3K27ac RPM, the design column ctx_eval uses
    bio = neighbourhood_bio(dhs, k27, gene, cell)

    gc = pd.read_csv(args.ctx).set_index("gene")
    gc_cols = ["prom_dnase", "n_cand_elem", "n_tss", "sum_nearby_enh", "ubiq_frac"]
    gene_ctx = gc.reindex(gene)[gc_cols].to_numpy(np.float32)

    expr = pd.read_csv(args.expr).set_index("gene")["log_expr"]
    ev = expr.reindex(gene).to_numpy(np.float32)

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
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
