# -*- coding: utf-8 -*-
"""Repackage the EPCrisprBenchmark K562 full-bio table into ``features.npz``.

This is the *only* code between the upstream benchmark and the matrix the
repository trains on.  It does not compute features; it slices the 12 design
columns from the sequence table, keeps the 128 JASPAR motif scores as-is, and
joins four auxiliary tables.  Every transformation after this point (PCA,
medians, standardisation) happens inside the leave-one-chromosome-out loop.

Inputs (none of them are redistributed here):

``--x``      ``epcrispr_seq_train/X.npy``  (n, 140) float32
             [12 design columns | 128 JASPAR motif scores]
``--meta``   ``epcrispr_seq_train/meta.csv``
             pair_id, state_idx, gene, d, contact, atac, mask, y, split
``--raw-tsv`` ``EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz``
             only ``chrom`` is used here; it is asserted row-aligned with meta
``--ctx``    ``results/ctx_feats.npz``  keys ``bio`` (n,4), ``gene`` (n,5)
``--expr``   ``results/gene_expr_sym.csv``  columns ``gene``, ``log_expr``

Example::

    python scripts/build_fullbio_npz.py \
        --x      phase3/realdata/epcrispr_seq_train/X.npy \
        --meta   phase3/realdata/epcrispr_seq_train/meta.csv \
        --raw-tsv data/raw/rE2G/training_K562/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz \
        --ctx    phase3/realdata/results/ctx_feats.npz \
        --expr   phase3/realdata/results/gene_expr_sym.csv \
        --out    data/fullbio/features.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--raw-tsv", required=True)
    p.add_argument("--ctx", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--out", default="data/fullbio/features.npz")
    args = p.parse_args(argv)

    X = np.load(args.x).astype(np.float32)
    meta = pd.read_csv(args.meta)
    raw = pd.read_csv(args.raw_tsv, sep="\t")
    if len(raw) != len(meta):
        raise SystemExit(f"row mismatch: raw={len(raw)} meta={len(meta)}")
    if not np.allclose(np.abs(raw["distanceToTSS"].to_numpy(float)), meta["d"].to_numpy(float)):
        raise SystemExit("distanceToTSS and meta.d disagree; the tables are not row-aligned")
    if X.shape[1] != 140:
        raise SystemExit(f"expected 140 columns (12 design + 128 motif), got {X.shape[1]}")

    ctx = np.load(args.ctx)
    expr = pd.read_csv(args.expr).set_index("gene")["log_expr"]
    ev = expr.reindex(meta["gene"]).to_numpy(np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        base12=X[:, :12],
        motif=X[:, 12:],
        bio=ctx["bio"].astype(np.float32),
        gene_ctx=ctx["gene"].astype(np.float32),
        expr=ev.astype(np.float32),
        atac=meta["atac"].to_numpy(np.float32),
        dist=meta["d"].to_numpy(np.float64),
        y=meta["y"].to_numpy(np.int8),
        chrom=np.asarray(raw["chrom"].astype(str).tolist(), dtype="U"),
        gene=np.asarray(meta["gene"].astype(str).tolist(), dtype="U"),
    )
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e6, 2), "MB")
    print(json.dumps({"n": int(len(meta)), "pos": int(meta["y"].sum()),
                      "chroms": int(raw["chrom"].nunique()),
                      "genes": int(meta["gene"].nunique())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
