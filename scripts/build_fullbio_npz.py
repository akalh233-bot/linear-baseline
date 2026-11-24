# -*- coding: utf-8 -*-
"""Repackage the EPCrisprBenchmark K562 full-bio table into ``features.npz``.

Joins the upstream K562 benchmark, 140-column sequence/mark design table,
gene-level context, expression prior, unused label-free TSV columns, pure hg38
genomic-topology features (intervening TSS/cCRE + genome-wide power-law cCRE
competition), and K562 Micro-C 3D contact features.  Asserts strict row-by-row
alignment on gene, |distanceToTSS|, Regulated, and ValidConnection across all
23 chromosomes before writing.

Example::

    python scripts/build_fullbio_npz.py \
        --x       ../phase3/realdata/epcrispr_seq_train/X.npy \
        --meta    ../phase3/realdata/epcrispr_seq_train/meta.csv \
        --raw-tsv ../data/raw/rE2G/training_K562/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz \
        --ctx     ../phase3/realdata/results/ctx_feats.npz \
        --expr    ../phase3/realdata/results/gene_expr_sym.csv \
        --tss     ../tools/tss_prot.npz \
        --ccre    ../refs/cCRE_v3.bed.gz \
        --hic     ../data/raw/hic/K562_MicroC_hg38.hic \
        --out     data/fullbio/features.npz
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--raw-tsv", required=True)
    p.add_argument("--ctx", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--tss", required=True, help="hg38 protein-coding TSS npz (e.g. tools/tss_prot.npz)")
    p.add_argument("--ccre", required=True, help="ENCODE cCRE v3 bed.gz (e.g. refs/cCRE_v3.bed.gz)")
    p.add_argument("--hic", required=True, help="K562 Micro-C .hic v9 file")
    p.add_argument("--out", default="data/fullbio/features.npz")
    args = p.parse_args(argv)

    X = np.load(args.x).astype(np.float32)
    meta = pd.read_csv(args.meta)
    raw = pd.read_csv(args.raw_tsv, sep="\t", low_memory=False)
    raw = raw[raw["ValidConnection"].astype(bool)].reset_index(drop=True)

    # Strict row-by-row alignment assertions across all 23 chromosomes (AGENT_RULES §1)
    if len(raw) != len(meta) or len(meta) != len(X):
        raise SystemExit(f"row mismatch: raw={len(raw)} meta={len(meta)} x={len(X)}")
    if X.shape[1] != 140:
        raise SystemExit(f"expected 140 columns (12 design + 128 motif), got {X.shape[1]}")
    gene = meta["gene"].astype(str).to_numpy()
    if not np.array_equal(raw["measuredGeneSymbol"].astype(str).to_numpy(), gene):
        raise SystemExit("measuredGeneSymbol and meta.gene disagree; tables are not row-aligned")
    if not np.allclose(np.abs(raw["distanceToTSS"].to_numpy(float)), meta["d"].to_numpy(float)):
        raise SystemExit("distanceToTSS and meta.d disagree; tables are not row-aligned")
    y = meta["y"].to_numpy(np.int8)
    if not np.array_equal(raw["Regulated"].astype(bool).to_numpy(), y.astype(bool)):
        raise SystemExit("Regulated and meta.y disagree")

    # Verify meta.atac is already log1p(DHS.RPM) so downstream code never double-logs
    expected_atac = np.log1p(np.clip(raw["DHS.RPM"].to_numpy(float), 0.0, None))
    if not np.allclose(meta["atac"].to_numpy(float), expected_atac, atol=1e-4):
        raise SystemExit("meta.atac does not equal log1p(DHS.RPM); check upstream builder")

    ctx = np.load(args.ctx)
    if len(ctx["bio"]) != len(meta) or len(ctx["gene"]) != len(meta):
        raise SystemExit(f"ctx_feats length mismatch: {len(ctx['bio'])} vs {len(meta)}")

    expr = pd.read_csv(args.expr).set_index("gene")["log_expr"]
    ev = expr.reindex(meta["gene"]).to_numpy(np.float64)

    tss_map, ccre_map = load_genomic_intervals(args.tss, args.ccre)
    raw_extra = compute_raw_extra(raw)
    genomic_ctx = compute_genomic_ctx(raw, tss_map, ccre_map)
    contact_3d = extract_microc_contacts(raw, args.hic)

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
        raw_extra=raw_extra,
        genomic_ctx=genomic_ctx,
        contact_3d=contact_3d,
        y=y,
        chrom=np.asarray(raw["chrom"].astype(str).tolist(), dtype="U"),
        gene=np.asarray(gene.tolist(), dtype="U"),
    )
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e6, 2), "MB")
    print(json.dumps({
        "n": int(len(meta)),
        "pos": int(y.sum()),
        "chroms": int(raw["chrom"].nunique()),
        "genes": int(meta["gene"].nunique()),
        "mc_10kb_nonzero": round(float((contact_3d[:, 1] > 0).mean()), 4),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
