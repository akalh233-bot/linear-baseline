# Data: the packaged full-bio matrices (v0.4.0)

Two derived tables are shipped, both with identical feature blocks:

* `data/fullbio/features.npz` (3.71 MB) — EPCrisprBenchmark K562 (10,356 pairs / 471 positives / 2,113 genes / 23 chromosomes).
* `data/fullbio/heldout_5ct.npz` (1.05 MB) — held-out five-cell-type table (4,378 pairs / 190 positives; 1,918 in-domain K562 [118 pos] + 2,460 across 4 non-K562 cell types [72 pos]).

## SHA-256 checksums

| file | SHA-256 | rows | positives |
| --- | --- | ---: | ---: |
| `data/fullbio/features.npz` | `5A7370EE7A0A08A39B0CB87B83E6F8C95D2350F1450389B7AE770809DA83AD1D` | 10,356 | 471 (4.55%) |
| `data/fullbio/heldout_5ct.npz` | `BB04A0244366E60AE8509C261DF8DC892479125A03F9B225F902E68CE1F56EC5` | 4,378 | 190 (4.34%) |

## Keys and provenance

| npz key | shape | description | ChIP? |
| --- | --- | --- | --- |
| `base12` | `(n, 12)` | `[logd, logd2, h3k27ac, h3k27ac_pct, ctcf, h3k4me1, h3k27me3, 4 peak flags, ubiq_expr]` (`base12[:, 11]` unused) | cols `2..10` |
| `motif` | `(n, 128)` | JASPAR2022 motif scores over 451 bp CRE window (scaled + PCA-4 fitted in-fold) | no |
| `atac` | `(n,)` | `log1p(DHS.RPM)` (already log-transformed upstream; used directly without double `log1p` since v0.4.0) | no |
| `bio` | `(n, 4)` | Per-cell-type leave-one-out over each gene's *tested* candidate elements: `[mean_dhs, frac_dhs_gt1, mean_h3k27ac, max_h3k27ac - own_h3k27ac]`. Singleton genes (`n=1`: 9.0% K562) collapse to 0. **Note**: cols 2–3 are measured H3K27ac and are strictly excluded from `P1_no_marks`; `nobio` arms omit all 4 to eliminate screen-design dependence. | cols `2..3` |
| `gene_ctx` | `(n, 5)` | Per-gene mean of `normalizedDNase_prom`, `numCandidateEnhGene`, `numTSSEnhGene`, `sumNearbyEnhancers`, `ubiquitousExpressedGene` from ENCODE-rE2G K562 BED (`ENCFF970QAX`). **Provenance note (v0.4.0 audit)**: `prom_dnase` and `ubiq_frac` are true gene-level constants, whereas `numCandidateEnhGene`, `numTSSEnhGene`, and `sumNearbyEnhancers` are pair-level in the source BED and were averaged per gene in `build_e2g_ctx.py`. True pair-level genomic counts are supplied cleanly in `genomic_ctx`. | no |
| `expr` | `(n,)` | Gasperini et al. control-gRNA GLM intercept (`log_expr`); median-imputed and z-scored strictly in-fold | no |
| `raw_extra` | `(n, 5)` | Unused label-free columns already present in both EPCrisprBenchmark TSVs: `[DHS.percentile, distanceToTSS < 0, log1p(width), log1p(H3K27ac.RPM.expandedRegion), H3K27ac.expandedRegion.percentile]`. | cols `3..4` |
| `genomic_ctx` | `(n, 7)` | Pure hg38 genome annotation (zero ChIP, zero cell-type dependence, zero screen-design dependence) from `tools/tss_prot.npz` (20,046 protein-coding TSSs) and `refs/cCRE_v3.bed.gz`: `[log1p(intervening_tss), log1p(intervening_ccre), log1p(ccre_5kb), log1p(ccre_50kb), log1p(tss_50kb), log(gw_ccre_powerlaw_sum_500kb), log(gw_ccre_powerlaw_share)]` with $\gamma = 0.87$ (Fulco et al. 2019). | no |
| `contact_3d` | `(n, 4)` | VC_SQRT-balanced K562 Micro-C (`data/raw/hic/K562_MicroC_hg38.hic`, 3×3 smoothed): `[log1p(5kb), log1p(10kb), log1p(10kb_O/E), log1p(25kb_O/E)]`. Used only in `P2_full_bio_3d`. | no (3D) |
| `y`, `chrom`, `gene`, (`cell`) | `(n,)` | `Regulated` label (0/1), chromosome, gene symbol, and (in `heldout_5ct.npz`) cell type. | – |

## Regeneration commands

Both scripts assert exact row-by-row alignment on `measuredGeneSymbol`, `|distanceToTSS|`, `Regulated`, `ValidConnection`, and `atac == log1p(DHS.RPM)` across all chromosomes before writing:

```bash
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
```
