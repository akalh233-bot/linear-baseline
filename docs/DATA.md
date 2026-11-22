# Data: the packaged full-bio matrices

Two derived tables are shipped, both with the same 26-column definition:

* `data/fullbio/features.npz` (3.4 MB) — the K562 training/evaluation table the
  headline LOCO numbers come from.
* `data/fullbio/heldout_5ct.npz` (0.9 MB) — the held-out five-cell-type table
  the zero-shot transfer arm is scored on.

They are shipped because the repository's headline numbers must be reproducible
without re-deriving anything from raw genotypes; the upstream data itself is
**not** redistributed.

## 1. `features.npz` — K562, 10,356 pairs

10,356 candidate (CRE, gene) pairs from the EPCrisprBenchmark K562 set, 471
positives (4.55%), 2,113 genes, 23 chromosomes.

| field | value |
| --- | --- |
| SHA-256 | `5D9B1EB29E1F31DD7E60CC4975CE46B8F637F162654BBCAFE644B96642F09748` |
| rows × columns | 10,356 × 26 (+ labels, chromosome, gene) |
| positives | 471 (4.55%) |
| genes | 2,113 (4.9 pairs per gene on average) |
| chromosomes | 23, used one at a time as the held-out fold |

## Upstream sources

1. **EPCrisprBenchmark, training_K562, GRCh38** — `Regulated` is the label and
   `ValidConnection` is the inclusion filter (`build_epcrispr.py` keeps rows
   with `ValidConnection == True`). Only this file supplies `chrom` and the
   label.
2. **Sequence design table** `epcrispr_seq_train/X.npy`, 140 columns: the first
   12 are design columns (`FEATS` in `build_epcrispr.py`: `logd`, `logd2`,
   `h3k27ac`, `h3k27ac_pct`, `ctcf`, `h3k4me1`, `h3k27me3`, and four peak
   flags), the last 128 are JASPAR motif scores over the CRE window.
3. **`results/ctx_feats.npz`** — `bio` (4) and `gene` (5), built by
   `ctx_eval.py` from the ENCODE-rE2G K562 element–gene bed.
4. **`results/gene_expr_sym.csv`** — `gene`, `log_expr` (Gasperini et al. GLM
   expression prior).

`scripts/build_fullbio_npz.py` performs the join and asserts two invariants: the
row counts match, and `|distanceToTSS|` in the raw TSV equals `meta.d` for every
row. The committed file was produced by exactly that script.

## Columns

| block | npz key | dims | definition |
| --- | --- | ---: | --- |
| distance | `base12[:,0:2]` | 2 | `log1p(d)/10` and its square, `d = |distanceToTSS|` |
| measured marks | `base12[:,2:11]` | 9 | H3K27ac RPM, H3K27ac percentile, CTCF, H3K4me1, H3K27me3 RPM, and four binary peak-overlap flags |
| motif | `motif` | 128 | JASPAR motif scores on the CRE window, PCA-4 **in-fold** |
| accessibility | `atac` | 1 | DHS RPM (log1p applied at train time) |
| neighbourhood | `bio` | 4 | leave-one-out over the gene's other candidate elements: mean accessibility, fraction with accessibility > 1, mean H3K27ac, max H3K27ac |
| gene side | `gene_ctx` | 5 | per gene, from the rE2G bed: `prom_dnase`, `n_cand_elem`, `n_tss`, `sum_nearby_enh`, `ubiq_frac` |
| expression prior | `expr` | 1 | `log_expr` per gene, z-scored globally at train time |
| label | `y` | 1 | `Regulated` (0/1) |
| grouping | `chrom`, `gene` | – | held-out fold and bootstrap cluster |

`base12[:,11]` (`ubiq_expr`) is shipped for completeness and is **not** used by
any reported arm.

Missing values: `gene_ctx` has 325 NaN entries and `expr` has 197, all from
genes absent from the auxiliary tables. They are imputed with **training-fold
medians** (gene side) or the global median (expression), never with test-fold
information.

## 2. `heldout_5ct.npz` — five cell types, 4,378 pairs

| field | value |
| --- | --- |
| SHA-256 | `2176953DCD16024A3F9CB6FC2717486E134F12A9CCBF9940924D4676F5C1E206` |
| rows × columns | 4,378 × 26 (+ labels, chromosome, gene, cell) |
| positives | 190 (4.34%) |
| cells | K562 1,918 (118 pos), WTC11 1,921 (15), HCT116 396 (34), Jurkat 75 (7), GM12878 68 (16) |

Built by `scripts/build_heldout_npz.py`, which asserts that the filtered raw TSV
(`ValidConnection == True`), the meta table and `X.npy` are row-aligned on gene,
`|distanceToTSS|`, and label. Keys are the same as `features.npz` plus `cell`.

Two columns are re-derived for this table and are worth calling out:

* **`bio` (neighbourhood, 4)** — recomputed *within each cell type*, with the
  same leave-one-out-over-the-gene's-other-elements formula as `ctx_eval.py`:
  mean accessibility of the other CREs, fraction of them with accessibility > 1,
  mean H3K27ac, and max H3K27ac minus the row's own. It uses the target cell
  line's ATAC/H3K27ac measurements but no labels.
* **`gene_ctx` (5) and `expr` (1)** — reindexed from the K562-derived rE2G and
  expression tables, i.e. they are **proxies**, not target-cell measurements.
  Coverage: 79 of 4,378 rows miss at least one gene-context value, 209 rows miss
  the expression prior. They are imputed at transfer time with K562 training
  medians, exactly as in LOCO.

Measured marks, accessibility and distance are the target cell line's own
columns; labels are used only for scoring. The model is never refitted on this
table.

Regeneration (same invariants as the K562 table, plus the cell-type column):

```bash
python scripts/build_heldout_npz.py \
  --x      phase3/realdata/epcrispr_seq_held/X.npy \
  --meta   phase3/realdata/epcrispr_seq_held/meta.csv \
  --raw-tsv data/raw/rE2G/heldout_5_cell_types/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz \
  --ctx    phase3/realdata/results/e2g_gene_ctx.csv \
  --expr   phase3/realdata/results/gene_expr_sym.csv \
  --out    data/fullbio/heldout_5ct.npz
```

## 3. Regeneration of the K562 table

```bash
python scripts/build_fullbio_npz.py \
  --x      phase3/realdata/epcrispr_seq_train/X.npy \
  --meta   phase3/realdata/epcrispr_seq_train/meta.csv \
  --raw-tsv data/raw/rE2G/training_K562/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz \
  --ctx    phase3/realdata/results/ctx_feats.npz \
  --expr   phase3/realdata/results/gene_expr_sym.csv \
  --out    data/fullbio/features.npz
```

The script aborts if the tables are not row-aligned or if `X` is not 140 wide;
it never silently reorders rows. The committed file reproduced by exactly that
script hashes to
`5D9B1EB29E1F31DD7E60CC4975CE46B8F637F162654BBCAFE644B96642F09748`.

## Citation and licence for the data

* Gasperini et al., *Cell* 2019, doi:10.1016/j.cell.2019.11.001 (CRISPRi
  enhancer–gene map, EPCrisprBenchmark labels).
* Fulco et al., *Nature Genetics* 2019, doi:10.1038/s41588-019-0537-1.
* ENCODE-rE2G (ENCODE Consortium) for the element–gene bed.

The packaged matrix is a derived product distributed under this repository's
MIT licence; the underlying measurements remain subject to the upstream terms.
If you redistribute it, keep this file with it so the provenance travels.
