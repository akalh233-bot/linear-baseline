# Data: the packaged full-bio matrix

`data/fullbio/features.npz` (3.4 MB, committed) is a **derived** table: 10,356
candidate (CRE, gene) pairs from the EPCrisprBenchmark K562 set, 471 positives
(4.55%), 2,113 genes, 23 chromosomes. It is shipped because the repository's
headline number must be reproducible without re-deriving anything from raw
genotypes; the upstream data itself is **not** redistributed.

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

## Regeneration

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
it never silently reorders rows.

## Citation and licence for the data

* Gasperini et al., *Cell* 2019, doi:10.1016/j.cell.2019.11.001 (CRISPRi
  enhancer–gene map, EPCrisprBenchmark labels).
* Fulco et al., *Nature Genetics* 2019, doi:10.1038/s41588-019-0537-1.
* ENCODE-rE2G (ENCODE Consortium) for the element–gene bed.

The packaged matrix is a derived product distributed under this repository's
MIT licence; the underlying measurements remain subject to the upstream terms.
If you redistribute it, keep this file with it so the provenance travels.
