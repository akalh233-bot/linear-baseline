# linear-baseline

**A 26-column logistic regression that is the honest floor for CRE→gene
prediction — and the number any sequence model has to beat.**

On the EPCrisprBenchmark K562 CRE–gene task, under **leave-one-chromosome-out**
cross-validation, this baseline reaches **AP 0.6715** (95% CI
[0.6151, 0.7226], 10,356 pairs / 471 positives / 23 chromosomes). With the nine
*measured* chromatin marks removed — the deployable setting — it reaches
**0.6507** (CI [0.5937, 0.7020]). The gap, **+0.0208 AP**, is the entire value
of H3K27ac/CTCF/H3K4me1/H3K27me3 ChIP data for this task.

Ranking is not deployment. Reading a probability as a decision at **p ≥ 0.90**
gives 72 calls at precision 0.889 and recall 0.136; at **p ≥ 0.95** it gives 35
calls at precision 0.943 and recall 0.070. The same fitted model, applied
zero-shot to **five held-out cell types** (4,378 pairs / 190 positives), keeps
pooled **AP 0.6661** (CI [0.5878, 0.7392]) and makes 19 calls at p ≥ 0.90 with
precision 0.947 — but on WTC11 and Jurkat the high-confidence set is empty.

> 中文摘要：这是把 `phase3/realdata` 里那条 full-bio 逻辑回归正式打包成仓库。
> 主结果：LOCO AP **0.6715**，去掉 9 个实测标记后 **0.6507**；距离 <20 kb 的
> 子集 0.8327，而 100 kb–1 Mb 只有 0.2763；只给距离时远端只有 0.0186。
> 概率阈值已作为结果的一部分报告：p≥0.90 时 72 个 call、精确率 0.889、
> 召回 0.136；p≥0.95 时 35 个 call、精确率 0.943、召回 0.070。
> 零样本迁移到 5 个留出细胞系（4,378 对 / 190 正）池化 AP 0.6661，
> p≥0.90 的 19 个 call 精确率 0.947；但 WTC11 与 Jurkat 在该阈值下没有 call。

---

## Headline numbers

Task: given a candidate CRE and a candidate target gene, predict whether a
CRISPRi perturbation of the CRE changes that gene's expression. Labels:
EPCrisprBenchmark (Gasperini et al. 2019; Fulco et al. 2019), K562.

| arm | features | AP | 95% CI | AUC | <20 kb | 20–100 kb | 100 kb–1 Mb |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| **P0 full bio** | 26 | **0.6715** | [0.6151, 0.7226] | 0.9459 | **0.8327** | 0.6538 | **0.2763** |
| P1 no measured marks | 17 | 0.6507 | [0.5937, 0.7020] | 0.9440 | 0.8076 | 0.6533 | 0.2563 |
| D0 distance only | 2 | 0.4210 | [0.3666, 0.4803] | 0.8931 | 0.6335 | 0.2095 | 0.0186 |
| Δ (marks) | +9 | **+0.0208** | – | +0.0018 | +0.0251 | +0.0005 | +0.0200 |

Bin sizes: 421 pairs <20 kb (196 positives), 1,399 at 20–100 kb (193), 7,938 at
100 kb–1 Mb (76). The CI resamples **genes** (2,113 clusters, 1,000 resamples),
not rows: two candidate CREs of one gene are not independent evidence.

### Operating thresholds: what a p ≥ 0.90 call means

A ranking metric does not tell you what happens when the probability is turned
into a decision. The call set for the P0 arm (K562 LOCO, 471 positives):

| call rule | calls | TP | FP | precision | recall | specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p ≥ 0.50 | 299 | 232 | 67 | 0.7759 | 0.4926 | 0.9932 |
| p ≥ 0.75 | 155 | 133 | 22 | 0.8581 | 0.2824 | 0.9978 |
| p ≥ 0.90 | 72 | 64 | 8 | 0.8889 | 0.1359 | 0.9992 |
| p ≥ 0.95 | 35 | 33 | 2 | 0.9429 | 0.0701 | 0.9998 |
| p ≥ 0.99 | 2 | 2 | 0 | 1.0000 | 0.0042 | 1.0000 |

Read it as a precision/recall trade, not a guarantee. The top probability decile
(mean predicted 0.949) contains 72 pairs of which 64 are real (0.889): at
0.90 the model is roughly right about "9 in 10" and slightly over-confident at
the very top, while overall calibration is good (ECE 0.0032, Brier 0.0232 — the
ECE is dominated by the 9,443 pairs below 0.1). Specificity 0.9992 means the
0.90 rule leaves 99.92% of the 9,885 negatives alone, but recall 0.136 means it
misses six of every seven positives. A 0.90 threshold is a **high-precision
shortlist**, not a genome-wide call set; P1, without marks, has precision 0.9016
at the same cut, so the marks do not buy threshold precision, only ranking.

### Zero-shot transfer to five held-out cell types

The fitted K562 model — no retraining, no target labels — applied to
`EPCrisprBenchmark_combined_data.heldout_5_cell_types` (4,378 pairs, 190
positives, 0 pair overlap with training):

| cell type | n | positives | P0 AP | AUC | calls @ p ≥ 0.90 | precision @ 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **pooled** | 4,378 | 190 | **0.6661** | 0.9506 | 19 | 0.9474 |
| K562 | 1,918 | 118 | 0.7194 | 0.9386 | 14 | 1.0000 |
| WTC11 | 1,921 | 15 | 0.5026 | 0.9756 | 0 | – |
| HCT116 | 396 | 34 | 0.6090 | 0.9355 | 3 | 0.6667 |
| GM12878 | 68 | 16 | 0.8853 | 0.9471 | 2 | 1.0000 |
| Jurkat | 75 | 7 | 0.6187 | 0.9643 | 0 | – |

Pooled AP is essentially unchanged from the in-domain 0.6715 (0.6661; CI
[0.5878, 0.7392]), and even the no-marks P1 arm transfers at 0.6602
(+0.0059 for the marks, far less than the in-domain +0.0208). The pooled
p ≥ 0.90 call set is 19 pairs, 18 of them real (0.947) — but the calls
concentrate in K562/GM12878, and the two cell types with the fewest positives
(WTC11, Jurkat) produce **no** high-confidence call at all: the model is not
merely less precise there, it is silent. The per-cell APs rest on 7–118
positives, so the ranking of cell types (WTC11 0.50 vs GM12878 0.89) is a
hypothesis, not a stable ordering.

This transfer is *zero-shot with measured marks*: the target cell line supplies
its own ChIP/ATAC columns (label-free), the neighbourhood summary is recomputed
within that cell line, while the five gene-side columns and the expression
prior are K562-derived proxies. It tests whether the 26-column function
generalises, not whether a retrained state-specific model would.

### What the numbers actually say

1. **Distance alone gets 0.42 AP** — and nearly all of it in the near field
   (<20 kb: 0.63; 100 kb–1 Mb: 0.019). The <20 kb bin is at AP 0.83 while
   100 kb–1 Mb collapses to 0.28. A model that only learns "close to TSS"
   already gets most of the easy cases; the hard, interesting regime is distal,
   and there the full model is still only at 0.28 with 76 positives.
2. **Measured marks are worth 0.02 AP**, and essentially all of it is in the
   <20 kb and >100 kb bins, not in the 20–100 kb middle.
3. **The in-domain number does transfer, the confidence does not.** Pooled AP on
   five held-out cell types is 0.6661, but the p ≥ 0.90 call set shrinks from 72
   pairs in-domain to 19 pooled, and to zero in WTC11/Jurkat. Sparse-positive
   cell types lose the high-confidence tail before they lose ranking.
4. **A probability threshold is a trade, not a truth.** p ≥ 0.90 buys precision
   0.889 at recall 0.136; p ≥ 0.95 buys 0.943 at 0.070. The marks improve the
   ranking (+0.0208 AP) but not the precision of the 0.90 call set (P0 0.889
   vs P1 0.902), so they cannot be justified by the shortlist alone.
5. **This is still a K562-trained, single-label-system floor.** The gene-side
   features come from a candidate universe that is known not to transfer
   perfectly, and the transfer arm gives the target cell line its measured
   marks. Treat 0.6507 as an optimistic deployable bound.

## Reproduce

```bash
python -m pip install -e ".[dev]"

# the arm: LOCO CV + gene-cluster bootstrap  (~6 s)
linearbaseline fullbio --config configs/fullbio.json --out results

# zero-shot transfer of that fitted model to 5 held-out cell types (~5 s)
linearbaseline transfer --config configs/transfer.json --out results

pytest -q
```

`results/fullbio_loco.json` and `results/heldout_transfer.json` in this
repository are the committed outputs of those two commands; they contain every
number above plus the per-bin sample sizes, the threshold tables, and the
calibration bins. `tests/test_fullbio.py::test_readme_matches_committed_results`
and `tests/test_fullbio.py::test_readme_matches_committed_transfer_results` fail
if the README drifts from either file.

## The packaged design matrix

`data/fullbio/features.npz` (3.4 MB, committed) holds the 10,356 × 26 matrix and
its labels. Keys: `base12`, `motif`, `bio`, `gene_ctx`, `expr`, `atac`, `dist`,
`y`, `chrom`, `gene`. Column order is defined once, in
`linearbaseline.fullbio.feature_names()`:

| block | dims | columns | fitted where |
| --- | ---: | --- | --- |
| distance | 2 | `log1p(d)/10`, its square | nowhere (raw) |
| motif | 4 | PCA-4 of 128 JASPAR motif scores | in-fold |
| accessibility | 1 | `log1p(DHS RPM)` | nowhere (raw) |
| neighbourhood | 4 | chromatin summary of the CRE's neighbourhood | shipped |
| gene side | 5 | promoter/gene context from the rE2G bed | shipped (NaN→in-fold median) |
| expression prior | 1 | log expression prior, z-scored | global median + z |
| measured marks | 9 | H3K27ac, its percentile, CTCF, H3K4me1, H3K27me3, and four peak flags | shipped |

"Fitted where" is the leakage contract: PCA, gene-context medians, feature
standardisation and the logistic weights are all estimated on the 22 training
chromosomes and applied to the held-out one. Nothing about the test chromosome —
not even a column median — enters the fit. `docs/DATA.md` documents provenance
and the exact upstream files.

`data/fullbio/heldout_5ct.npz` (0.9 MB, committed) is the same 26 columns for
the 4,378-pair held-out five-cell-type table, built with an extra `cell` key.
Its `bio` block is recomputed within each target cell type (label-free), its
measured marks/ATAC are the target cell line's own measurements, and its
`gene_ctx`/`expr` columns are K562-derived proxies. `docs/DATA.md` gives the
regeneration command and SHA-256.

## Design decisions

* **Leave-one-chromosome-out, not random locus splits.** Random locus splits
  leave the model surrounded by the same chromatin environment; holding out a
  whole chromosome is the cheapest honest test of cross-region generalisation.
* **Gene-cluster bootstrap.** Rows sharing a gene are correlated; resampling
  rows would produce an interval roughly 1/√(rows per gene) too narrow.
* **In-fold PCA for the motif block.** 128 correlated JASPAR scores would
  otherwise let the model fit fold-specific motif noise; PCA-4 fitted on
  training chromosomes only is what makes the 26 columns stable.
* **AP, not accuracy, as the headline.** Positives are 4.55% of pairs; the
  decision threshold is not meaningful until it is reported. Every arm
  therefore also carries a threshold table and a reliability check, so "p ≥
  0.90" is evaluated rather than assumed.
* **Transfer is zero-shot by construction.** `fit_transfer` estimates every
  fitted quantity on K562 and only transforms the target rows; the target
  labels are read after prediction and never enter a feature. The target cell
  line still supplies its measured marks and accessibility — the deployable
  question is "same function, new marks", not "retrain the state arm".
* **Arms are shipped, not implied.** The nested P0 / P1 / D0 arms are the only
  comparisons the repository makes: what the marks buy, what distance alone
  buys, and what the function does in five new cell lines. Ablations that need
  re-derived raw data are left to a paper, where they can carry their own
  provenance.

## Status

- [x] real-data full-bio arm (LOCO, gene-cluster CI, marks ablation)
- [x] operating points (threshold table: calls/precision/recall/specificity/F1/MCC)
- [x] calibration diagnostics (Brier, ECE, reliability bins) — no re-fitting yet
- [x] zero-shot transfer to five held-out cell types (pooled AP 0.6661)
- [x] packaged design matrices + provenance doc
- [x] unit tests + GitHub Actions CI (both arms run end-to-end)
- [ ] isotonic/Platt recalibration and a recalibrated call set
- [ ] head-to-head table against a CNN / distilled student on the same splits
- [ ] state-specific arm retrained per cell type (needs matched labels)

## Citation & license

See `CITATION.cff`. Upstream data: Gasperini et al., *Cell* 2019
(doi:10.1016/j.cell.2019.11.001) and Fulco et al., *Nature Genetics* 2019
(doi:10.1038/s41588-019-0537-1). Code and packaged matrix: MIT (`LICENSE`).
