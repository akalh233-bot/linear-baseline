# linear-baseline

**A 26-column logistic regression that is the honest floor for CRE→gene
prediction — and the number any sequence model has to beat.**

On the EPCrisprBenchmark K562 CRE–gene task, under **leave-one-chromosome-out**
cross-validation, this baseline reaches **AP 0.6715** (95% CI
[0.6151, 0.7226], 10,356 pairs / 471 positives / 23 chromosomes). With the nine
*measured* chromatin marks removed — the deployable setting — it reaches
**0.6507** (CI [0.5937, 0.7020]). The gap, **+0.0208 AP**, is the entire value
of H3K27ac/CTCF/H3K4me1/H3K27me3 ChIP data for this task.

> 中文摘要：这是把 `phase3/realdata` 里那条 full-bio 逻辑回归正式打包成仓库。
> 主结果：LOCO AP **0.6715**，去掉 9 个实测标记后 **0.6507**；距离 <20 kb 的
> 子集 0.8327，而 100 kb–1 Mb 只有 0.2763；只给距离时远端只有 0.0186。

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

### What the numbers actually say

1. **Distance alone gets 0.42 AP** — and nearly all of it in the near field
   (<20 kb: 0.63; 100 kb–1 Mb: 0.019). The <20 kb bin is at AP 0.83 while
   100 kb–1 Mb collapses to 0.28. A model that only learns "close to TSS"
   already gets most of the easy cases; the hard, interesting regime is distal,
   and there the full model is still only at 0.28 with 76 positives.
2. **Measured marks are worth 0.02 AP**, and essentially all of it is in the
   <20 kb and >100 kb bins, not in the 20–100 kb middle.
3. **This is a K562-only, single-label-system floor.** It says nothing about
   cross-cell-state transfer, and its gene-side features come from a candidate
   universe that is known not to transfer perfectly. Treat 0.6507 as an
   optimistic deployable bound.

## Reproduce

```bash
python -m pip install -e ".[dev]"

# the arm: LOCO CV + gene-cluster bootstrap  (~6 s)
linearbaseline fullbio --config configs/fullbio.json --out results

pytest -q
```

`results/fullbio_loco.json` in this repository is the committed output of that
command; it contains every number above plus the per-bin sample sizes, and
`tests/test_fullbio.py::test_readme_matches_committed_results` fails if the
README drifts from it.

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
  decision threshold is not meaningful until calibration is done.
* **Arms are shipped, not implied.** The nested P0 / P1 / D0 arms are the only
  comparisons the repository makes: what the marks buy, and what distance alone
  buys. Ablations that need re-derived raw data are left to a paper, where they
  can carry their own provenance.

## Status

- [x] real-data full-bio arm (LOCO, gene-cluster CI, marks ablation)
- [x] packaged 3.4 MB design matrix + provenance doc
- [x] unit tests + GitHub Actions CI (the fullbio arm runs end-to-end)
- [ ] probability calibration (isotonic / Platt) and ECE reporting
- [ ] head-to-head table against a CNN / distilled student on the same splits
- [ ] cross-cell-state arm (needs matched labels; not yet available)

## Citation & license

See `CITATION.cff`. Upstream data: Gasperini et al., *Cell* 2019
(doi:10.1016/j.cell.2019.11.001) and Fulco et al., *Nature Genetics* 2019
(doi:10.1038/s41588-019-0537-1). Code and packaged matrix: MIT (`LICENSE`).
