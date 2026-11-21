# Design notes

Why the baseline is built the way it is, and what it deliberately refuses to do.

The repository ships exactly one arm: a standardised logistic regression over 26
columns, evaluated leave-one-chromosome-out on the EPCrisprBenchmark K562
CRE→gene set. Its purpose is to be the floor that any sequence model has to
beat, and to be auditable column by column.

## 1. The unit of generalisation is the chromosome

Regulatory datasets leak in three usual ways:

1. random rows — two windows from the same enhancer land in train and test;
2. random loci but shared neighbourhoods — a held-out locus sits inside the
   same chromatin domain as a training one and carries the same state;
3. augmentation leakage — a mutated or reverse-complemented copy of a test
   sequence is used in training.

This baseline fixes (1) and (3) by construction: splits are 23 chromosome
holdouts, and the model does no augmentation. (2) is why the holdout is a whole
chromosome rather than a locus: a linear model with neighbourhood features will
happily memorise "this region looks active" if any part of the region is in
training.

The price is a noisy estimate — 23 folds, some with a handful of positives —
which is why every headline number carries a gene-cluster bootstrap interval.

## 2. Feature fitting is strictly in-fold

`loco_oof` fits, on the 22 training chromosomes only: the PCA-4 basis of the
128 JASPAR motif scores, the gene-context medians used for NaN imputation, the
standardisation means/variances, and the logistic weights. Nothing about the
test chromosome — not even a column median — enters the fit.

The single exception is the expression prior: a global median imputation plus a
z-score of `gene_expr_sym`. It is a property of the gene annotation rather than
of the labels, and it is applied identically to every fold.

`C` is fixed at 1.0 rather than tuned per fold. Tuning it on the held-out
chromosome would be selection on the test set; tuning it on the training
chromosomes would need an inner split, and at 471 positives the inner AP is too
noisy to choose from. Fixing it at the value used by the upstream
`phase3/realdata` pipeline keeps this repository a faithful packaging of that
result.

## 3. The 26 columns are fixed, not searched

| block | dims | what it captures | fitted where |
| --- | ---: | --- | --- |
| distance | 2 | `log1p` distance, its square | nowhere (raw), scaled in-fold |
| motif | 4 | PCA-4 of 128 JASPAR scores | in-fold |
| accessibility | 1 | `log1p` DHS RPM at the CRE | nowhere (raw) |
| neighbourhood | 4 | leave-one-out summary of the gene's other candidate CREs | shipped |
| gene side | 5 | promoter/gene context from the rE2G bed | shipped (NaN → in-fold median) |
| expression prior | 1 | z-scored log expression of the target gene | global median + z |
| measured marks | 9 | H3K27ac, its percentile, CTCF, H3K4me1, H3K27me3, four peak flags | shipped |

The count is asserted by a test, so a silent feature addition shows up as a
failing 26. Why PCA-4 rather than 128 raw scores: the JASPAR columns are
strongly correlated and only 471 positives exist, so raw scores let the model
fit fold-specific motif noise; four components fitted in-fold keep the design
matrix stable without looking at the test chromosome.

## 4. Arms are nested by data availability, not by model capacity

| arm | what it costs to deploy |
| --- | --- |
| D0 distance only | nothing beyond a TSS annotation |
| P1 no measured marks | ATAC + sequence + gene annotation |
| P0 full bio | the above plus four ChIP marks in the target cell type |

The interesting number is not P0; it is P1, because that is what a new cell type
actually gives you. P0 exists to price the marks: +0.0208 AP, and earlier proxy
experiments recovered only 6% of that from ATAC + sequence, so the marks are not
replaceable by a bigger model here.

D0 prices distance. It reaches 0.42 AP with high AUC and near-chance AP in the
distal bin (0.0186), which is the honest statement of how much of this task is
"is the CRE near the TSS".

## 5. Metric: average precision, with a gene-cluster interval

AP answers "if I rank all candidates, how much precision do I buy at the top of
the list". It is prevalence-sensitive (4.55% positives here), which is why the
prevalence and the positive count are stated next to every AP.

The interval resamples **genes**, not rows. With 4.9 pairs per gene, and with
each pair sharing the gene-side, expression and neighbourhood columns, a
row-level bootstrap treats correlated rows as independent evidence and
understates the interval. `bootstrap_ap_ci` also refuses to return an interval
if fewer than half of the resamples contain both classes: sparse positives fail
loudly instead of silently returning a degenerate `[x, x]`.

## 6. What would make this baseline wrong

* **Proposing its own candidates.** The negatives come from the benchmark's
  tested pairs. A model that generates candidates will face a different,
  harder negative distribution; the 0.6715 does not transfer to that setting.
* **Gene-side features from a candidate universe.** `prom_dnase`,
  `n_cand_elem`, `n_tss`, `sum_nearby_enh`, `ubiq_frac` come from the rE2G bed.
  They encode the candidate universe itself, and external validation in this
  project showed that class of feature transfers poorly. That is why P1 (0.6507)
  is described as an optimistic bound, not a deployable number.
* **A different cell state.** The nine mark columns are K562 ChIP data. In
  another cell type they must be re-measured; without them the model is P1, and
  nothing here tests cross-state transfer.
* **A different negative-sampling scheme.** AU/AP depend on the negative
  distribution. Numbers from GC-matched or distance-matched sampling are not
  comparable to these and must be reported separately.

## 7. Non-goals

No probability calibration (raw logistic outputs are reported, so miscalibration
is visible). No sequence model, no augmentation, no multi-task head. Those are
the things this baseline exists to be compared against, not to contain.
