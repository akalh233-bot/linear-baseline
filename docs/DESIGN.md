# Design notes

Why the baseline is built the way it is, and what it deliberately refuses to do.

## 1. The unit of generalisation is the locus

Regulatory datasets leak in three usual ways:

1. random rows — two windows from the same enhancer land in train and test;
2. random loci but shared neighbourhoods — a held-out locus is 200 bp from a
   training one and carries the same chromatin state;
3. augmentation leakage — a mutated or reverse-complemented copy of a test
   sequence is used in training.

This repository fixes (1) and (3) by construction: splits are by `locus_id`,
and the baseline does not augment. (2) is a dataset-level responsibility: the
locus ids you supply must already merge overlapping windows.

## 2. Folds: round-robin after sorting by positive rate

`label_balanced_folds` sorts loci by their positive rate and assigns
`fold = rank % n_splits`. Two properties matter:

* **no locus straddles folds** — every row of a locus has the same fold;
* **balance** — after sorting, consecutive ranks have similar rates, and
  round-robin spreads them across folds.

Cost is O(n log n) and measured at ~8 ms for 50k loci, against ~11 s for
`sklearn.model_selection.StratifiedGroupKFold` on the same input. The saved time
is real: in the Step-0 diagnostic it was the single largest cost item, larger
than all model fits combined.

## 3. Metric: average precision, with a locus-level interval

AP answers "if I rank all candidates, how much precision do I buy in the top
of the list". It is prevalence-sensitive, which is why the majority baseline is
reported next to it.

The interval resamples **loci**, not rows. With 2 rows per locus, a row-level
bootstrap produces intervals roughly 1/√2 too narrow and, in the nested design,
coverage drops from 0.95 to ~0.72. Locus-level resampling restores nominal
coverage (measured 0.917 at 60 loci × 400 resamples).

`bootstrap_ap_ci` refuses to return an interval if fewer than half of the
resamples contain both classes; sparse positives fail loudly instead of
silently returning a degenerate `[x, x]`.

## 4. Features: the weakest thing that still uses sequence

| group | dims | what it captures | what it cannot |
| --- | ---: | --- | --- |
| 3-mer frequency | 64 | motif composition, GC | spacing, orientation, order |
| distance | 4 | TSS proximity, smooth + binned | 3D contact, insulation |
| ATAC | 1 | accessibility | cell-state-specific footprint |
| context | 11 | TF identity, cell state | TF expression level, cooperativity |

Deliberately excluded: PWM scanning, k-mer *positions*, learned embeddings.
Every exclusion makes the baseline weaker; that is the point. If a CNN cannot
beat 3-mer bags by a margin larger than the seed noise floor, the CNN is not
using sequence structure.

## 5. Model selection

`C` is chosen by mean grouped-CV AP over a small grid, then the model is refit
on all training rows. No early stopping, no calibration — probabilities are
raw logistic outputs, and `log_loss`/`brier` are reported so miscalibration is
visible rather than hidden.

## 6. What would make this baseline wrong

* **Distance-dominant labels.** If the label is mostly "is this within 20 kb of
  a TSS", the linear model with distance features will hit AP 0.8+ without
  looking at sequence at all. In that case report the distance-only arm first;
  it is the real baseline.
* **Batch effects across loci.** If loci come from a few studies with different
  peak-calling thresholds, a model can learn study identity. Include the study
  as a context feature only as a diagnostic, never as a predictor.
* **Negative sampling.** AU/AP depend on the negative distribution. Matched
  negatives (GC- and distance-matched) are mandatory for a fair comparison
  against published numbers.

## 7. The real-data arm (full bio)

The synthetic ladder above is a plumbing test. The arm that produces the
headline number is `linearbaseline fullbio`, and it differs in four ways.

**Split: leave-one-chromosome-out (23 folds), not a random locus split.** A
random locus split leaves the held-out CRE inside the same chromatin
neighbourhood as training CREs, and a linear model will happily learn "this
region looks active". Holding out a whole chromosome is the cheapest test of
cross-region generalisation we can run without a second cell type.

**Feature fitting is strictly in-fold.** PCA-4 of the 128 JASPAR scores, the
gene-context medians, the standardisation statistics and the logistic weights
are estimated on the 22 training chromosomes and applied to the held-out one.
`loco_oof` never touches the test chromosome, not even for a column median; the
only global statistic is the expression prior's z-score, and that is a property
of the gene annotation, not of the labels.

**The 26 columns are fixed, not searched.** Distance (2), motif PCA (4),
accessibility (1), neighbourhood (4), gene side (5), expression prior (1),
measured marks (9). The count is asserted in a test, so a silent feature
addition shows up as a failing 26.

**Arms are nested by data availability, not by model capacity.**

| arm | what it costs to deploy |
| --- | --- |
| D0 distance only | nothing beyond a TSS annotation |
| P1 no measured marks | ATAC + sequence + gene annotation |
| P0 full bio | the above plus four ChIP marks in the target cell type |

The interesting number is not P0; it is P1, because that is what a new cell type
actually gives you. P0 exists to price the marks: +0.0208 AP, and earlier
proxy experiments recovered only 6% of that from ATAC + sequence, so the marks
are not replaceable by a bigger model here.

**Interval: gene-cluster bootstrap.** The 10,356 pairs come from 2,113 genes
(4.9 pairs per gene), and pairs of one gene share the gene-side, expression and
neighbourhood columns. Resampling rows would treat them as independent; the
committed CI resamples genes.
