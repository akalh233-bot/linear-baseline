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
