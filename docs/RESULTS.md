# Results

Two tasks are reported here: the **real-data** full-bio CRE→gene baseline on
K562, and its **zero-shot transfer** to five held-out cell types. Every number
below is the committed output of

```bash
linearbaseline fullbio  --config configs/fullbio.json  --out results
linearbaseline transfer --config configs/transfer.json --out results
```

→ `results/fullbio_loco.json` and `results/heldout_transfer.json`. Runtime ~6 s
and ~2 s on CPU, 1,000 gene-cluster resamples included in both.

## 1. EPCrisprBenchmark K562, CRE→gene

* 10,356 candidate (CRE, gene) pairs, **471 positives** (4.55%).
* 23 chromosomes held out one at a time (**LOCO**); 2,113 genes.
* 26 columns; PCA-4 / medians / standardisation / weights all fitted on the 22
  training chromosomes.
* Negative sampling: as shipped by the benchmark (all tested pairs, not a
  GC-matched subsample) — this matters when comparing to published numbers.

| arm | features | AP | 95% CI (gene bootstrap) | AUC |
| --- | ---: | ---: | --- | ---: |
| P0 full bio | 26 | **0.6715** | [0.6151, 0.7226] | 0.9459 |
| P1 no measured marks | 17 | 0.6507 | [0.5937, 0.7020] | 0.9440 |
| D0 distance only | 2 | 0.4210 | [0.3666, 0.4803] | 0.8931 |

By CRE→TSS distance:

| bin | pairs | positives | P0 AP | P1 AP | D0 AP | Δ marks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <20 kb | 421 | 196 | 0.8327 | 0.8076 | 0.6335 | +0.0251 |
| 20–100 kb | 1,399 | 193 | 0.6538 | 0.6533 | 0.2095 | +0.0005 |
| 100 kb–1 Mb | 7,938 | 76 | 0.2763 | 0.2563 | 0.0186 | +0.0200 |

### 1.1 Operating points (p ≥ threshold)

Every row is a complete call set: a pair is called positive iff its probability
is at least the threshold. `calls = TP + FP`; precision is what share of the
shortlist is real, recall is what share of the 471 positives is recovered.

| p ≥ | calls | TP | FP | precision | recall | specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 299 | 232 | 67 | 0.7759 | 0.4926 | 0.9932 | 0.6026 |
| 0.60 | 252 | 207 | 45 | 0.8214 | 0.4395 | 0.9954 | 0.5726 |
| 0.70 | 194 | 162 | 32 | 0.8351 | 0.3439 | 0.9968 | 0.4872 |
| 0.75 | 155 | 133 | 22 | 0.8581 | 0.2824 | 0.9978 | 0.4249 |
| 0.80 | 127 | 111 | 16 | 0.8740 | 0.2357 | 0.9984 | 0.3712 |
| 0.85 | 103 | 92 | 11 | 0.8932 | 0.1953 | 0.9989 | 0.3206 |
| 0.90 | 72 | 64 | 8 | 0.8889 | 0.1359 | 0.9992 | 0.2357 |
| 0.95 | 35 | 33 | 2 | 0.9429 | 0.0701 | 0.9998 | 0.1304 |
| 0.99 | 2 | 2 | 0 | 1.0000 | 0.0042 | 1.0000 | 0.0085 |

The no-marks P1 arm, at 0.90: 61 calls (55 TP / 6 FP), precision 0.9016, recall
0.1168; at 0.95: 32 calls, precision 0.9688. In other words the nine measured
marks improve ranking (+0.0208 AP) but not the precision of the 0.90 shortlist —
P1 is if anything slightly more precise there because it is less confident
overall. The threshold table is in `results/fullbio_loco.json` under
`arms.*.all.thresholds`, per distance bin as well: near field (<20 kb) at 0.90
gives 65 calls at precision 0.8923 and recall 0.2959, i.e. the same 0.90 rule
recovers three times more positives close to the TSS.

### 1.2 Calibration

| arm | Brier | ECE (10 bins) | top decile mean predicted | top decile observed |
| --- | ---: | ---: | ---: | ---: |
| P0 full bio | 0.0232 | 0.0032 | 0.9485 | 0.8889 |
| P1 no marks | 0.0242 | 0.0043 | 0.9481 | 0.9016 |

The pooled ECE is small mostly because 9,443 of 10,356 pairs sit below 0.1 and
the model is right that they are negative; the decision-relevant statement is in
the top decile: predicted 0.9485, observed 0.8889. So p ≥ 0.90 is a ~89% precision
statement in-domain, not a 95% one, and it is over-confident at the very top by
about six points. `calibration` (Brier, ECE, all ten bins) is stored in
`arms.*.all.calibration`. No recalibration (isotonic/Platt) is applied; the raw
logistic outputs are reported.

### 1.3 Reading the table honestly

1. **0.67 is a floor for a 26-column model, not a ceiling.** It uses a
   candidate universe supplied by the benchmark; a model that proposes its own
   candidates will look worse on the same negatives. Distance alone is 0.42 —
   high AUC (0.89) but poor AP, because it cannot rank *within* the distal bin.
2. **The distal bin is the real problem.** 76 positives at 100 kb–1 Mb, AP
   0.28 for the full model and 0.019 for distance alone. Any claim of "distal
   enhancer prediction" has to be judged here, not on the pooled 0.67.
3. **The marks are worth 0.02, and only 6% of that is reconstructible from
   ATAC + sequence** (measured earlier in the project with a Ridge proxy:
   +0.0013 of +0.0208). Dropping the ChIP data and accepting 0.6507 is the
   defensible deployable choice.
4. **CIs are gene-level.** 4.9 pairs per gene on average, and pairs of one gene
   share the gene-side, expression and neighbourhood columns, so a row-level
   bootstrap would understate the interval. The CI above resamples genes.
5. **This is one cell line and one label system for the headline number.** The
   zero-shot transfer below is the only cross-cell-state evidence here; it gives
   the target cell line its measured marks, so it does not cover a cell type
   without ChIP/ATAC data.

### Why LOCO and not a random locus split

A random locus split lets the held-out CRE sit in the same chromatin
neighbourhood as training CREs; the model can win by memorising the region.
Holding out an entire chromosome removes that shortcut. The price is a noisier
estimate (23 folds, some with very few positives), which is why the headline
also carries a bootstrap interval.

## 2. Zero-shot transfer to five held-out cell types

`linearbaseline transfer` fits the 26-column model on the whole K562 table
(PCA, gene-context medians, expression centring, standardisation, weights) and
applies it unchanged to the held-out five-cell-type benchmark: 4,378 pairs, 190
positives, zero overlap with the training pairs. The target cell line supplies
its own measured marks, accessibility and neighbourhood columns (all
label-free); the five gene-side columns and the expression prior are K562
proxies. Results: `results/heldout_transfer.json`.

| cell type | n | positives | P0 AP | P0 CI | P1 AP | calls @ p ≥ 0.90 | precision @ 0.90 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **pooled** | 4,378 | 190 | **0.6661** | [0.5878, 0.7392] | 0.6602 | 19 | 0.9474 |
| K562 | 1,918 | 118 | 0.7194 | [0.6233, 0.8013] | 0.7219 | 14 | 1.0000 |
| WTC11 | 1,921 | 15 | 0.5026 | – | 0.5633 | 0 | – |
| HCT116 | 396 | 34 | 0.6090 | [0.3841, 0.8035] | 0.5814 | 3 | 0.6667 |
| GM12878 | 68 | 16 | 0.8853 | – | 0.8728 | 2 | 1.0000 |
| Jurkat | 75 | 7 | 0.6187 | – | 0.8441 | 0 | – |

Intervals are gene-cluster bootstraps, reported only for cell types with at
least 20 positives; below that the interval degenerates and a point AP is the
honest output. What the table says:

1. **The function transfers at the pooled level.** 0.6661 vs 0.6715 in-domain,
   with an interval that overlaps the in-domain one. On 190 positives that is a
   real, if modest, result: the K562-fitted linear function is not a K562
   artefact.
2. **Marks stop mattering.** The no-marks P1 arm transfers at 0.6602, so the
   +0.0208 in-domain mark premium becomes +0.0059 out of domain. The measured
   marks are largely a K562 ranking refinement.
3. **High-confidence calls are rarer and uneven.** Pooled p ≥ 0.90 gives 19
   calls (18 true, 0.947). K562 and GM12878 carry 16 of them; WTC11 (15
   positives) and Jurkat (7) produce none at all. A threshold calibrated
   in-domain does not keep its call rate in a sparse target cell type.
4. **Per-cell AP order is not stable.** Jurkat is 0.6187 (P0) but 0.8441 (P1)
   on 7 positives, the opposite direction from the pooled mark premium; the
   cell-type ranking should be read as a hypothesis, not a fixed ordering.
5. **This is transfer with measured target marks, not a state-specific arm.**
   Retraining per cell type — the thing a "cross-cell-state" claim would need —
   is not done here.

## 3. Rules for adding a row

1. One row per dataset version; record the `.npz` SHA-256 in `run.json`.
2. The CI is cluster-level (gene or chromosome), 1,000 resamples minimum.
3. Report the split seed and the number of held-out clusters.
4. Never compare numbers across different negative-sampling schemes.
5. If the label is distance-dominated, report the distance-only arm first.
6. A change in the feature set must change the column count (asserted by a
   test) or the `docs/DATA.md` SHA-256 — never silently.
