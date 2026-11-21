# Results

One arm is reported here: the **real-data** full-bio CRE→gene baseline. It is
the reason the repository exists, and every number below is the committed
output of

```bash
linearbaseline fullbio --config configs/fullbio.json --out results
```

→ `results/fullbio_loco.json`. Runtime ~6 s on CPU, 1,000 gene-cluster
resamples included.

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

### Reading the table honestly

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
5. **This is one cell line and one label system.** No cross-cell-state claim is
   made. The marks in a different cell type would have to be re-measured.

### Why LOCO and not a random locus split

A random locus split lets the held-out CRE sit in the same chromatin
neighbourhood as training CREs; the model can win by memorising the region.
Holding out an entire chromosome removes that shortcut. The price is a noisier
estimate (23 folds, some with very few positives), which is why the headline
also carries a bootstrap interval.

## 2. Rules for adding a row

1. One row per dataset version; record the `.npz` SHA-256 in `run.json`.
2. The CI is cluster-level (gene or chromosome), 1,000 resamples minimum.
3. Report the split seed and the number of held-out clusters.
4. Never compare numbers across different negative-sampling schemes.
5. If the label is distance-dominated, report the distance-only arm first.
6. A change in the feature set must change the column count (asserted by a
   test) or the `docs/DATA.md` SHA-256 — never silently.
