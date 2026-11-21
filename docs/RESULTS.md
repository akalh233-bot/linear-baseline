# Results

Two worlds are reported here, in this order: the **real-data** arm (the reason
the repository exists) and the **synthetic** smoke world (a plumbing test).

Every real-data number below is the committed output of

```bash
linearbaseline fullbio --data data/fullbio/features.npz --out results
```

→ `results/fullbio_loco.json`. Runtime ~5.5 s on CPU.

## 1. Real data: EPCrisprBenchmark K562, CRE→gene

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
4. **CIs are gene-level.** With 2.2 pairs per gene on average the interval is
   only modestly wider than a row bootstrap, but for genes with many candidate
   CREs the difference is large; the CI above is the conservative one.
5. **This is one cell line and one label system.** No cross-cell-state claim is
   made. The marks in a different cell type would have to be re-measured.

### Why LOCO and not a random locus split

A random locus split lets the held-out CRE sit in the same chromatin
neighbourhood as training CREs; the model can win by memorising the region.
Holding out an entire chromosome removes that shortcut. The price is a noisier
estimate (23 folds, some with very few positives), which is why the headline
also carries a bootstrap interval.

## 2. Synthetic smoke world (pipeline check only)

```bash
linearbaseline synthetic --out data/synthetic.npz
linearbaseline train     --config configs/synthetic.json --out results
linearbaseline ablations --config configs/synthetic.json --out results
```

World: 4,000 loci × 2 rows, 200 bp, motif dosage 1–9 per locus, prevalence
0.380. Split by locus 3,200/800; `C` by 5 grouped folds over {0.01, 0.1, 1, 10}.

| arm | features | C | AP | AUC | log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0 | – | 0.3663 | – | 0.6575 |
| distance-only | 4 | 0.01 | 0.3845 | 0.5239 | 0.6568 |
| k-mer only | 64 | 0.01 | 0.7599 | 0.8533 | 0.4555 |
| k-mer + distance | 68 | 0.01 | 0.7626 | 0.8548 | 0.4537 |
| full | 80 | 0.1 | 0.7596 | 0.8543 | 0.4550 |

Main arm: AP 0.7596, locus-bootstrap 95% CI [0.7125, 0.8000], 1,000 resamples.

These numbers prove the plumbing (splits, metrics, CI, CLI). They are **not**
evidence about any real dataset, and they must never be quoted as such.

## 3. Rules for adding a row

1. One row per dataset version; record the `.npz` SHA-256 in `run.json`.
2. The CI is cluster-level (locus or gene), 1,000 resamples minimum.
3. Report the split seed and the number of held-out clusters.
4. Never compare numbers across different negative-sampling schemes.
5. If the label is distance-dominated, report the distance-only arm first.
