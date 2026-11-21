# Results

## Synthetic smoke world (current)

Command:

```bash
linearbaseline synthetic --out data/synthetic.npz
linearbaseline train      --config configs/synthetic.json --out results
linearbaseline ablations  --config configs/synthetic.json --out results
```

World: 4000 loci × 2 rows, 200 bp sequences, motif dosage 1–9 per locus,
prevalence 0.380. Split by locus 3200/800. `C` chosen by 5 grouped folds over
`{0.01, 0.1, 1, 10}`.

| arm | features | C | AP | AUC | log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0 | – | 0.3663 | – | 0.6575 |
| distance-only | 4 | 0.01 | 0.3845 | 0.5239 | 0.6568 |
| kmer-only | 64 | 0.01 | 0.7599 | 0.8533 | 0.4555 |
| kmer+distance | 68 | 0.01 | 0.7626 | 0.8548 | 0.4537 |
| full | 80 | 0.1 | 0.7596 | 0.8543 | 0.4550 |

Main arm: AP 0.7596, locus-bootstrap 95% CI [0.7125, 0.8000], 1000 resamples.

### How to read these numbers

* They are **synthetic**. The generator is part of this repo; the labels are
  drawn from a known latent variable. Passing them proves the plumbing, not the
  biology.
* `distance-only` at chance is expected and useful: it shows that a feature
  which is highly predictive in real CRE–gene data is *not* predictive here,
  so any claim of signal has to come from sequence.
* `full` ≈ `kmer-only` because the toy world's accessibility and context terms
  are weak relative to motif dosage. On real data we expect ATAC/context to
  matter more; the ladder will then separate.

## Real data

Not yet run. When a bundle exists, record:

| dataset | loci | positives | AP | CI | notes |
| --- | ---: | ---: | ---: | --- | --- |
| – | – | – | – | – | – |

Rules for filling this table:

1. one row per dataset version (hash the `.npz` and record it in `run.json`);
2. the CI is locus-level, 1000 resamples minimum;
3. report the split seed and the number of held-out loci;
4. never compare numbers across different negative-sampling schemes.
