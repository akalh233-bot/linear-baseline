# linear-baseline

**Honest linear baselines for cis-regulatory sequence-to-function prediction.**
A TF–CRE / CRE–gene classifier that is deliberately boring: 3-mer frequencies,
distance, accessibility and context, fed to L2 logistic regression — with
locus-aware splits and locus-level bootstrap confidence intervals.

> 中文摘要：这是我为 Tri-Omics Transformer / KD-GRN 研究线准备的**线性基线仓库**。
> 它只做一件事——给出"任何序列模型必须先赢过的地板"，并且把这个地板做得可复现、
> 可审计：按 locus 分组划分（不泄漏）、AP 为主指标、locus 级 bootstrap 置信区间、
> 特征组消融阶梯。仓库目前是**占位状态**：合成数据跑通、CI 通过，真实数据结果待填。

---

## Why this exists

Every sequence model in this project (CNN, dilated CNN, Transformer, distilled
student) claims a gain over *something*. That something has to be defined
precisely, otherwise a +0.01 AP difference is indistinguishable from seed noise.

This repository defines the floor:

1. **Same split as the fancy models** — grouped by genomic locus, never random
   rows. A model that wins only under random splitting has learned locus
   identity, not regulatory grammar.
2. **Same metric** — average precision (AP), with a locus-level bootstrap CI.
   Rows inside one locus are correlated; resampling rows understates the error
   by a factor of ~2 in our experience.
3. **Same ablations** — majority, distance-only, k-mer-only, k-mer+distance,
   full. If "full" does not beat "k-mer-only", the extra covariates are noise.

The baseline also doubles as the Gate-0B reference in the KD-GRN protocol:
a teacher model that cannot beat this student by a pre-registered margin has
nothing to distil.

## Results (synthetic world, smoke scale)

Generated with `linearbaseline synthetic` (4000 loci × 2 rows, 200 bp,
seed 0), split by locus 80/20, five grouped folds for `C` selection.
Held-out 800 loci, 1000 locus-bootstrap resamples.

| arm | features | C | AP | AUC | log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0 | – | 0.3663 | – | 0.6575 |
| distance-only | 4 | 0.01 | 0.3845 | 0.5239 | 0.6568 |
| kmer-only | 64 | 0.01 | 0.7599 | 0.8533 | 0.4555 |
| kmer+distance | 68 | 0.01 | 0.7626 | 0.8548 | 0.4537 |
| **full** | 80 | 0.1 | **0.7596** | 0.8543 | 0.4550 |

Headline: **held-out AP 0.7596** (95% CI [0.7125, 0.8000]) against a 0.3663
majority baseline. Distance alone is at chance (0.3845), which is expected:
in this toy world the label is driven by motif dosage, not by TSS proximity.
That contrast is the point — the ablation ladder makes the claim falsifiable.

These are **synthetic** numbers, produced by the repository's own generator.
They demonstrate that the pipeline is wired correctly and that the metrics are
sane. They are **not** evidence about any real regulatory dataset. See
`docs/RESULTS.md` for the reproduction commands and the honesty caveats.

## Install

```bash
python -m pip install -e ".[dev]"
```

or, without installing, prefix commands with `PYTHONPATH=src`
(PowerShell: `$env:PYTHONPATH="src"`).

## Run

```bash
# 1. synthetic bundle (regenerated deterministically from its seed)
linearbaseline synthetic --out data/synthetic.npz

# 2. fit + held-out evaluation + bootstrap CI
linearbaseline train --config configs/synthetic.json --out results

# 3. feature-group ablation ladder
linearbaseline ablations --config configs/synthetic.json --out results

# 4. tests
pytest -q
```

Outputs land in `results/`: `metrics.json`, `run.json` (full config + versions),
`predictions.npz`, `coefficients.csv`, `ablations.json`.

## Data contract

`train` accepts any `.npz` with these arrays (see `docs/DESIGN.md` for details):

| key | dtype | meaning |
| --- | --- | --- |
| `seq` | uint8 `(N, L)` | A=0, C=1, G=2, T=3 |
| `y` | int8 `(N,)` | binary label |
| `locus_id` | `<U*` or int `(N,)` | genomic locus; **the unit of splitting** |
| `tf_id` | `<U*` `(N,)` | optional context feature |
| `cell_state` | `<U*` `(N,)` | optional context feature |
| `distance_bp` | float `(N,)` | optional CRE→TSS distance |
| `atac` | float `(N,)` | optional accessibility summary |

Point `configs/real_example.json` at your bundle and run the same two commands.
Rows that share a `locus_id` are never split across train and test.

## Design decisions

* **Grouped, label-balanced folds.** Folds are assigned round-robin after
  sorting loci by positive rate: O(n log n), ~8 ms at 50k loci, no locus
  straddles a fold. This replaces `StratifiedGroupKFold`, which takes ~11 s at
  the same scale for the same guarantee.
* **AP, not accuracy.** Positives are rare and the decision threshold is not
  meaningful yet.
* **Locus-level bootstrap.** Resampling loci (not rows) is the only interval
  that reflects cross-locus generalisation. Degenerate resamples are counted,
  and the run fails loudly if fewer than half are usable.
* **No sequence order.** 3-mer bags cannot see motif spacing or orientation.
  That is intentional: it is the weakest model that still uses sequence, and it
  makes the "we need convolutions" argument testable.

## Status

- [x] feature builders, logistic regression, grouped CV
- [x] locus-level bootstrap CI, ablation ladder
- [x] unit + end-to-end tests, GitHub Actions CI
- [ ] real-data run (ChIP/MPRA-derived bundles)
- [ ] calibrated probabilities (isotonic / Platt) and ECE reporting
- [ ] comparison table against the CNN / distilled student

## Citation

If you use this code, cite the repository (see `CITATION.cff`). The scientific
context is the perturbation-aware distillation protocol described in the
project's internal reports; this repo only provides the baseline.

## License

MIT — see `LICENSE`.
