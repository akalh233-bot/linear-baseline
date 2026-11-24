# linear-baseline

**Auditable logistic-regression floors for CRE→gene prediction — from zero-ChIP deployable (AP 0.6772) to full-bio (AP 0.7001) and +Micro-C 3D contact (AP 0.7042).**

On the EPCrisprBenchmark K562 CRE–gene task under **23-fold leave-one-chromosome-out (LOCO)** cross-validation (10,356 pairs / 471 positives / 2,113 genes):

* **`P2_full_bio_3d` (42 cols, +K562 Micro-C 3D)**: **AP 0.7042** (95% CI [0.6547, 0.7519]); `p ≥ 0.90` gives 88 calls at precision 0.9659 and recall 0.1805.
* **`P0_full_bio` (38 cols, portable, no 3D)**: **AP 0.7001** (95% CI [0.6495, 0.7472]); `P0_nobio` (34 cols, zero screen-derived `bio`) reaches **0.7007**.
* **`P1_no_marks` (25 cols, strictly zero-ChIP, enforced by unit test)**: **AP 0.6772** (95% CI [0.6240, 0.7276]), with `100 kb–1 Mb` distal AP **0.3196**; removing screen-derived `bio` (`P1_no_marks_nobio`, 23 cols) keeps **0.6745** (distal **0.3311**). Adding 1 kb CNN-predicted surrogate marks (`P1_no_marks+1kbCNN`, still zero measured ChIP) reaches **0.6815** LOCO (matching 196 kb Enformer **0.6832**) and **0.6083** on 4 non-K562 transfer.
* **Zero-shot transfer to held-out cell types** (4,378 pairs / 190 positives, split explicitly into **4 non-K562 cell types** [2,460 pairs / 72 positives] and in-domain K562 [1,918 pairs / 118 positives]):
  * `P0_full_bio`: all-5 pooled **AP 0.7151** (CI [0.6358, 0.7865], `p ≥ 0.90` precision **0.9444**), **4 non-K562 pooled AP 0.6753** (CI [0.5316, 0.8073]).
  * `P1_no_marks` (strictly zero-ChIP): all-5 pooled **0.6572**, **4 non-K562 pooled 0.5945** — the true cross-cell ChIP mark premium is **+0.0579** pooled (CI [+0.0283, +0.0918]) and **+0.0808** on 4 non-K562 (CI [-0.0023, +0.1575]).
  * `P2_full_bio_3d` (K562-only Micro-C applied to other cell lines): all-5 **0.7037**, 4 non-K562 **0.6391** (-0.0362 vs `P0_full_bio`), showing that K562-specific 3D contact improves in-domain K562 (`>=1 Mb` jumps 0.3791 → 0.7440) but should not be transferred blindly to other cell types.

> 中文摘要（v0.4.0）：已落地 P0 正确性修复、P1 干净特征升级、P2 本地 K562 Micro-C 三维接触臂，以及发文级补充实验（9 档阈值精确率/召回率全表、嵌套 CV Platt/Isotonic 再校准、与 1 kb CNN / Borzoi / Enformer 同折正面对比、`direct_vs_indirect_negative` 加权评测与定量 `EffectSize` 相关性、看家基因缓冲 / 间隔 TSS 竞争 / `MYC`、`GATA1`、`HDAC6`、`HBE1`、`CCND1` 等文献位点机制解析）。

---

## Headline numbers (K562 23-fold LOCO)

Task: given a candidate CRE and target gene, predict whether CRISPRi perturbation changes gene expression (`Regulated` in EPCrisprBenchmark K562; Gasperini et al. 2019; Fulco et al. 2019). Fixed `C = 1.0` L2 logistic regression; all preprocessing fitted strictly on the 22 training chromosomes.

| arm | features | AP | 95% CI (gene boot) | AUC | <20 kb | 20–100 kb | 100 kb–1 Mb | >=1 Mb |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **P2_full_bio_3d** (+K562 Micro-C) | 42 | **0.7042** | [0.6547, 0.7519] | 0.9482 | 0.8476 | **0.7133** | **0.3212** | **0.7440** |
| **P0_full_bio** (portable) | 38 | **0.7001** | [0.6495, 0.7472] | 0.9444 | **0.8539** | 0.7032 | 0.3087 | 0.3791 |
| P0_nobio (no screen-derived `bio`) | 34 | 0.7007 | [0.6505, 0.7483] | 0.9444 | 0.8531 | 0.7149 | 0.3073 | 0.4373 |
| **P1_no_marks** (strictly zero-ChIP) | 25 | **0.6772** | [0.6240, 0.7276] | 0.9421 | 0.8284 | 0.6870 | 0.3196 | 0.5038 |
| P1_no_marks_nobio (zero-ChIP, no `bio`) | 23 | 0.6745 | [0.6203, 0.7245] | 0.9415 | 0.8256 | 0.6848 | **0.3311** | 0.5091 |
| D0_distance_only | 2 | 0.4210 | [0.3666, 0.4803] | 0.8931 | 0.6335 | 0.2095 | 0.0186 | 0.0073 |
| Δ marks (`P0 - P1`, paired CI) | +13 | **+0.0229** | [+0.0080, +0.0382] | +0.0023 | +0.0256 | +0.0161 | -0.0109 | -0.1248 |
| Δ 3D (`P2 - P0`, paired CI) | +4 | +0.0042 | [-0.0088, +0.0170] | +0.0038 | -0.0064 | +0.0101 | +0.0125 | +0.3650 |

Bin sizes: `<20 kb` 421 pairs (196 pos), `20–100 kb` 1,399 (193), `100 kb–1 Mb` 7,938 (76), `>=1 Mb` 598 (6). All CIs resample **genes** (2,113 clusters, 1,000 resamples).

### Operating thresholds & calibration (K562 LOCO)

#### `P0_full_bio` (38 cols, portable) — full 9-threshold call set

| p ≥ | calls | TP | FP | precision | recall | specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 323 | 252 | 71 | 0.7802 | 0.5350 | 0.9928 | 0.6348 |
| 0.60 | 265 | 221 | 44 | 0.8340 | 0.4692 | 0.9955 | 0.6005 |
| 0.70 | 213 | 182 | 31 | 0.8545 | 0.3864 | 0.9969 | 0.5322 |
| 0.75 | 189 | 163 | 26 | 0.8624 | 0.3461 | 0.9974 | 0.4939 |
| 0.80 | 151 | 139 | 12 | 0.9205 | 0.2951 | 0.9988 | 0.4469 |
| 0.85 | 108 | 103 | 5 | 0.9537 | 0.2187 | 0.9995 | 0.3558 |
| 0.90 | 78 | 76 | 2 | 0.9744 | 0.1614 | 0.9998 | 0.2769 |
| 0.95 | 35 | 34 | 1 | 0.9714 | 0.0722 | 0.9999 | 0.1344 |
| 0.99 | 7 | 7 | 0 | 1.0000 | 0.0149 | 1.0000 | 0.0293 |

Full 9-threshold tables for `P2_full_bio_3d`, `P1_no_marks`, and transfer strata (`all5` & `4-non-K562`) are in [docs/RESULTS.md](docs/RESULTS.md).

| arm | Brier | ECE | top decile pred / obs | calls @ p ≥ 0.90 | prec @ 0.90 | rec @ 0.90 | calls @ p ≥ 0.95 | prec @ 0.95 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| P2_full_bio_3d | 0.0221 | 0.0046 | 0.9495 / 0.9659 | 88 | 0.9659 | 0.1805 | 43 | 0.9767 |
| P0_full_bio | 0.0223 | 0.0052 | 0.9467 / 0.9744 | 78 | 0.9744 | 0.1614 | 35 | 0.9714 |
| P1_no_marks | 0.0233 | 0.0036 | 0.9472 / 0.9661 | 59 | 0.9661 | 0.1210 | 28 | 0.9643 |

Nested-CV recalibration (`scripts/run_publication_suite.py`): raw logistic probabilities are already well-calibrated (`ECE 0.0052`, top decile `0.947 / 0.974`). Nested Isotonic reduces `ECE` to `0.0030` and lifts `p ≥ 0.90` recall (`0.1614 → 0.1868`, 92 calls at precision `0.9565`; 4-non-K562 transfer `17` calls at precision `0.9412`) at a slight ranking cost (`AP 0.6827` from step ties).

### Head-to-head vs 1 kb CNN, Borzoi (524 kb), and Enformer (196 kb) on the exact same 10,356 K562 LOCO split

| model | category | AP | AUC | <20 kb | 20–100 kb | 100 kb–1 Mb | >=1 Mb | ΔAP vs Enformer (G) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Borzoi 524 kb (`D`) | no ChIP + Borzoi | 0.6500 | 0.9379 | 0.8256 | 0.6458 | 0.2059 | 0.0664 | -0.0332 [-0.0549, -0.0111] |
| Enformer 196 kb (`G`) | no ChIP + Enformer | 0.6832 | 0.9438 | 0.8481 | 0.6781 | 0.2304 | 0.1500 | (reference) |
| Enformer + ChIP (`H`) | measured ChIP + Enformer | 0.6943 | 0.9462 | 0.8635 | 0.6874 | 0.2236 | 0.1269 | +0.0111 [+0.0011, +0.0224] |
| **P1_no_marks (v0.4.0)** | strictly zero-ChIP | **0.6772** | 0.9421 | 0.8284 | 0.6870 | **0.3196** | **0.5038** | -0.0061 [-0.0414, +0.0306] |
| **P1_no_marks + 1kbCNN** | zero-ChIP + 1 kb CNN | **0.6815** | 0.9426 | 0.8299 | 0.6902 | **0.3188** | **0.4460** | -0.0018 [-0.0387, +0.0351] |
| **P0_full_bio (v0.4.0)** | measured ChIP | **0.7001** | 0.9444 | 0.8539 | 0.7032 | 0.3087 | 0.3791 | +0.0168 [-0.0191, +0.0542] |
| **P2_full_bio_3d (v0.4.0)** | measured ChIP + Micro-C | **0.7042** | 0.9482 | 0.8476 | **0.7133** | **0.3212** | **0.7440** | +0.0210 [-0.0173, +0.0635] |

### Zero-shot transfer to held-out cell types

The K562-fitted models — no retraining, no target labels — applied to `heldout_5ct.npz` (4,378 pairs / 190 positives, 0 pair overlap with training). Because **62.1% of positives (118/190) are in-domain K562**, the **4 genuinely held-out non-K562 cell types** (`WTC11, HCT116, GM12878, Jurkat`; 2,460 pairs / 72 positives) are reported separately:

| stratum / cell type | n | pos | P0 AP | P0 AUC | P1 (zero-ChIP) AP | P2 (+K562 3D) AP | P0 calls @ 0.90 | P0 prec @ 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **all 5 cells (pooled)** | 4,378 | 190 | **0.7151** | 0.9546 | 0.6572 | 0.7037 | 18 | **0.9444** |
| **4 non-K562 cells (pooled)** | 2,460 | 72 | **0.6753** | 0.9772 | 0.5945 | 0.6391 | 9 | 0.8889 |
| K562 (in-domain held-out) | 1,918 | 118 | 0.7595 | 0.9387 | 0.7146 | 0.7668 | 9 | 1.0000 |
| WTC11 | 1,921 | 15 | 0.6398 | 0.9712 | 0.6266 | 0.6666 | 0 | – |
| HCT116 | 396 | 34 | 0.6418 | 0.9461 | 0.5755 | 0.5613 | 6 | 0.8333 |
| GM12878 | 68 | 16 | 0.9143 | 0.9651 | 0.8547 | 0.9525 | 3 | 1.0000 |
| Jurkat | 75 | 7 | 0.6137 | 0.9076 | 0.8917 | 0.6860 | 0 | – |

Under ENCODE-rE2G `direct_vs_indirect_negative` sample weighting, `P0_full_bio` reaches **0.7209** (K562 LOCO), **0.7630** (`all5` transfer), and **0.7103** (`4-non-K562` transfer).

## Biological findings explained by the model

1. **Housekeeping / ubiquitous gene buffering (Fulco et al. 2019; Bergman et al. 2022)**: Ubiquitously expressed genes (`n=4,804`) have a **5.1× lower** positive rate (`1.42%` vs `7.26%`) and **5.9× lower** `>=20 kb` positive rate (`0.77%` vs `4.56%`) than non-ubiquitous genes (`n=5,552`), reflected by strong negative weights on `gene_ctx5` (`ubiq_frac`, `-0.6405`) and `gene_ctx1` (`prom_dnase`, `-0.4617`). `P0_full_bio` reaches **0.7384** AP (**0.6461** distal `>=20 kb`) on non-ubiquitous genes vs `0.4624` (`0.1249` distal) on ubiquitous genes.
2. **Intervening promoter competition & 3D looping (`>=20 kb`)**: **60.7%** of true distal enhancers (`167/275`) have **zero intervening protein-coding TSSs** (median `0`, mean `1.86`) vs **6.7%** of distal negatives (median `8`, mean `14.62`), while carrying **3.4×** higher genome-wide cCRE competition share and **2.4×** higher Micro-C 10 kb O/E (`2.25` vs `0.94`).
3. **Classic literature loci (out-of-fold)**: On **MYC** (105 pairs, 9 pos up to 1.96 Mb), `D0` collapses (`0.0840`), `P0` reaches `0.5230`, and **`P2_full_bio_3d` jumps to `0.8234`** (rescued by 10.38× Micro-C 10 kb O/E at 1.85 Mb with 0 intervening TSSs). On **GATA1** and **HDAC6** (Fulco et al. 2019), `P0` reaches **1.0000** AP (`P1`: `1.0000` / `0.8333`); on **HBE1** (β-globin LCR), `P0` reaches **0.9821**.

## Reproduce

```bash
python -m pip install -e ".[dev]"

linearbaseline fullbio  --config configs/fullbio.json  --out results
linearbaseline transfer --config configs/transfer.json --out results
python scripts/run_publication_suite.py
pytest -q
```

`results/fullbio_loco.json` and `results/heldout_transfer.json` are the committed outputs, and `tests/test_fullbio.py` enforces exact sync with this README plus bit-identical zero-ChIP invariance for `P1_no_marks`.

## Status

- [x] real-data full-bio arm (LOCO, gene-cluster CI, strict zero-ChIP `P1_no_marks` + `nobio` controls)
- [x] full 9-threshold operating-point tables (LOCO + `all5` + `4-non-K562` transfer)
- [x] nested-CV Platt & Isotonic recalibration (`scripts/run_publication_suite.py`)
- [x] head-to-head table against 1 kb CNN, Borzoi, and Enformer on identical 23-fold LOCO splits
- [x] Micro-C 3D contact arm (`P2_full_bio_3d`) + biological mechanism & locus explanations
- [ ] state-specific arm retrained per cell type (needs matched multi-cell CRISPRi labels)

## Citation & license

See `CITATION.cff`. Upstream data: Gasperini et al., *Cell* 2019 (doi:10.1016/j.cell.2019.11.001), Fulco et al., *Nature Genetics* 2019 (doi:10.1038/s41588-019-0537-1), and ENCODE-rE2G (Gschwind et al.). Code and packaged matrices: MIT (`LICENSE`).
