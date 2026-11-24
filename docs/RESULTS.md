# Results (v0.4.0)

Committed outputs of:

```bash
linearbaseline fullbio  --config configs/fullbio.json  --out results
linearbaseline transfer --config configs/transfer.json --out results
python scripts/run_publication_suite.py
```

→ `results/fullbio_loco.json`, `results/heldout_transfer.json`, and `results/publication_suite.json`.

## 1. EPCrisprBenchmark K562, 23-fold LOCO (10,356 pairs / 471 positives)

| arm | dims | AP | 95% CI (gene boot) | AUC | <20 kb | 20–100 kb | 100 kb–1 Mb | >=1 Mb |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| P2_full_bio_3d | 42 | **0.7042** | [0.6547, 0.7519] | 0.9482 | 0.8476 | **0.7133** | **0.3212** | **0.7440** |
| P0_full_bio | 38 | **0.7001** | [0.6495, 0.7472] | 0.9444 | **0.8539** | 0.7032 | 0.3087 | 0.3791 |
| P0_nobio | 34 | 0.7007 | [0.6505, 0.7483] | 0.9444 | 0.8531 | 0.7149 | 0.3073 | 0.4373 |
| P1_no_marks (zero-ChIP) | 25 | **0.6772** | [0.6240, 0.7276] | 0.9421 | 0.8284 | 0.6870 | 0.3196 | 0.5038 |
| P1_no_marks_nobio | 23 | 0.6745 | [0.6203, 0.7245] | 0.9415 | 0.8256 | 0.6848 | 0.3311 | 0.5091 |
| D0_distance_only | 2 | 0.4210 | [0.3666, 0.4803] | 0.8931 | 0.6335 | 0.2095 | 0.0186 | 0.0073 |

Paired gene-cluster bootstrap deltas (1,000 resamples):
* `delta_marks (P0 - P1)`: **+0.0229**, 95% CI `[+0.0080, +0.0382]`, `P(>0) = 0.998`.
* `delta_3d (P2 - P0)`: **+0.0042**, 95% CI `[-0.0088, +0.0170]`, `P(>0) = 0.740` (concentrated in `20–100 kb` +0.0101, `100 kb–1 Mb` +0.0125, and `>=1 Mb` +0.3650).

### 1.1 Full 9-threshold operating points (K562 LOCO)

#### `P2_full_bio_3d` (42 cols)

| p ≥ | calls | TP | FP | precision | recall | specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 339 | 260 | 79 | 0.7670 | 0.5520 | 0.9920 | 0.6420 |
| 0.60 | 283 | 231 | 52 | 0.8163 | 0.4904 | 0.9947 | 0.6127 |
| 0.70 | 226 | 191 | 35 | 0.8451 | 0.4055 | 0.9965 | 0.5481 |
| 0.75 | 195 | 170 | 25 | 0.8718 | 0.3609 | 0.9975 | 0.5105 |
| 0.80 | 154 | 139 | 15 | 0.9026 | 0.2951 | 0.9985 | 0.4448 |
| 0.85 | 125 | 116 | 9 | 0.9280 | 0.2463 | 0.9991 | 0.3893 |
| 0.90 | 88 | 85 | 3 | 0.9659 | 0.1805 | 0.9997 | 0.3041 |
| 0.95 | 43 | 42 | 1 | 0.9767 | 0.0892 | 0.9999 | 0.1634 |
| 0.99 | 4 | 4 | 0 | 1.0000 | 0.0085 | 1.0000 | 0.0168 |

#### `P0_full_bio` (38 cols, portable)

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

#### `P1_no_marks` (25 cols, strictly zero-ChIP)

| p ≥ | calls | TP | FP | precision | recall | specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 300 | 235 | 65 | 0.7833 | 0.4989 | 0.9934 | 0.6096 |
| 0.60 | 243 | 198 | 45 | 0.8148 | 0.4204 | 0.9954 | 0.5546 |
| 0.70 | 193 | 166 | 27 | 0.8601 | 0.3524 | 0.9973 | 0.5000 |
| 0.75 | 167 | 148 | 19 | 0.8862 | 0.3142 | 0.9981 | 0.4639 |
| 0.80 | 135 | 124 | 11 | 0.9185 | 0.2633 | 0.9989 | 0.4092 |
| 0.85 | 95 | 90 | 5 | 0.9474 | 0.1911 | 0.9995 | 0.3180 |
| 0.90 | 59 | 57 | 2 | 0.9661 | 0.1210 | 0.9998 | 0.2151 |
| 0.95 | 28 | 27 | 1 | 0.9643 | 0.0573 | 0.9999 | 0.1082 |
| 0.99 | 1 | 1 | 0 | 1.0000 | 0.0021 | 1.0000 | 0.0042 |

### 1.2 Calibration & nested-CV recalibration

| arm / method | LOCO AP | Brier | ECE | top decile pred / obs | LOCO @ 0.90 (c / prec / rec) | TR5 @ 0.90 (c / prec) | TR4nc @ 0.90 (c / prec) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| P0_full_bio (raw) | 0.7001 | 0.0223 | 0.0052 | 0.947 / 0.974 | 78 / 0.974 / 0.161 | 18 / 0.944 | 9 / 0.889 |
| P0_full_bio (Platt) | 0.6980 | 0.0225 | 0.0038 | 0.944 / 0.978 | 45 / 0.978 / 0.093 | 10 / 0.900 | 4 / 0.750 |
| P0_full_bio (Isotonic) | 0.6827 | 0.0227 | 0.0030 | 0.960 / 0.957 | 92 / 0.957 / 0.187 | 34 / 0.971 | 17 / 0.941 |
| P1_no_marks (raw) | 0.6772 | 0.0233 | 0.0036 | 0.947 / 0.966 | 59 / 0.966 / 0.121 | 14 / 1.000 | 4 / 1.000 |
| P1_no_marks (Platt) | 0.6764 | 0.0234 | 0.0032 | 0.942 / 0.976 | 42 / 0.976 / 0.087 | 8 / 1.000 | 1 / 1.000 |

## 2. Zero-shot transfer (`heldout_5ct.npz`, 4,378 pairs / 190 positives)

Because in-domain K562 accounts for 118/190 (62.1%) of held-out positives, both **all 5 cells** and **4 non-K562 cells** (2,460 pairs / 72 positives) are reported:

| arm | all-5 AP [95% CI] | all-5 AUC | **4 non-K562 AP [95% CI]** | 4 non-K562 AUC | all-5 @0.90 (c / prec) | 4 non-K562 @0.90 (c / prec) |
| --- | --- | ---: | --- | ---: | --- | --- |
| P0_full_bio | **0.7151** [0.6358, 0.7865] | 0.9546 | **0.6753** [0.5316, 0.8073] | 0.9772 | 18 / 0.9444 | 9 / 0.8889 |
| P0_nobio | 0.6733 [0.5883, 0.7512] | 0.9463 | 0.6545 [0.5047, 0.7879] | 0.9749 | 15 / 0.9333 | 8 / 0.8750 |
| P2_full_bio_3d | 0.7037 [0.6103, 0.7878] | 0.9542 | 0.6391 [0.4862, 0.8019] | 0.9760 | 23 / 0.8696 | 11 / 0.7273 |
| P1_no_marks | 0.6572 [0.5728, 0.7353] | 0.9344 | 0.5945 [0.4468, 0.7466] | 0.9599 | 14 / 1.0000 | 4 / 1.0000 |
| P1_no_marks_nobio | 0.6229 [0.5384, 0.7040] | 0.9257 | 0.5859 [0.4327, 0.7299] | 0.9577 | 11 / 0.9091 | 4 / 0.7500 |

### 2.1 Full 9-threshold operating points (Zero-shot transfer)

#### `P0_full_bio` — all 5 cells (4,378 pairs / 190 pos) & 4 non-K562 (2,460 pairs / 72 pos)

| p ≥ | all5 calls | all5 TP/FP | all5 prec | all5 rec | 4nc calls | 4nc TP/FP | 4nc prec | 4nc rec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 100 | 85 / 15 | 0.8500 | 0.4474 | 42 | 32 / 10 | 0.7619 | 0.4444 |
| 0.60 | 84 | 73 / 11 | 0.8690 | 0.3842 | 39 | 31 / 8 | 0.7949 | 0.4306 |
| 0.70 | 61 | 57 / 4 | 0.9344 | 0.3000 | 28 | 24 / 4 | 0.8571 | 0.3333 |
| 0.75 | 51 | 48 / 3 | 0.9412 | 0.2526 | 23 | 20 / 3 | 0.8696 | 0.2778 |
| 0.80 | 42 | 41 / 1 | 0.9762 | 0.2158 | 20 | 19 / 1 | 0.9500 | 0.2639 |
| 0.85 | 27 | 26 / 1 | 0.9630 | 0.1368 | 12 | 11 / 1 | 0.9167 | 0.1528 |
| 0.90 | 18 | 17 / 1 | 0.9444 | 0.0895 | 9 | 8 / 1 | 0.8889 | 0.1111 |
| 0.95 | 10 | 9 / 1 | 0.9000 | 0.0474 | 4 | 3 / 1 | 0.7500 | 0.0417 |
| 0.99 | 2 | 2 / 0 | 1.0000 | 0.0105 | 0 | 0 / 0 | 0.0000 | 0.0000 |

#### `P1_no_marks` (zero-ChIP) — all 5 cells & 4 non-K562

| p ≥ | all5 calls | all5 TP/FP | all5 prec | all5 rec | 4nc calls | 4nc TP/FP | 4nc prec | 4nc rec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 88 | 73 / 15 | 0.8295 | 0.3842 | 33 | 24 / 9 | 0.7273 | 0.3333 |
| 0.60 | 61 | 54 / 7 | 0.8852 | 0.2842 | 23 | 20 / 3 | 0.8696 | 0.2778 |
| 0.70 | 46 | 44 / 2 | 0.9565 | 0.2316 | 18 | 17 / 1 | 0.9444 | 0.2361 |
| 0.75 | 41 | 40 / 1 | 0.9756 | 0.2105 | 16 | 15 / 1 | 0.9375 | 0.2083 |
| 0.80 | 30 | 29 / 1 | 0.9667 | 0.1526 | 11 | 10 / 1 | 0.9091 | 0.1389 |
| 0.85 | 20 | 19 / 1 | 0.9500 | 0.1000 | 6 | 5 / 1 | 0.8333 | 0.0694 |
| 0.90 | 14 | 14 / 0 | 1.0000 | 0.0737 | 4 | 4 / 0 | 1.0000 | 0.0556 |
| 0.95 | 5 | 5 / 0 | 1.0000 | 0.0263 | 1 | 1 / 0 | 1.0000 | 0.0139 |
| 0.99 | 0 | 0 / 0 | 0.0000 | 0.0000 | 0 | 0 / 0 | 0.0000 | 0.0000 |

### 2.2 Per-cell breakdown

| cell type | n | pos | P0 AP | P1 AP | P2 AP | P0 calls @ 0.90 | P0 prec @ 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K562 | 1,918 | 118 | 0.7595 | 0.7146 | 0.7668 | 9 | 1.0000 |
| WTC11 | 1,921 | 15 | 0.6398 | 0.6266 | 0.6666 | 0 | – |
| HCT116 | 396 | 34 | 0.6418 | 0.5755 | 0.5613 | 6 | 0.8333 |
| GM12878 | 68 | 16 | 0.9143 | 0.8547 | 0.9525 | 3 | 1.0000 |
| Jurkat | 75 | 7 | 0.6137 | 0.8917 | 0.6860 | 0 | – |

## 3. Head-to-head vs 1 kb CNN, Borzoi, Enformer & weighted evaluation (`publication_suite.json`)

* **Same 10,356 K562 23-fold LOCO split**: `P1_no_marks` (`0.6772`) + 1 kb CNN surrogate marks = **`0.6815`** (zero measured ChIP, matching 196 kb Enformer `0.6832`); `P0_full_bio` (`0.7001`) and `P2_full_bio_3d` (`0.7042`) outperform `Enformer + ChIP` (`0.6943`).
* **ENCODE-rE2G `direct_vs_indirect_negative` weighted AP**: `P0_full_bio` reaches **0.7209** (K562 LOCO), **0.7630** (`all5` transfer), **0.7103** (`4-non-K562` transfer); `P1_no_marks` reaches **0.6977** / **0.7103** / **0.6442**. Quantitative `-EffectSize` Spearman correlation across all K562 pairs: `0.1956` (`P2`), `0.1908` (`P0`), `0.1884` (`P1`).

## 4. Biological explanations & classic locus case studies

1. **Housekeeping / ubiquitous gene buffering**: `ubiquitous` genes (`n=4,804`) have 5.1× lower positive rate (`1.42%` vs `7.26%`) and 5.9× lower `>=20 kb` positive rate (`0.77%` vs `4.56%`) than `non-ubiquitous` genes (`n=5,552`). `P0_full_bio` achieves **0.7384** AP (**0.6461** distal) on non-ubiquitous genes vs `0.4624` (`0.1249` distal) on ubiquitous genes.
2. **Intervening TSS competition (`>=20 kb`)**: 60.7% of distal positives have 0 intervening protein-coding TSSs (mean `1.86`) vs 6.7% of distal negatives (mean `14.62`), alongside 3.4× higher genome-wide cCRE share and 2.4× higher Micro-C 10 kb O/E.
3. **Locus case studies (out-of-fold)**: `MYC` (105 pairs, 9 pos up to 1.96 Mb): `D0 0.0840 → P1 0.3765 → P0 0.5230 → P2 0.8234` (1.85 Mb enhancer has 10.38× Micro-C O/E and 0 intervening TSSs); `GATA1`: `P0/P1/P2 = 1.0000`; `HDAC6`: `P0/P2 = 1.0000`, `P1 = 0.8333`; `HBE1`: `P0 = 0.9821`, `P1 = 0.9571`; `CCND1` (`HCT116`): `P0 = 0.7611` vs `P2 = 0.6667` (illustrating cell-type-specific 3D contact mismatch).
