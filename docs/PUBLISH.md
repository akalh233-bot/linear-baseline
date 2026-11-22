# Publishing to GitHub

Repository: <https://github.com/akalh233-bot/linear-baseline>
Local checkout: `E:\triomics\linear-baseline` (branch `main`, tracks `origin/main`).

**Status: pushed.** `origin/main` tracks the local branch and the working tree is
clean after each commit. The credential is cached by Git Credential Manager, so
later pushes are silent.

> If a commit is made from the sandboxed agent environment, `git push` may fail
> (`SEC_E_NO_CREDENTIALS` or a timeout to `github.com:443`) because the
> credential store and outbound network are not reachable there. The commit is
> still valid locally; run `git push` from a normal shell.

## Re-pushing after a change

```powershell
cd E:\triomics\linear-baseline
git status -sb        # expect "## main...origin/main"
git add -A
git commit -m "docs: ..."
git push
```

If a push fails with `remote: Repository not found`, check `git remote -v`: the
URL must be exactly `https://github.com/akalh233-bot/linear-baseline.git`.

## Recommended settings

| field | value |
| --- | --- |
| description | Honest linear baselines for cis-regulatory sequence-to-function prediction |
| topics | `regulatory-genomics` `tf-binding` `cre-gene` `baseline` `logistic-regression` `bioinformatics` |
| default branch | `main` |
| branch protection | require the `ci / test` check on `main` |

CI runs on every push to `main` and on pull requests (`.github/workflows/ci.yml`,
Python 3.10 and 3.12: `pytest` plus end-to-end `linearbaseline fullbio` and
`linearbaseline transfer` runs).

## What is verified locally

```text
pytest -q                          -> 11 passed
linearbaseline fullbio             -> P0 AP 0.6715 [0.6151, 0.7226]
                                      P1 AP 0.6507 [0.5937, 0.7020]
                                      D0 AP 0.4210 [0.3666, 0.4803]
                                      P0 @ p>=0.90: 72 calls, prec 0.889, rec 0.136
linearbaseline transfer            -> P0 pooled AP 0.6661 over 4,378 pairs / 190 pos
                                      p>=0.90: 19 calls, 18 TP, prec 0.947
```

The three APs, the nine-threshold tables and the calibration blocks are asserted
against the two committed JSON files and against `README.md` by tests, so a
silent drift fails CI rather than publishing a stale number.

## Committed vs regenerated

* **Committed:** `data/fullbio/features.npz` (3.4 MB) and
  `data/fullbio/heldout_5ct.npz` (0.92 MB), both with SHA-256 in
  `docs/DATA.md`; `results/fullbio_loco.json` and
  `results/heldout_transfer.json`. These must stay bit-identical for the README
  to be reproducible.
* **Ignored:** `data/*` other than the two packaged matrices, everything under
  `results/` except the two JSON files, `__pycache__`, `.pytest_cache`.

## If you ever need to redo the commit author

```powershell
git config user.name  "akalh233-bot"
git config user.email "akalh233@gmail.com"
git commit --amend --reset-author --no-edit
```
