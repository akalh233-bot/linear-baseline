"""Command line entry point.

    linearbaseline fullbio  --config configs/fullbio.json --out results
    linearbaseline transfer --train data/fullbio/features.npz --out results

The package ships one arm on purpose: the full-bio logistic regression on the
EPCrisprBenchmark K562 CRE->gene task, evaluated leave-one-chromosome-out, plus
the zero-shot transfer of that fitted model to the held-out five-cell-type
table.  Everything mechanical (feature assembly, in-fold fitting, bootstrap)
lives in ``linearbaseline.fullbio``; this module only parses arguments and
writes JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from . import __version__
from .fullbio import load_fullbio, run_arms, run_transfer

DEFAULTS: Dict = {
    "data": "data/fullbio/features.npz",
    "C": 1.0,
    "bootstrap": 1000,
    "seed": 7,
}

TRANSFER_DEFAULTS: Dict = {
    "train": "data/fullbio/features.npz",
    "data": "data/fullbio/heldout_5ct.npz",
    "C": 1.0,
    "bootstrap": 1000,
    "seed": 7,
}


def load_config(path: str | None, defaults: Dict | None = None) -> Dict:
    cfg = dict(DEFAULTS if defaults is None else defaults)
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError(f"{path}: expected a flat JSON object")
    for key in override:
        if key not in cfg:
            raise ValueError(f"{path}: unknown key {key!r}; allowed: {sorted(cfg)}")
    cfg.update(override)
    return cfg


def cmd_fullbio(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_config(args.config)
    if args.data is not None:
        cfg["data"] = args.data
    if args.C is not None:
        cfg["C"] = args.C
    if args.bootstrap is not None:
        cfg["bootstrap"] = args.bootstrap
    if args.seed is not None:
        cfg["seed"] = args.seed

    d = load_fullbio(cfg["data"])
    arms = run_arms(d, C=float(cfg["C"]), n_boot=int(cfg["bootstrap"]), seed=int(cfg["seed"]))

    payload = {
        "data": str(cfg["data"]),
        "n": int(len(d["y"])),
        "pos": int(d["y"].sum()),
        "n_chrom": int(len(np.unique(d["chrom"]))),
        "n_gene": int(len(np.unique(d["gene"]))),
        "C": float(cfg["C"]),
        "arms": arms,
        "seconds": round(time.time() - started, 2),
        "versions": {
            "package": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "command": ["linearbaseline", *sys.argv[1:]],
        },
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "fullbio_loco.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=float)

    header = f"{'arm':<18}{'AP':>8}{'95% CI':>20}{'AUC':>8}{'<20kb':>8}{'20-100kb':>10}{'100kb-1Mb':>11}"
    print(header)
    for name in ("P0_full_bio", "P1_no_marks", "D0_distance_only"):
        a = arms[name]["all"]
        ci = f"[{a['ap_ci_low']:.4f},{a['ap_ci_high']:.4f}]"
        row = [a["ap"], a["auc"]]
        for label in ("<20kb", "20-100kb", "100kb-1Mb"):
            row.append(arms[name].get(label, {}).get("ap", float("nan")))
        print(f"{name:<18}{row[0]:>8.4f}{ci:>20}{row[1]:>8.4f}{row[2]:>8.4f}{row[3]:>10.4f}{row[4]:>11.4f}")
    for name in ("P0_full_bio", "P1_no_marks"):
        bits = []
        for t in (0.90, 0.95):
            r = _at_threshold(arms[name]["all"], t)
            bits.append(f"p>={t:.2f}: {r['flagged']} calls, prec {r['precision']:.3f}, rec {r['recall']:.3f}")
        print(f"{name:<18}" + "  |  ".join(bits))
    print(f"wrote {outdir / 'fullbio_loco.json'}")
    return 0


def _at_threshold(row: Dict, threshold: float) -> Dict:
    """The threshold row closest to ``threshold`` (tables are small, lists are fine)."""
    return min(row["thresholds"], key=lambda r: abs(r["threshold"] - threshold))


def cmd_transfer(args: argparse.Namespace) -> int:
    started = time.time()
    cfg = load_config(args.config, TRANSFER_DEFAULTS)
    for key in ("train", "data"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.C is not None:
        cfg["C"] = args.C
    if args.bootstrap is not None:
        cfg["bootstrap"] = args.bootstrap
    if args.seed is not None:
        cfg["seed"] = args.seed

    train = load_fullbio(cfg["train"])
    test = load_fullbio(cfg["data"])
    arms = run_transfer(train, test, C=float(cfg["C"]), n_boot=int(cfg["bootstrap"]), seed=int(cfg["seed"]))
    cell = test["cell"].astype(str)

    payload = {
        "train": str(cfg["train"]),
        "data": str(cfg["data"]),
        "n": int(len(test["y"])),
        "pos": int(test["y"].sum()),
        "n_gene": int(len(np.unique(test["gene"]))),
        "C": float(cfg["C"]),
        "cells": {
            label: {"n": int((cell == label).sum()), "pos": int(test["y"][cell == label].sum())}
            for label in sorted(np.unique(cell))
        },
        "arms": arms,
        "note": (
            "Zero-shot: PCA, gene-context medians, expression centring and logistic "
            "weights are fitted on the K562 training table only. The target cell line "
            "supplies its own measured marks, accessibility and neighbourhood columns "
            "(label-free), while gene_ctx/expr are K562-derived proxies."
        ),
        "seconds": round(time.time() - started, 2),
        "versions": {
            "package": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "command": ["linearbaseline", *sys.argv[1:]],
        },
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "heldout_transfer.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=float)

    header = f"{'arm':<14}{'cell':<9}{'n':>6}{'pos':>5}{'AP':>8}{'c@90':>6}{'p@90':>7}{'c@95':>6}{'p@95':>7}"
    print(header)
    for name in ("P0_full_bio", "P1_no_marks"):
        for label, cell_summary in arms[name]["cells"].items():
            a = cell_summary["all"]
            r90, r95 = _at_threshold(a, 0.90), _at_threshold(a, 0.95)
            print(
                f"{name:<14}{label:<9}{a['n']:>6}{a['pos']:>5}{a['ap']:>8.4f}"
                f"{r90['flagged']:>6}{r90['precision']:>7.3f}{r95['flagged']:>6}{r95['precision']:>7.3f}"
            )
    print(f"wrote {outdir / 'heldout_transfer.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linearbaseline", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fb = sub.add_parser("fullbio", help="EPCrisprBenchmark K562 CRE->gene arm (LOCO)")
    fb.add_argument("--config", default=None, help="flat JSON config (see configs/fullbio.json)")
    fb.add_argument("--data", default=None, help="features.npz (default data/fullbio/features.npz)")
    fb.add_argument("--out", default="results")
    fb.add_argument("--C", type=float, default=None, help="logistic L2 strength (default 1.0)")
    fb.add_argument("--bootstrap", type=int, default=None, help="gene-cluster resamples (default 1000)")
    fb.add_argument("--seed", type=int, default=None)
    fb.set_defaults(func=cmd_fullbio)

    tr = sub.add_parser("transfer", help="zero-shot K562 -> 5 held-out cell types")
    tr.add_argument("--config", default=None, help="flat JSON config (see configs/transfer.json)")
    tr.add_argument("--train", default=None, help="K562 features.npz (default data/fullbio/features.npz)")
    tr.add_argument("--data", default=None, help="held-out table (default data/fullbio/heldout_5ct.npz)")
    tr.add_argument("--out", default="results")
    tr.add_argument("--C", type=float, default=None, help="logistic L2 strength (default 1.0)")
    tr.add_argument("--bootstrap", type=int, default=None, help="gene-cluster resamples (default 1000)")
    tr.add_argument("--seed", type=int, default=None)
    tr.set_defaults(func=cmd_transfer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
