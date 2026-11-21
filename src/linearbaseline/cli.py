"""Command line entry point.

    linearbaseline fullbio --config configs/fullbio.json --out results

The package ships one arm on purpose: the full-bio logistic regression on the
EPCrisprBenchmark K562 CRE->gene task, evaluated leave-one-chromosome-out.
Everything mechanical (feature assembly, in-fold fitting, bootstrap) lives in
``linearbaseline.fullbio``; this module only parses arguments and writes JSON.
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
from .fullbio import load_fullbio, run_arms

DEFAULTS: Dict = {
    "data": "data/fullbio/features.npz",
    "C": 1.0,
    "bootstrap": 1000,
    "seed": 7,
}


def load_config(path: str | None) -> Dict:
    cfg = dict(DEFAULTS)
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
    with open(outdir / "fullbio_loco.json", "w", encoding="utf-8") as handle:
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
    print(f"wrote {outdir / 'fullbio_loco.json'}")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
