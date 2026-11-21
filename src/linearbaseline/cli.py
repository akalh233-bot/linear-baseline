"""Command line entry point.

    linearbaseline synthetic --out data/synthetic.npz
    linearbaseline train     --config configs/synthetic.json --out results
    linearbaseline ablations --config configs/synthetic.json --out results

Every subcommand is deterministic given its seed and writes machine-readable
JSON next to any human-readable table.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import __version__
from .data import load_npz, make_synthetic, train_test_split_by_locus
from .features import build_features
from .metrics import average_precision, auroc, brier, log_loss, bootstrap_ap_ci
from .model import fit_logistic, select_C, top_coefficients

DEFAULTS: Dict = {
    "data": {"kind": "synthetic", "n_loci": 4000, "rows_per_locus": 2, "seq_len": 200, "seed": 0},
    "features": {"kmer_k": 3},
    "split": {"n_folds": 5, "test_frac": 0.2, "seed": 42},
    "model": {"C_grid": [0.01, 0.1, 1.0, 10.0], "max_iter": 2000, "class_weight": None},
    "eval": {"n_bootstrap": 1000, "seed": 7},
}


def _deep_update(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None) -> Dict:
    if path is None:
        return dict(DEFAULTS)
    with open(path, "r", encoding="utf-8") as handle:
        return _deep_update(DEFAULTS, json.load(handle))


def _materialise_data(cfg: Dict, force: bool = False) -> Dict[str, np.ndarray]:
    dcfg = cfg["data"]
    if dcfg.get("kind", "synthetic") == "synthetic" and dcfg.get("path") and not force:
        path = Path(dcfg["path"])
        if path.exists():
            return load_npz(str(path))
    if dcfg.get("kind", "synthetic") == "synthetic":
        return make_synthetic(
            n_loci=int(dcfg["n_loci"]),
            rows_per_locus=int(dcfg["rows_per_locus"]),
            seq_len=int(dcfg["seq_len"]),
            seed=int(dcfg["seed"]),
        )
    if not dcfg.get("path"):
        raise ValueError("config.data.path is required for kind != synthetic")
    return load_npz(str(dcfg["path"]))


def _evaluate(y: np.ndarray, prob: np.ndarray, locus: np.ndarray, n_boot: int, seed: int) -> Dict:
    out = {
        "ap": average_precision(y, prob),
        "auroc": auroc(y, prob),
        "log_loss": log_loss(y, prob),
        "brier": brier(y, prob),
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
    }
    if out["n_pos"] and out["n_pos"] < out["n"]:
        ci = bootstrap_ap_ci(y, prob, locus, n_boot=n_boot, seed=seed)
        out.update({"ap_ci_low": ci["ci_low"], "ap_ci_high": ci["ci_high"], "n_boot_valid": ci["n_valid"]})
    return out


def cmd_synthetic(args: argparse.Namespace) -> int:
    data = make_synthetic(
        n_loci=args.loci,
        rows_per_locus=args.rows_per_locus,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"wrote {out}  rows={len(data['y'])}  loci={data['locus_id'].max() + 1}")
    print(f"positives {int(data['y'].sum())} / {len(data['y'])} = {data['y'].mean():.3f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    started = time.time()
    data = _materialise_data(cfg)
    X, names = build_features(data, cfg["features"])
    y = np.asarray(data["y"])
    locus = np.asarray(data["locus_id"])

    tr, te = train_test_split_by_locus(
        locus, test_frac=float(cfg["split"]["test_frac"]), seed=int(cfg["split"]["seed"])
    )
    C, cv_scores = select_C(
        X[tr],
        y[tr],
        locus[tr],
        cfg["model"]["C_grid"],
        n_splits=int(cfg["split"]["n_folds"]),
        max_iter=int(cfg["model"]["max_iter"]),
        verbose=args.verbose,
    )
    model = fit_logistic(
        X[tr], y[tr], C=C, max_iter=int(cfg["model"]["max_iter"]), class_weight=cfg["model"].get("class_weight")
    )
    prob = model.predict_proba(X[te])[:, 1]

    metrics = _evaluate(y[te], prob, locus[te], int(cfg["eval"]["n_bootstrap"]), int(cfg["eval"]["seed"]))
    metrics.update({"C": C, "cv_ap": cv_scores, "n_features": int(X.shape[1]), "seconds": round(time.time() - started, 2)})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=float)
    with open(outdir / "run.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": __version__,
                "config": cfg,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "command": sys.argv,
            },
            handle,
            indent=2,
            default=float,
        )
    np.savez_compressed(
        outdir / "predictions.npz",
        prob=prob.astype(np.float32),
        y=y[te].astype(np.int8),
        locus_id=locus[te],
    )
    coefs = top_coefficients(model, names, k=25)
    with open(outdir / "coefficients.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "coefficient"])
        writer.writerows(coefs)

    print(f"loci train/test: {len(np.unique(locus[tr]))}/{len(np.unique(locus[te]))}")
    print(f"features: {X.shape[1]}   best C: {C}")
    print(f"held-out AP : {metrics['ap']:.4f}  [{metrics.get('ap_ci_low', float('nan')):.4f}, {metrics.get('ap_ci_high', float('nan')):.4f}]")
    print(f"held-out AUC: {metrics['auroc']:.4f}   log-loss {metrics['log_loss']:.4f}")
    print(f"wrote {outdir}")
    return 0


LADDER: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("majority", ()),
    ("distance-only", ("distance",)),
    ("kmer-only", ("kmer",)),
    ("kmer+distance", ("kmer", "distance")),
    ("full", ("kmer", "distance", "atac", "context")),
)


def cmd_ablations(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = _materialise_data(cfg)
    y = np.asarray(data["y"])
    locus = np.asarray(data["locus_id"])
    tr, te = train_test_split_by_locus(
        locus, test_frac=float(cfg["split"]["test_frac"]), seed=int(cfg["split"]["seed"])
    )

    rows: List[Dict] = []
    for label, groups in LADDER:
        if not groups:  # majority-class reference
            prevalence = float(np.mean(y[tr]))
            prob = np.full(len(te), prevalence, dtype=np.float64)
            rows.append(
                {
                    "arm": label,
                    "features": 0,
                    "C": None,
                    "ap": average_precision(y[te], prob),
                    "auroc": float("nan"),
                    "log_loss": log_loss(y[te], prob),
                }
            )
            continue
        try:
            X, names = build_features(data, cfg["features"], only=groups)
        except ValueError as exc:
            print(f"skip {label}: {exc}")
            continue
        C, _ = select_C(
            X[tr], y[tr], locus[tr], cfg["model"]["C_grid"], n_splits=int(cfg["split"]["n_folds"])
        )
        model = fit_logistic(X[tr], y[tr], C=C, max_iter=int(cfg["model"]["max_iter"]))
        prob = model.predict_proba(X[te])[:, 1]
        rows.append(
            {
                "arm": label,
                "features": int(X.shape[1]),
                "C": C,
                "ap": average_precision(y[te], prob),
                "auroc": auroc(y[te], prob),
                "log_loss": log_loss(y[te], prob),
            }
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "ablations.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=float)

    print(f"{'arm':<16}{'feat':>6}{'C':>8}{'AP':>8}{'AUC':>8}{'logloss':>10}")
    for row in rows:
        C = "-" if row["C"] is None else f"{row['C']:g}"
        print(f"{row['arm']:<16}{row['features']:>6}{C:>8}{row['ap']:>8.4f}{row['auroc']:>8.4f}{row['log_loss']:>10.4f}")
    print(f"wrote {outdir / 'ablations.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linearbaseline", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="write a synthetic npz bundle")
    syn.add_argument("--out", default="data/synthetic.npz")
    syn.add_argument("--loci", type=int, default=4000)
    syn.add_argument("--rows-per-locus", type=int, default=2)
    syn.add_argument("--seq-len", type=int, default=200)
    syn.add_argument("--seed", type=int, default=0)
    syn.set_defaults(func=cmd_synthetic)

    train = sub.add_parser("train", help="fit and evaluate the linear baseline")
    train.add_argument("--config", default=None)
    train.add_argument("--out", default="results")
    train.add_argument("--verbose", action="store_true")
    train.set_defaults(func=cmd_train)

    abl = sub.add_parser("ablations", help="feature-group ablation ladder")
    abl.add_argument("--config", default=None)
    abl.add_argument("--out", default="results")
    abl.set_defaults(func=cmd_ablations)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
