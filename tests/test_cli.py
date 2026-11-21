import json

import numpy as np

from linearbaseline.cli import main


def test_end_to_end_synthetic_train(tmp_path):
    data_path = tmp_path / "data" / "synthetic.npz"
    assert main(["synthetic", "--out", str(data_path), "--loci", "300", "--seq-len", "120"]) == 0
    assert data_path.exists()

    config = {
        "data": {"kind": "synthetic", "path": str(data_path), "n_loci": 300, "rows_per_locus": 2, "seq_len": 120},
        "split": {"n_folds": 3, "test_frac": 0.25, "seed": 11},
        "model": {"C_grid": [0.1, 1.0], "max_iter": 800},
        "eval": {"n_bootstrap": 300, "seed": 3},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outdir = tmp_path / "results"
    assert main(["train", "--config", str(config_path), "--out", str(outdir)]) == 0

    metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["C"] in (0.1, 1.0)
    assert 0.0 <= metrics["ap"] <= 1.0
    assert metrics["ap"] > 0.5  # synthetic world is learnable
    assert metrics["n_features"] == 64 + 4 + 1 + 8 + 3
    assert (outdir / "coefficients.csv").read_text(encoding="utf-8").splitlines()[0] == "feature,coefficient"

    preds = np.load(outdir / "predictions.npz")
    assert preds["prob"].shape == preds["y"].shape


def test_ablation_ladder_runs(tmp_path):
    data_path = tmp_path / "data" / "synthetic.npz"
    main(["synthetic", "--out", str(data_path), "--loci", "400", "--seq-len", "120"])
    config = {
        "data": {"kind": "synthetic", "path": str(data_path), "n_loci": 400, "rows_per_locus": 2, "seq_len": 120},
        "split": {"n_folds": 3, "test_frac": 0.25, "seed": 5},
        "model": {"C_grid": [1.0], "max_iter": 500},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outdir = tmp_path / "results"
    assert main(["ablations", "--config", str(config_path), "--out", str(outdir)]) == 0

    rows = json.loads((outdir / "ablations.json").read_text(encoding="utf-8"))
    arms = [r["arm"] for r in rows]
    assert arms[0] == "majority"
    assert "kmer-only" in arms
    assert "full" in arms
    ap_by_arm = {r["arm"]: r["ap"] for r in rows}
    assert ap_by_arm["full"] > ap_by_arm["majority"]
