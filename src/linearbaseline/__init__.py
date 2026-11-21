"""Linear baselines for cis-regulatory sequence-to-function prediction.

The package is intentionally small: one feature assembly, one logistic
regression, one evaluation harness.  Its job is to be the honest floor that any
sequence model has to beat, and to make that floor reproducible.

The shipped arm is the *full-bio* CRE->gene model on the EPCrisprBenchmark K562
set (leave-one-chromosome-out, AP 0.6715).
"""

__version__ = "0.2.0"

from .fullbio import (
    MARK_ALL,
    distance_mask,
    feature_names,
    load_fullbio,
    loco_oof,
    run_arms,
    summarize,
)
from .metrics import average_precision, auroc, bootstrap_ap_ci

__all__ = [
    "__version__",
    "MARK_ALL",
    "average_precision",
    "auroc",
    "bootstrap_ap_ci",
    "distance_mask",
    "feature_names",
    "load_fullbio",
    "loco_oof",
    "run_arms",
    "summarize",
]
