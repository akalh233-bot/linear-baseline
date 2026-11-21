"""Linear baselines for cis-regulatory sequence-to-function prediction.

The package is intentionally small: one feature builder, one logistic
regression, one evaluation harness.  Its job is to be the honest floor that any
sequence model has to beat, and to make that floor reproducible.
"""

__version__ = "0.1.0"

from .metrics import average_precision, auroc, bootstrap_ap_ci, log_loss, brier
from .model import fit_logistic, label_balanced_folds, select_C

__all__ = [
    "__version__",
    "average_precision",
    "auroc",
    "bootstrap_ap_ci",
    "log_loss",
    "brier",
    "fit_logistic",
    "label_balanced_folds",
    "select_C",
]
