"""Internal evaluation metric computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, r2_score, roc_curve
from sklearn.preprocessing import label_binarize


def eval_classification_core(
    all_predicted: list[np.ndarray],
    all_true: list[np.ndarray],
    n_classes: int | None = None,
) -> pd.DataFrame:
    if n_classes is None:
        n_classes = all_predicted[0].shape[1] if all_predicted[0].ndim > 1 else 2

    probs = []
    for p in all_predicted:
        if p.ndim == 1:
            probs.append(p)
        elif p.max() <= 0:
            probs.append(np.exp(p))
        else:
            probs.append(p)

    if n_classes == 2:
        return _eval_binary(probs, all_true)
    return _eval_multiclass(probs, all_true, n_classes)


def _eval_binary(probs, all_true):
    records = []
    for fold, (p, t) in enumerate(zip(probs, all_true)):
        pos = p[:, 1] if p.ndim > 1 else p
        fpr, tpr, _ = roc_curve(t, pos)
        records.append({"fold": fold + 1, "AUC": auc(fpr, tpr)})
    return pd.DataFrame(records)


def _eval_multiclass(probs, all_true, n_classes):
    records = []
    for fold, (p, t) in enumerate(zip(probs, all_true)):
        t_bin = label_binarize(t, classes=list(range(n_classes)))
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(t_bin[:, c], p[:, c])
            records.append({"fold": fold + 1, "class": str(c), "AUC": auc(fpr, tpr)})
        fpr_mi, tpr_mi, _ = roc_curve(t_bin.ravel(), p.ravel())
        records.append({"fold": fold + 1, "class": "micro", "AUC": auc(fpr_mi, tpr_mi)})
        all_fpr = np.unique(np.concatenate(
            [roc_curve(t_bin[:, c], p[:, c])[0] for c in range(n_classes)]
        ))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            f_c, t_c, _ = roc_curve(t_bin[:, c], p[:, c])
            mean_tpr += np.interp(all_fpr, f_c, t_c)
        mean_tpr /= n_classes
        records.append({"fold": fold + 1, "class": "macro", "AUC": auc(all_fpr, mean_tpr)})
    return pd.DataFrame(records)


def eval_regression_core(
    all_predicted: list[np.ndarray],
    all_true: list[np.ndarray],
) -> pd.DataFrame:
    records = []
    for fold, (pred, true) in enumerate(zip(all_predicted, all_true)):
        pred_flat = pred.squeeze()
        records.append({
            "fold": fold + 1,
            "R2": r2_score(true, pred_flat),
            "correlation": np.corrcoef(true, pred_flat)[0, 1],
        })
    return pd.DataFrame(records)
