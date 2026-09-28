from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def support_metrics(true_support, pred_support):
    t, p = set(true_support), set(pred_support)
    tp = len(t & p)
    precision = tp / len(p) if p else (1.0 if not t else 0.0)
    recall = tp / len(t) if t else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_support": float(t == p),
    }


def l1_coordinate_error(pred, target):
    return float(np.abs(np.asarray(pred) - np.asarray(target)).sum())


def spearman_safe(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")
