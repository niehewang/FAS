from __future__ import annotations

import numpy as np


def conformal_pvalue(calibration_scores, test_score):
    """Conservative +1 conformal rank p-value for large-is-anomalous scores."""
    cal = np.asarray(calibration_scores, dtype=float).reshape(-1)
    if cal.size == 0:
        raise ValueError("at least one calibration score is required")
    return float((1 + np.sum(cal >= float(test_score))) / (cal.size + 1))


def bank_sufficient(calibration_scores, test_score, alpha=0.05):
    p = conformal_pvalue(calibration_scores, test_score)
    return {"p_value": p, "bank_sufficient": bool(p >= alpha), "alpha": alpha}
