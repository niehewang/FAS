from __future__ import annotations

import numpy as np

from .conformal import conformal_pvalue


def signal_pvalue(null_signal_scores, test_signal_score):
    """Conformal p-value for a *large* ancestry-specific signal score.

    Under the base-like/no-task-delta null, unusually large ||y|| is evidence
    that there is enough functional signal to attempt ancestry decomposition.
    A small p-value therefore means "signal detected".
    """
    return conformal_pvalue(null_signal_scores, test_signal_score)


def selective_decision(
    signal_p: float,
    open_p: float | None = None,
    alpha_signal: float = 0.05,
    alpha_open: float = 0.05,
):
    """Three-state selective FAS decision.

    Returns one of:
      - low_signal: ancestry coordinates should not be interpreted;
      - bank_insufficient: signal exists but candidate bank is inadequate;
      - decomposable: signal exists and candidate bank is not rejected.
    """
    if not (0.0 <= signal_p <= 1.0):
        raise ValueError("signal_p must be in [0, 1]")
    if signal_p >= alpha_signal:
        return {
            "state": "low_signal",
            "report_coordinates": False,
            "signal_p": float(signal_p),
            "open_p": None if open_p is None else float(open_p),
        }
    if open_p is None:
        raise ValueError("open_p is required once signal gate passes")
    if not (0.0 <= open_p <= 1.0):
        raise ValueError("open_p must be in [0, 1]")
    if open_p < alpha_open:
        state = "bank_insufficient"
        report = False
    else:
        state = "decomposable"
        report = True
    return {
        "state": state,
        "report_coordinates": report,
        "signal_p": float(signal_p),
        "open_p": float(open_p),
    }


def signal_score(y, weights=None):
    """Weighted L2 norm used by the low-signal gate."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if weights is None:
        return float(np.linalg.norm(y))
    W = np.asarray(weights, dtype=float)
    yw = y * W if W.ndim == 1 else W @ y
    return float(np.linalg.norm(yw))
