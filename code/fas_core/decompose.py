from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _whiten(A, y, weights=None):
    if weights is None:
        return A, y
    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        if W.shape[0] != y.shape[0]:
            raise ValueError("1D weights must match observation dimension")
        return A * W[:, None], y * W
    return W @ A, W @ y


def fas_decompose(A, y, weights=None, support_threshold=0.01, eps=1e-12):
    """NNLS FAS decomposition.

    Parameters
    ----------
    A : [D, K] ancestry dictionary.
    y : [D] centered target response.
    weights : optional diagonal-vector or full whitening matrix.

    Returns coefficients gamma, normalized coordinates pi, explainedness rho,
    relative residual, and support indices. `support_threshold` is applied to
    the scale-free relative coordinate pi (not raw gamma).
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if A.shape[0] != y.shape[0]:
        raise ValueError("A and y have incompatible shapes")
    Aw, yw = _whiten(A, y, weights)
    gamma, residual_norm = nnls(Aw, yw)
    pred = Aw @ gamma
    denom = float(np.dot(yw, yw)) + eps
    rho = float(np.clip(np.dot(pred, pred) / denom, 0.0, 1.0))
    gamma_sum = float(gamma.sum())
    pi = gamma / gamma_sum if gamma_sum > eps else np.zeros_like(gamma)
    relative_residual = float(np.linalg.norm(yw - pred) / (np.linalg.norm(yw) + eps))
    support = np.flatnonzero(pi > support_threshold).tolist()
    return {
        "gamma": gamma,
        "pi": pi,
        "rho": rho,
        "unexplained": 1.0 - rho,
        "relative_residual": relative_residual,
        "residual_norm": float(residual_norm),
        "support": support,
    }
