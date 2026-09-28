from __future__ import annotations

import numpy as np

from .decompose import fas_decompose


def bootstrap_fas(
    A,
    response_samples,
    *,
    n_boot: int = 1000,
    support_threshold: float = 0.01,
    seed: int = 0,
    block_bootstrap: bool = False,
):
    """Empirical FAS stability for a fixed selected probe set.

    Parameters
    ----------
    A : ndarray [B*d, K]
        Fixed ancestry dictionary for the selected probes.
    response_samples : ndarray [B, m, d]
        Centered target response replicate vectors per probe. Replicates should
        already represent edited-minus-base response samples (or paired-seed
        differences). Each bootstrap replicate resamples within-probe generation
        replicates, averages them, then performs FAS decomposition.
    block_bootstrap : bool
        If True, resample probe blocks as well. This is a sensitivity analysis,
        not a selection-adjusted confidence interval.
    """
    A = np.asarray(A, dtype=float)
    X = np.asarray(response_samples, dtype=float)
    if X.ndim != 3:
        raise ValueError("response_samples must have shape [B, m, d]")
    B, m, d = X.shape
    if A.shape[0] != B * d:
        raise ValueError("A row count must equal B*d")
    if m < 1 or n_boot < 1:
        raise ValueError("m and n_boot must be positive")

    rng = np.random.default_rng(seed)
    K = A.shape[1]
    pis = np.zeros((n_boot, K), dtype=float)
    rhos = np.zeros(n_boot, dtype=float)
    supports = np.zeros((n_boot, K), dtype=float)

    A_blocks = A.reshape(B, d, K)
    for b in range(n_boot):
        if block_bootstrap:
            block_ids = rng.integers(0, B, size=B)
        else:
            block_ids = np.arange(B)
        y_blocks = []
        a_blocks = []
        for j in block_ids:
            ids = rng.integers(0, m, size=m)
            y_blocks.append(X[j, ids].mean(axis=0))
            a_blocks.append(A_blocks[j])
        yb = np.concatenate(y_blocks, axis=0)
        Ab = np.concatenate(a_blocks, axis=0)
        out = fas_decompose(Ab, yb, support_threshold=support_threshold)
        pis[b] = out["pi"]
        rhos[b] = out["rho"]
        supports[b, out["support"]] = 1.0

    return {
        "pi_mean": pis.mean(axis=0),
        "pi_median": np.median(pis, axis=0),
        "pi_ci_low": np.quantile(pis, 0.025, axis=0),
        "pi_ci_high": np.quantile(pis, 0.975, axis=0),
        "support_frequency": supports.mean(axis=0),
        "rho_mean": float(rhos.mean()),
        "rho_ci_low": float(np.quantile(rhos, 0.025)),
        "rho_ci_high": float(np.quantile(rhos, 0.975)),
        "pi_samples": pis,
        "rho_samples": rhos,
    }
