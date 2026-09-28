from __future__ import annotations

import numpy as np


def make_sibling_task_fields(n_probes=300, k=4, d=8, seed=0):
    """Synthetic controlled IFRF bank with heterogeneous probe discriminativeness."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n_probes, 1, d))
    unique = rng.normal(size=(n_probes, k, d))
    informativeness = rng.lognormal(mean=-0.4, sigma=0.9, size=(n_probes, 1, 1))
    # Siblings share a substantial component, while selected probes expose unique directions.
    fields = 0.75 * shared + informativeness * unique
    return fields


def make_descendant(A, support, weights=None, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    k = A.shape[1]
    gamma = np.zeros(k)
    if weights is None:
        weights = np.ones(len(support)) / len(support)
    gamma[np.asarray(support, dtype=int)] = np.asarray(weights, dtype=float)
    y = A @ gamma + noise * rng.normal(size=A.shape[0])
    return y, gamma
