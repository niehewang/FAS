from __future__ import annotations

import numpy as np


def information_matrix(blocks, selected, ridge=1e-6):
    """Return H = ridge I + sum_p G_p^T G_p.

    blocks: sequence of arrays with shape [d_p, K].
    selected: iterable of integer probe indices.
    """
    if not blocks:
        raise ValueError("blocks must be non-empty")
    k = blocks[0].shape[1]
    H = ridge * np.eye(k, dtype=float)
    for idx in selected:
        G = np.asarray(blocks[idx], dtype=float)
        H += G.T @ G
    return H


def _logdet_psd(M):
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        return -np.inf
    return float(val)


def greedy_logdet_select(blocks, budget, ridge=1e-6, lazy=False):
    """Greedy D-optimal probe selection.

    Returns (selected_indices, marginal_gains).
    `lazy` is reserved for a future priority-queue implementation; the shipped
    reference implementation uses exact greedy evaluation for transparency.
    """
    if budget <= 0:
        return [], []
    n = len(blocks)
    if budget > n:
        raise ValueError(f"budget={budget} exceeds number of probes={n}")
    k = blocks[0].shape[1]
    H = ridge * np.eye(k, dtype=float)
    selected = []
    gains = []
    remaining = set(range(n))

    for _ in range(budget):
        base = _logdet_psd(H)
        best_idx, best_gain = None, -np.inf
        for idx in remaining:
            G = np.asarray(blocks[idx], dtype=float)
            candidate = H + G.T @ G
            gain = _logdet_psd(candidate) - base
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        selected.append(best_idx)
        gains.append(float(best_gain))
        G = np.asarray(blocks[best_idx], dtype=float)
        H += G.T @ G
        remaining.remove(best_idx)
    return selected, gains


def dictionary_geometry(A):
    """Return stability- and angle-oriented dictionary diagnostics.

    `sigma_min_raw` is the smallest singular value of the actual dictionary and
    is the quantity appearing in the NNLS perturbation bound.
    `sigma_min_unit` is computed after unit-normalizing each selected column and
    diagnoses angular separability independently of column energy. Mutual
    coherence is also computed on this unit-column dictionary.
    """
    A = np.asarray(A, dtype=float)
    raw_svals = np.linalg.svd(A, compute_uv=False)
    sigma_min_raw = float(raw_svals[-1]) if len(raw_svals) else 0.0
    sigma_max_raw = float(raw_svals[0]) if len(raw_svals) else 0.0
    condition_number_raw = float(sigma_max_raw / sigma_min_raw) if sigma_min_raw > 1e-12 else float("inf")
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    An = A / np.maximum(norms, 1e-12)
    unit_svals = np.linalg.svd(An, compute_uv=False)
    sigma_min_unit = float(unit_svals[-1]) if len(unit_svals) else 0.0
    gram = np.abs(An.T @ An)
    np.fill_diagonal(gram, 0.0)
    coherence = float(gram.max()) if gram.size else 0.0
    return {
        "sigma_min_raw": sigma_min_raw,
        "sigma_min_unit": sigma_min_unit,
        "sigma_min": sigma_min_unit,  # backwards-compatible alias
        "condition_number_raw": condition_number_raw,
        "coherence": coherence,
    }


def greedy_logdet_select_partitioned(blocks, budget, groups, quotas=None, ridge=1e-6):
    """Greedy D-optimal selection under partition quotas.

    Parameters
    ----------
    blocks : sequence of [d_p, K] probe blocks.
    budget : total number of probes.
    groups : group label per probe (e.g., functional domain/intervention type).
    quotas : optional dict {group: max_count}. If omitted, budget is split as
        evenly as possible across observed groups.

    This is the practical FAS selector for heterogeneous probe pools. It keeps
    the D-optimal marginal gain while preventing a few high-energy groups from
    monopolizing the online query budget.
    """
    if len(groups) != len(blocks):
        raise ValueError("groups must align with blocks")
    if budget <= 0:
        return [], []
    if budget > len(blocks):
        raise ValueError(f"budget={budget} exceeds number of probes={len(blocks)}")
    uniq = list(dict.fromkeys(groups))
    if quotas is None:
        base, extra = divmod(budget, len(uniq))
        quotas = {g: base + (i < extra) for i, g in enumerate(uniq)}
    else:
        quotas = dict(quotas)
    if sum(int(quotas.get(g, 0)) for g in uniq) < budget:
        raise ValueError("partition quotas cannot accommodate requested budget")
    k = blocks[0].shape[1]
    H = ridge * np.eye(k, dtype=float)
    selected, gains = [], []
    remaining = set(range(len(blocks)))
    counts = {g: 0 for g in uniq}
    for _ in range(budget):
        base_val = _logdet_psd(H)
        best_idx, best_gain = None, -np.inf
        for idx in remaining:
            g = groups[idx]
            if counts.get(g, 0) >= int(quotas.get(g, 0)):
                continue
            G = np.asarray(blocks[idx], dtype=float)
            gain = _logdet_psd(H + G.T @ G) - base_val
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        if best_idx is None:
            raise ValueError("partition quotas made selection infeasible")
        selected.append(best_idx); gains.append(float(best_gain))
        G = np.asarray(blocks[best_idx], dtype=float)
        H += G.T @ G
        counts[groups[best_idx]] += 1
        remaining.remove(best_idx)
    return selected, gains
