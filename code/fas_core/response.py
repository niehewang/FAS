from __future__ import annotations

import numpy as np


def mean_embedding(samples):
    """Mean of already-computed output embeddings, shape [m, d] -> [d]."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        return x
    return x.mean(axis=0)


def interventional_response(base_embeddings, edited_embeddings):
    return mean_embedding(edited_embeddings) - mean_embedding(base_embeddings)


def centered_response(model_response, base_model_response):
    return np.asarray(model_response, dtype=float) - np.asarray(base_model_response, dtype=float)


def global_ancestor_norms(task_fields, eps=1e-12):
    """task_fields shape [N probes, K ancestors, d]."""
    x = np.asarray(task_fields, dtype=float)
    if x.ndim != 3:
        raise ValueError("task_fields must have shape [N,K,d]")
    norms = np.sqrt(np.sum(x * x, axis=(0, 2)))
    return np.maximum(norms, eps)


def probe_blocks(task_fields, eps=1e-12):
    """Convert [N,K,d] task fields into N blocks [d,K] with global norms."""
    x = np.asarray(task_fields, dtype=float)
    norms = global_ancestor_norms(x, eps=eps)
    scaled = x / norms[None, :, None]
    return [scaled[p].T.copy() for p in range(scaled.shape[0])], norms


def stack_selected(task_fields, selected, norms=None, eps=1e-12):
    """Build A_Q with shape [B*d, K] from [N,K,d]."""
    x = np.asarray(task_fields, dtype=float)
    if norms is None:
        norms = global_ancestor_norms(x, eps=eps)
    blocks = [(x[p] / norms[:, None]).T for p in selected]
    return np.concatenate(blocks, axis=0)


def functional_mixture_reference(task_fields, weights, eps=1e-12):
    """Exact response target and FAS reference coordinates for y=sum_i w_i S_i.

    Because FAS globally normalizes ancestor column i by c_i=||S_i|| over the
    full candidate pool, the exact NNLS coefficient in the normalized dictionary
    is gamma_i=w_i*c_i, not w_i.  The simplex coordinate is normalize(gamma).
    Returns (centered_response [N,d], gamma [K], pi [K], norms [K]).
    """
    x=np.asarray(task_fields,dtype=float)
    w=np.asarray(weights,dtype=float).reshape(-1)
    if x.ndim!=3: raise ValueError("task_fields must have shape [N,K,d]")
    if w.shape[0]!=x.shape[1]: raise ValueError("weights length must equal number of ancestors")
    if np.any(w<0): raise ValueError("functional mixture weights must be nonnegative")
    norms=global_ancestor_norms(x,eps=eps)
    y=np.einsum("k,nkd->nd",w,x)
    gamma=w*norms
    total=float(gamma.sum())
    pi=gamma/total if total>eps else np.zeros_like(gamma)
    return y,gamma,pi,norms
