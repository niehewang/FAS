#!/usr/bin/env python3
"""Run a synthetic end-to-end FAS sanity check without downloading any models."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from fas_core.active_probe import greedy_logdet_select, dictionary_geometry
from fas_core.response import probe_blocks, stack_selected
from fas_core.decompose import fas_decompose
from fas_core.synthetic import make_sibling_task_fields, make_descendant


def main():
    fields = make_sibling_task_fields(n_probes=300, k=4, d=8, seed=7)
    blocks, norms = probe_blocks(fields)
    rng = np.random.default_rng(7)
    print("Synthetic FAS sanity demo")
    print("budget | act raw | rnd raw | act unit | rnd unit | active L1 | random L1")
    for budget in [8, 16, 32, 64]:
        active, _ = greedy_logdet_select(blocks, budget=budget, ridge=1e-4)
        random = rng.choice(len(blocks), size=budget, replace=False).tolist()
        A_active = stack_selected(fields, active, norms)
        A_random = stack_selected(fields, random, norms)
        y_a, gamma = make_descendant(A_active, support=[0, 2], weights=[0.65, 0.35], seed=budget)
        y_r, _ = make_descendant(A_random, support=[0, 2], weights=[0.65, 0.35], seed=budget)
        fa = fas_decompose(A_active, y_a)
        fr = fas_decompose(A_random, y_r)
        target = gamma / gamma.sum()
        la = np.abs(fa["pi"] - target).sum()
        lr = np.abs(fr["pi"] - target).sum()
        ga = dictionary_geometry(A_active)
        gr = dictionary_geometry(A_random)
        print(f"{budget:>6} | {ga['sigma_min_raw']:>7.4f} | {gr['sigma_min_raw']:>7.4f} | {ga['sigma_min_unit']:>8.4f} | {gr['sigma_min_unit']:>8.4f} | {la:>9.4f} | {lr:>9.4f}")


if __name__ == "__main__":
    main()
