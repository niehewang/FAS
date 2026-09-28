#!/usr/bin/env python3
"""Aggregate dictionary geometry and ancestry error with bootstrap CIs.

Input must contain sigma_min_raw, sigma_min_unit, coherence, coordinate_l1 and
may contain logdet,strategy,budget,seed. Output schema matches the paper CSV.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def bootstrap_corr(x,y,n=2000,seed=42):
    rng=np.random.default_rng(seed); vals=[]; m=len(x)
    if m<3: return (float('nan'),float('nan'))
    for _ in range(n):
        idx=rng.integers(0,m,m)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        c=spearmanr(x[idx],y[idx]).statistic
        if np.isfinite(c): vals.append(c)
    return tuple(np.quantile(vals,[.025,.975])) if vals else (float('nan'),float('nan'))


def relation(name,x,y):
    r=spearmanr(x,y).statistic if len(x)>=2 else float('nan'); lo,hi=bootstrap_corr(x,y)
    return {'quantity':name,'spearman':r,'ci_low':lo,'ci_high':hi}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--output',default='results/derived/geometry_mechanism.csv'); args=ap.parse_args()
    rows=list(csv.DictReader(open(args.input,encoding='utf-8',newline=''))); err=np.array([float(r['coordinate_l1']) for r in rows])
    outs=[
        relation('sigma_min_raw_vs_l1',np.array([float(r['sigma_min_raw']) for r in rows]),err),
        relation('sigma_min_unit_vs_l1',np.array([float(r['sigma_min_unit']) for r in rows]),err),
        relation('coherence_vs_l1',np.array([float(r['coherence']) for r in rows]),err),
    ]
    if rows and all((r.get('logdet') or '').strip() for r in rows):
        outs.append(relation('logdet_vs_l1',np.array([float(r['logdet']) for r in rows]),err))
    else: outs.append({'quantity':'logdet_vs_l1','spearman':'','ci_low':'','ci_high':''})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['quantity','spearman','ci_low','ci_high']); w.writeheader(); w.writerows(outs)
    print(f'Wrote {out}')
if __name__=='__main__': main()
