#!/usr/bin/env python3
"""Generate exact functional-mixture targets from a held-out response bank.

These targets are *not* claimed to be real composed models. They are a theory
sanity/death test: if y = sum_i w_i S_i exactly, FAS with globally normalized
columns has the exact coefficient gamma_i = w_i * c_i, where c_i is the fixed
global norm of ancestor i.  Therefore the reference simplex coordinate is
normalize(w_i c_i), not the raw construction weight w_i.

This isolates identifiability / active-probe geometry before expensive real
model merging or distillation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'code'))
from fas_core.response import functional_mixture_reference

DEFAULT_SPECS=[
    ('syn_math_code', {'Math':0.5,'Code':0.5}),
    ('syn_math_medical', {'Math':0.3,'Medical':0.7}),
    ('syn_code_science', {'Code':0.7,'Science':0.3}),
    ('syn_three_parent', {'Math':0.2,'Code':0.5,'Medical':0.3}),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',required=True,help='held-out/evaluation response bank NPZ')
    ap.add_argument('--output-dir',default='runs/pilot/synthetic_targets')
    ap.add_argument('--manifest',default='runs/pilot/synthetic_targets.csv')
    ap.add_argument('--noise-std',type=float,default=0.0,help='optional iid response noise for stress testing')
    ap.add_argument('--seed',type=int,default=123)
    args=ap.parse_args()

    z=np.load(args.bank,allow_pickle=False)
    fields=np.asarray(z['task_fields'],float) # [N,K,d], raw centered fields
    names=[str(x) for x in z['ancestor_names'].tolist()]
    name_to_i={n:i for i,n in enumerate(names)}
    outdir=ROOT/args.output_dir;outdir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    rows=[]
    for tid,spec in DEFAULT_SPECS:
        w=np.zeros(len(names),float)
        for n,v in spec.items():
            if n not in name_to_i: raise KeyError(f'{n} not in bank names {names}')
            w[name_to_i[n]]=float(v)
        y,gamma,pi,norms=functional_mixture_reference(fields,w)
        if args.noise_std>0: y=y+rng.normal(scale=args.noise_std,size=y.shape)
        p=outdir/f'{tid}.npz'
        np.savez_compressed(p,centered_response=y,construction_weights=w,true_gamma=gamma,true_pi=pi,ancestor_names=np.asarray(names))
        support=[names[i] for i,x in enumerate(w) if x>0]
        rows.append({
            'target_id':tid,
            'target_centered_npz':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            'true_support':';'.join(support),
            'true_pi':';'.join(f'{x:.12g}' for x in pi),
            'construction_weights':';'.join(f'{x:.12g}' for x in w),
            'reference_type':'exact_functional_mixture',
        })
    mp=ROOT/args.manifest;mp.parent.mkdir(parents=True,exist_ok=True)
    with mp.open('w',encoding='utf-8',newline='') as f:
        fw=csv.DictWriter(f,fieldnames=list(rows[0].keys()));fw.writeheader();fw.writerows(rows)
    print(json.dumps({'manifest':str(mp),'targets':len(rows),'ancestor_norms':dict(zip(names,map(float,norms)))},ensure_ascii=False))

if __name__=='__main__':main()
