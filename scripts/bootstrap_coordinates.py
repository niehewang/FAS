#!/usr/bin/env python3
"""Bootstrap FAS coordinates from precomputed target response replicates.

NPZ input must contain:
  A: [B*d, K]
  response_samples: [B, m, d]
Optional:
  ancestor_names: [K] strings
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fas_core.bootstrap import bootstrap_fas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_npz')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--n-boot',type=int,default=1000)
    ap.add_argument('--support-threshold',type=float,default=0.01)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--block-bootstrap',action='store_true')
    ap.add_argument('--output',default='results/derived/coordinate_uncertainty.csv')
    args=ap.parse_args()
    z=np.load(args.input_npz,allow_pickle=True)
    A=z['A']; X=z['response_samples']
    names=[str(x) for x in z['ancestor_names']] if 'ancestor_names' in z else [f'a{i}' for i in range(A.shape[1])]
    out=bootstrap_fas(A,X,n_boot=args.n_boot,support_threshold=args.support_threshold,seed=args.seed,block_bootstrap=args.block_bootstrap)
    rows=[]
    for i,n in enumerate(names):
        rows.append({'run_id':args.run_id,'ancestor':n,'mean':out['pi_mean'][i],'median':out['pi_median'][i],
                     'ci_low':out['pi_ci_low'][i],'ci_high':out['pi_ci_high'][i],
                     'support_frequency':out['support_frequency'][i]})
    p=ROOT/args.output; p.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    if p.exists():
        old=pd.read_csv(p); old=old[old['run_id'].astype(str)!=args.run_id]; df=pd.concat([old,df],ignore_index=True)
    df.to_csv(p,index=False)
    print(p)

if __name__=='__main__':main()
