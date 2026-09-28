#!/usr/bin/env python3
"""Chance-level parent-F1 under genealogy-label permutation.

This does not rerun models. It keeps each prediction fixed and randomizes the
identity of the true support while preserving its cardinality within the same
candidate bank. It is a diagnostic for whether reported support recovery is
meaningfully above label-chance.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_list(s):
    s=str(s or '').strip()
    if not s:return []
    if s.startswith('['):return [str(x) for x in json.loads(s)]
    return [x for x in s.split(';') if x]

def f1(t,p):
    t=set(t);p=set(p);tp=len(t&p)
    pr=tp/len(p) if p else (1.0 if not t else 0.0)
    re=tp/len(t) if t else 1.0
    return 2*pr*re/(pr+re) if pr+re else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--n-perm',type=int,default=2000);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--paper-output',default='results/templates/negative_controls.csv');args=ap.parse_args()
    df=pd.read_csv(args.input,keep_default_na=False);rng=np.random.default_rng(args.seed);vals=[]
    for _ in range(args.n_perm):
        scores=[]
        for _,r in df.iterrows():
            names=parse_list(r.get('parent_names',''));true=parse_list(r.get('true_support',''));pred=parse_list(r.get('pred_support',''))
            if not pred and str(r.get('pred_pi','')).strip():
                pi=[float(x) for x in parse_list(r.get('pred_pi',''))];pred=[n for n,x in zip(names,pi) if x>0.01]
            if not names:continue
            k=min(len(true),len(names));perm=list(rng.choice(names,size=k,replace=False)) if k else []
            scores.append(f1(perm,pred))
        if scores: vals.append(float(np.mean(scores)))
    if not vals: raise SystemExit('no usable rows')
    mean=float(np.mean(vals));lo=float(np.quantile(vals,.025));hi=float(np.quantile(vals,.975))
    pp=Path(args.paper_output)
    old=pd.read_csv(pp,keep_default_na=False) if pp.exists() else pd.DataFrame(columns=['control','false_positive_or_f1','signal_pass_or_expected','interpretation'])
    old=old[old.control.astype(str)!='Genealogy label permutation']
    row={'control':'Genealogy label permutation','false_positive_or_f1':mean,'signal_pass_or_expected':f'95% [{lo:.3f},{hi:.3f}]','interpretation':'chance Parent-F1'}
    pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_csv(pp,index=False)
    print(f'permutation Parent-F1 mean={mean:.4f}, 95%=[{lo:.4f},{hi:.4f}] -> {pp}')
if __name__=='__main__':main()
