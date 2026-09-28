#!/usr/bin/env python3
"""Select the FAS support threshold on held-out validation compositions only.

Input CSV follows results/raw_templates/decomposition_predictions.csv and must
contain true_support, parent_names and pred_pi. The selected threshold maximizes
mean Parent-F1; ties prefer the larger threshold (more conservative support).
Never run this script on test descendants.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def lst(s):
    s=(s or '').strip()
    if not s:return []
    if s.startswith('['):return [str(x) for x in json.loads(s)]
    return [x for x in s.split(';') if x]

def floats(s): return np.array([float(x) for x in lst(s)])
def f1(t,p):
    t=set(t);p=set(p); tp=len(t&p); pr=tp/len(p) if p else (1.0 if not t else 0.0); re=tp/len(t) if t else 1.0
    return 2*pr*re/(pr+re) if pr+re else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('validation_csv');ap.add_argument('--grid',default='0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2');ap.add_argument('--output',default='runs/calibration/support_threshold.json');args=ap.parse_args()
    rows=list(csv.DictReader(open(args.validation_csv,encoding='utf-8',newline=''))); grid=[float(x) for x in args.grid.split(',')]
    scores=[]
    for th in grid:
        vals=[]
        for r in rows:
            names=lst(r['parent_names']); pi=floats(r['pred_pi']); true=lst(r['true_support']); pred=[n for n,v in zip(names,pi) if v>th]; vals.append(f1(true,pred))
        scores.append({'threshold':th,'mean_parent_f1':float(np.mean(vals)) if vals else float('nan')})
    best=max(scores,key=lambda x:(x['mean_parent_f1'],x['threshold']))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'selected':best,'grid':scores,'n_validation':len(rows)},indent=2),encoding='utf-8')
    print(json.dumps({'selected':best,'n_validation':len(rows)},indent=2))
if __name__=='__main__':main()
