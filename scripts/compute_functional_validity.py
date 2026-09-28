#!/usr/bin/env python3
"""Compare construction/FAS quantities against counterfactual Shapley values.

Input CSV columns:
  case_id,domain,quantity,parent,value,shapley
where `quantity` may be "Merge weights", "Teacher exposure",
"Absolute-output decomposition", or "FAS coordinates". Correlations are
computed within each case/domain and then summarized by quantity to avoid
artificially inflating sample size by pooling unrelated simplexes.
"""
from __future__ import annotations
import argparse,csv,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--output',default='results/derived/functional_validity.csv'); args=ap.parse_args()
    rows=list(csv.DictReader(open(args.input,encoding='utf-8',newline='')))
    groups={}
    for r in rows: groups.setdefault((r['quantity'],r.get('case_id',''),r.get('domain','')),[]).append(r)
    per=[]
    for (q,c,d),rr in groups.items():
        x=np.array([float(r['value']) for r in rr]); y=np.array([float(r['shapley']) for r in rr])
        corr=spearmanr(x,y).statistic if len(rr)>=2 else math.nan
        per.append((q,c,d,corr))
    outs=[]
    for q in sorted(set(x[0] for x in per)):
        vals=np.array([x[3] for x in per if x[0]==q and np.isfinite(x[3])])
        outs.append({'quantity':q,'spearman':float(vals.mean()) if len(vals) else '', 'std':float(vals.std(ddof=1)) if len(vals)>1 else (0.0 if len(vals)==1 else ''),'n_cases':len(vals)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outs[0].keys()) if outs else ['quantity','spearman','std','n_cases']); w.writeheader(); w.writerows(outs)
    print(f'Wrote {out}')
if __name__=='__main__': main()
