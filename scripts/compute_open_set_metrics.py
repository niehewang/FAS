#!/usr/bin/env python3
"""Compute conditional open-set AUROC, FPR@95TPR, and false-abstention rates.

Input CSV columns:
  score,is_unknown[,setting,p_open,p_signal,state]
Higher `score` must mean more evidence that the ancestor bank is insufficient.

Important selective-inference rule:
The bank-sufficiency test is only meaningful *conditional on the target passing
the low-signal gate*. Rows explicitly marked state=low_signal are therefore
excluded by default. If a p_signal column is present and state is absent,
--alpha-signal is used to exclude p_signal >= alpha_signal. This mirrors the
paper's Low-Signal -> Bank-Insufficient/Decomposable decision sequence.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def fpr_at_tpr(y,s,target=.95):
    fpr,tpr,_=roc_curve(y,s)
    idx=np.where(tpr>=target)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def signal_eligible(r, alpha_signal):
    state=(r.get("state") or "").strip().lower()
    if state:
        return state != "low_signal"
    ps=(r.get("p_signal") or "").strip()
    if ps:
        return float(ps) < alpha_signal
    return True


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--output",default="results/derived/open_set_metrics.csv")
    ap.add_argument("--alphas",default="0.01,0.05,0.10"); ap.add_argument("--alpha-signal",type=float,default=0.05)
    ap.add_argument("--include-low-signal",action="store_true",help="diagnostic only; not the paper's conditional open-set metric")
    args=ap.parse_args()
    rows=list(csv.DictReader(open(args.input,encoding="utf-8",newline="")))
    before=len(rows)
    if not args.include_low_signal:
        rows=[r for r in rows if signal_eligible(r,args.alpha_signal)]
    groups=sorted(set(r.get("setting","all") or "all" for r in rows)); alphas=[float(x) for x in args.alphas.split(",")]
    outs=[]
    for g in groups+["ALL"]:
        rr=rows if g=="ALL" else [r for r in rows if (r.get("setting","all") or "all")==g]
        row={"setting":g,"n":len(rr),"n_excluded_low_signal":before-len(rows) if g=="ALL" else "","auroc":"","fpr95":""}
        if rr:
            y=np.asarray([int(r["is_unknown"]) for r in rr]); s=np.asarray([float(r["score"]) for r in rr])
            if len(set(y.tolist()))==2:
                row["auroc"]=float(roc_auc_score(y,s)); row["fpr95"]=fpr_at_tpr(y,s)
            known=[r for r in rr if int(r["is_unknown"])==0 and (r.get("p_open") or "").strip()]
        else:
            known=[]
        for a in alphas:
            row[f"false_abstain_delta{a:g}"]=sum(float(r["p_open"])<a for r in known)/len(known) if known else ""
        outs.append(row)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in outs:
        for k in r:
            if k not in fields: fields.append(k)
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(outs)
    print(f"Wrote {out}; retained {len(rows)}/{before} rows after signal conditioning")
if __name__=="__main__": main()
