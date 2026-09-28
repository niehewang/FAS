#!/usr/bin/env python3
"""Select active probes within one functional domain and return global indices."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.active_probe import greedy_logdet_select

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',required=True);ap.add_argument('--probes',required=True);ap.add_argument('--domain',required=True);ap.add_argument('--budget',type=int,default=32);ap.add_argument('--ridge',type=float,default=1e-3);ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
    inds=[i for i,r in enumerate(rows) if str(r.get('domain','')).lower()==args.domain.lower()]
    if len(inds)<args.budget: raise SystemExit(f'domain={args.domain}: only {len(inds)} probes < budget={args.budget}')
    z=np.load(args.bank);fields=np.asarray(z['task_fields'],float)
    if fields.shape[0]!=len(rows): raise ValueError(f'bank/probe count mismatch {fields.shape[0]} vs {len(rows)}')
    blocks=[fields[i] for i in inds]
    local,gains=greedy_logdet_select(blocks,args.budget,ridge=args.ridge)
    global_sel=[inds[i] for i in local]
    obj={'domain':args.domain,'budget':args.budget,'selected':global_sel,'marginal_gains':[float(x) for x in gains],'candidate_count':len(inds)}
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');print(p)
if __name__=='__main__':main()
