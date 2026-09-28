#!/usr/bin/env python3
"""Collect per-domain utilities for counterfactual subset descendants."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--eval-dir',required=True);ap.add_argument('--output',default='results/raw/subset_utilities.csv');args=ap.parse_args()
    plan=list(csv.DictReader(open(args.plan,encoding='utf-8',newline='')))
    mapping={'empty':''}
    for r in plan: mapping[r['calibration_id']]=r['subset']
    rows=[]; missing=[]
    for eid,subset in mapping.items():
        p=Path(args.eval_dir)/f'{eid}.json'
        if not p.exists(): missing.append(str(p));continue
        obj=json.loads(p.read_text(encoding='utf-8'))
        for domain,v in obj.get('by_domain',{}).items():
            rows.append({'subset':subset,'utility':float(v['utility']),'domain':domain,'n':int(v['n']),'source':str(p)})
        rows.append({'subset':subset,'utility':float(obj['overall_utility']),'domain':'ALL','n':int(obj['n']),'source':str(p)})
    if missing: raise SystemExit(f'missing evaluation outputs: {missing[:5]} (total {len(missing)})')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subset','utility','domain','n','source']);w.writeheader();w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {out}')
if __name__=='__main__':main()
