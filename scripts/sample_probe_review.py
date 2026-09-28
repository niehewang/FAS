#!/usr/bin/env python3
"""Create a small stratified TSV for human audit of automatically proposed probes."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='data/probes/formal_interventions_candidate.jsonl');ap.add_argument('--output',default='data/governance/probe_review_sample.tsv');ap.add_argument('--per-domain',type=int,default=20);ap.add_argument('--seed',type=int,default=20260924);args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    rng=random.Random(args.seed);out=[]
    for d in sorted({r['domain'] for r in rows}):
        arr=[r for r in rows if r['domain']==d]
        # Prefer review-needed rows, but include some auto candidates.
        arr.sort(key=lambda r:(not r.get('needs_review',False),r['type'],r['probe_id']))
        groups={}
        for r in arr:groups.setdefault(r['type'],[]).append(r)
        chosen=[]
        types=sorted(groups)
        while len(chosen)<min(args.per_domain,len(arr)):
            progressed=False
            for t in types:
                g=groups[t]
                if g:
                    chosen.append(g.pop(rng.randrange(len(g))));progressed=True
                    if len(chosen)>=args.per_domain:break
            if not progressed:break
        out.extend(chosen)
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    fields=['probe_id','domain','type','needs_review','base_query','edited_query','decision','reviewer_note']
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
        for r in out:w.writerow({k:r.get(k,'') for k in fields})
    print('PROBE_REVIEW_SAMPLE',len(out),p)
if __name__=='__main__':main()
