#!/usr/bin/env python3
"""Create a deterministic negative-control probe file by mismatching edited queries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--output',required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--within-domain',action='store_true');args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    rng=np.random.default_rng(args.seed);out=[]
    if args.within_domain:
        groups={}
        for i,r in enumerate(rows):groups.setdefault(r.get('domain',''),[]).append(i)
        mapping={}
        for ids in groups.values():
            perm=np.array(ids);rng.shuffle(perm)
            if len(ids)>1 and any(a==b for a,b in zip(ids,perm)):
                perm=np.roll(perm,1)
            mapping.update(dict(zip(ids,perm.tolist())))
    else:
        ids=list(range(len(rows)));perm=np.array(ids);rng.shuffle(perm)
        if len(ids)>1 and any(a==b for a,b in zip(ids,perm)):perm=np.roll(perm,1)
        mapping=dict(zip(ids,perm.tolist()))
    for i,r in enumerate(rows):
        donor=rows[mapping[i]];z=dict(r);z['probe_id']=str(r['probe_id'])+'__pairshuffle';z['edited_query']=donor['edited_query'];z['negative_control']='probe_pair_shuffle';z['donor_probe_id']=donor['probe_id'];out.append(z)
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf-8');print(p,len(out))
if __name__=='__main__':main()
