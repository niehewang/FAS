#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,itertools,json
from pathlib import Path
PARENTS=['Math','Medical','Science']; W={'Math':0.2,'Medical':0.5,'Science':0.3}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); a=ap.parse_args(); rows=[]
    for k in range(1,4):
        for sub in itertools.combinations(PARENTS,k):
            rows.append({'calibration_id':'subset_'+'_'.join(sub),'subset':';'.join(sub),'parents':json.dumps(list(sub)),'weights':json.dumps({p:W[p] for p in sub},sort_keys=True),'weight_rule':'fixed'})
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['calibration_id','subset','parents','weights','weight_rule']); w.writeheader(); w.writerows(rows)
    print(p)
if __name__=='__main__':main()
