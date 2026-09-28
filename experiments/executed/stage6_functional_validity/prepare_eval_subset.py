#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
DOMAINS=['Math','Medical','Science']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--n-per-domain',type=int,default=96); ap.add_argument('--seed',default='20260926'); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    chosen=[]; man={'source':a.input,'seed':a.seed,'n_per_domain':a.n_per_domain,'domains':{}}
    for d in DOMAINS:
        rr=[r for r in rows if str(r.get('domain',''))==d]
        if len(rr)<a.n_per_domain: raise SystemExit(f'{d}: only {len(rr)} utility rows, require {a.n_per_domain}')
        def key(r):
            ident=str(r.get('id',''))+'\n'+str(r.get('prompt',''))
            return hashlib.sha256((a.seed+'|'+d+'|'+ident).encode()).hexdigest()
        take=sorted(rr,key=key)[:a.n_per_domain]
        chosen.extend(take); man['domains'][d]={'available':len(rr),'selected':len(take),'ids':[str(r.get('id','')) for r in take]}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in chosen: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    Path(a.manifest).write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(chosen)} utility rows -> {out}')
if __name__=='__main__': main()
