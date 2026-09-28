#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
DOMAINS=['math','code','medical','science']

def norm(s): return ' '.join(str(s or '').split())
def sig(r): return hashlib.sha256((norm(r.get('prompt'))+'\n'+norm(r.get('response'))).encode('utf-8')).hexdigest()
def readj(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--output',required=True); ap.add_argument('--per-domain',type=int,default=500); ap.add_argument('--seed',type=int,default=20260924); a=ap.parse_args()
    P=Path(a.project); rows=[]; meta={}
    for d in DOMAINS:
        raw=readj(P/f'data/training/formal_raw/{d}.jsonl'); train=readj(P/f'data/training/formal/{d}.jsonl')
        tr={sig(r) for r in train}; hold=[r for r in raw if sig(r) not in tr]
        hold.sort(key=lambda r: hashlib.sha256(f"{a.seed}:{d}:{r.get('id','')}:{norm(r.get('prompt'))}".encode()).hexdigest())
        if len(hold)<a.per_domain: raise SystemExit(f'{d}: only {len(hold)} held-out raw rows, need {a.per_domain}')
        use=hold[:a.per_domain]
        for j,r in enumerate(use): rows.append({'id':f'kd_{d}_{j:04d}','domain':d.capitalize(),'prompt':r['prompt'],'source_domain':d,'source_id':r.get('id','')})
        meta[d]={'raw_n':len(raw),'train_n':len(train),'heldout_n':len(hold),'used_n':len(use)}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta['total']=len(rows); meta['sha256']=hashlib.sha256(out.read_bytes()).hexdigest(); (out.with_suffix(out.suffix+'.meta.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
