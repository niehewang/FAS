#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from jsonl_utils_v1 import strict_read_jsonl,tolerant_source_read_jsonl,atomic_write_jsonl
DOMAINS=['math','code','medical','science']
def norm(s): return ' '.join(str(s or '').split())
def sig(r): return hashlib.sha256((norm(r.get('prompt'))+'\n'+norm(r.get('response'))).encode('utf-8')).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);ap.add_argument('--per-domain',type=int,default=500);ap.add_argument('--seed',type=int,default=20260924);a=ap.parse_args()
    P=Path(a.project);rows=[];meta={'reader':'stage5_v1_1_integrity_guard','malformed_raw_rows_excluded':{}}
    for d in DOMAINS:
        raw,bad=tolerant_source_read_jsonl(P/f'data/training/formal_raw/{d}.jsonl',max_bad=5)
        train=strict_read_jsonl(P/f'data/training/formal/{d}.jsonl')
        tr={sig(r) for r in train}; hold=[r for r in raw if sig(r) not in tr]
        hold.sort(key=lambda r: hashlib.sha256(f"{a.seed}:{d}:{r.get('id','')}:{norm(r.get('prompt'))}".encode()).hexdigest())
        if len(hold)<a.per_domain: raise SystemExit(f'{d}: only {len(hold)} held-out valid raw rows, need {a.per_domain}')
        use=hold[:a.per_domain]
        for j,r in enumerate(use): rows.append({'id':f'kd_{d}_{j:04d}','domain':d.capitalize(),'prompt':r['prompt'],'source_domain':d,'source_id':r.get('id','')})
        meta[d]={'raw_valid_n':len(raw),'train_n':len(train),'heldout_n':len(hold),'used_n':len(use),'malformed_raw_n':len(bad)}
        meta['malformed_raw_rows_excluded'][d]=bad
    out=Path(a.output); atomic_write_jsonl(out,rows)
    meta['total']=len(rows);meta['sha256']=hashlib.sha256(out.read_bytes()).hexdigest();meta['note']='Malformed records, if any, are physically invalid formal_raw lines and are excluded only from KD holdout candidates; frozen formal expert-training rows are parsed strictly.'
    tmp=out.with_suffix(out.suffix+'.meta.json.tmp');tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(out.with_suffix(out.suffix+'.meta.json'))
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
