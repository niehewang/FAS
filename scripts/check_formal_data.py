#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',default='data/training/formal');ap.add_argument('--expected',type=int,default=3000);ap.add_argument('--manifest',default='data/governance/formal_data_lock.json');args=ap.parse_args()
    domains=['math','code','medical','science'];out={'expected_per_domain':args.expected,'domains':{}}
    all_ids=set()
    for d in domains:
        p=ROOT/args.dir/f'{d}.jsonl'
        if not p.exists():raise SystemExit(f'missing {p}')
        rows=read(p)
        if len(rows)!=args.expected:raise SystemExit(f'{d}: n={len(rows)} expected={args.expected}')
        ids=[r['id'] for r in rows]
        if len(ids)!=len(set(ids)):raise SystemExit(f'{d}: duplicate ids')
        cross=set(ids)&all_ids
        if cross:raise SystemExit(f'{d}: {len(cross)} ids overlap earlier domain')
        all_ids.update(ids)
        out['domains'][d]={'n':len(rows),'sha256':sha(p),'source_datasets':sorted({r.get('source_dataset','') for r in rows}),'licenses':sorted({r.get('license','') for r in rows}),'source_revisions':sorted({r.get('source_revision','') for r in rows})}
    m=ROOT/args.manifest;m.parent.mkdir(parents=True,exist_ok=True);m.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FORMAL_DATA_OK',json.dumps({d:v['n'] for d,v in out['domains'].items()}))
if __name__=='__main__':main()
