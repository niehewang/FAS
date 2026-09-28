#!/usr/bin/env python3
"""Materialize a JSONL containing only selected global probe indices.

Keeps `global_probe_index` so an online audit can query only B probes while the
ancestry dictionary is still built from the global bank-estimate pool.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probes',required=True);ap.add_argument('--selected',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
    obj=json.loads(Path(args.selected).read_text(encoding='utf-8')); inds=[int(x) for x in obj['selected']]
    if len(set(inds))!=len(inds): raise SystemExit('selected indices contain duplicates')
    if any(i<0 or i>=len(rows) for i in inds): raise SystemExit('selected index outside probe pool')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for local,i in enumerate(inds):
            r=dict(rows[i]);r['global_probe_index']=i;r['selected_order']=local
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'{len(inds)} probes -> {out}')
if __name__=='__main__':main()
