#!/usr/bin/env python3
"""Assign evidence tiers to the deterministic run manifest.

The full design space remains intact, but expensive execution defaults to the
minimal evidence set.  Matching is deterministic and auditable.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import yaml

def _split(s): return [x for x in str(s or '').split(';') if x]

def matches(row, rule):
    if rule.get('scenario') and row['scenario'] != rule['scenario']: return False
    if rule.get('variants') and row['variant'] not in set(map(str,rule['variants'])): return False
    if rule.get('seeds') and int(row['seed']) not in set(map(int,rule['seeds'])): return False
    if rule.get('students') and row['student'] not in set(map(str,rule['students'])): return False
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--policy',default='configs/evidence_tiers.yaml');ap.add_argument('--output',default='data/metadata/experiment_run_manifest_tiered.csv');ap.add_argument('--out-dir',default='data/metadata/evidence_tiers');args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')))
    cfg=yaml.safe_load(Path(args.policy).read_text(encoding='utf-8'))
    default=cfg['policy'].get('default_tier','conditional')
    for r in rows:
        assigned=None
        for rule in cfg.get('rules',[]):
            if matches(r,rule):
                assigned=rule;break
        r['evidence_tier']=(assigned or {}).get('tier',default)
        r['trigger']=(assigned or {}).get('trigger','')
        r['claims']=';'.join((assigned or {}).get('claims',[]))
    fields=list(rows[0].keys())
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    od=Path(args.out_dir);od.mkdir(parents=True,exist_ok=True)
    for tier in ['core','conditional','optional']:
        sub=[r for r in rows if r['evidence_tier']==tier]
        p=od/f'{tier}.csv'
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sub)
    from collections import Counter
    c=Counter(r['evidence_tier'] for r in rows)
    print(' '.join(f'{k}={c[k]}' for k in sorted(c)))
    print(f'wrote {out}')
if __name__=='__main__':main()
