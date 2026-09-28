#!/usr/bin/env python3
"""Build a baseline target manifest from the canonical experiment run manifest.

Only completed/ready runs with a concrete checkpoint/API path are emitted.
modelDNA applicability is determined from the published access assumptions:
open-weight same-shape merge/post-processing settings can be attempted; KD and
cross-scale students are marked N/A by construction.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--output',default='data/metadata/baseline_target_manifest.csv');ap.add_argument('--include-planned',action='store_true');args=ap.parse_args()
    rows=list(csv.DictReader(open(args.manifest,encoding='utf-8',newline='')));out=[]
    for r in rows:
        ck=(r.get('checkpoint_or_api') or '').strip()
        if not ck and not args.include_planned: continue
        sc=r['scenario']
        same_scale=r.get('student','') in ('','same')
        mdna = sc.startswith(('L1-','L2-','L3-','L4-')) and same_scale
        if sc.startswith(('L1-','L2-')):
            setting='clean'
        elif sc.startswith(('L3-','L4-')):
            setting='sft'
        elif sc.startswith(('L5-','L6-')):
            setting='kd'
        elif sc.startswith('L7-'):
            setting='deep'
        else:
            setting='open'
        out.append({
            'target_id':r['run_id'],
            'checkpoint':ck,
            'setting':setting,
            'scenario':sc,
            'modeldna_applicable':int(mdna),
            'status':r.get('status',''),
            'reason_if_na':'' if mdna else 'distillation/cross-scale or unsupported lineage setting',
        })
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    fields=['target_id','checkpoint','setting','scenario','modeldna_applicable','status','reason_if_na']
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    print(f'{len(out)} baseline targets -> {p}')
if __name__=='__main__':main()
