#!/usr/bin/env python3
"""Generate reproducible mergekit YAMLs for TIES/DARE-TIES planned runs.

A checkpoint map YAML resolves symbolic sibling names to materialized model
paths/IDs. Example:
  base_model: Qwen/Qwen3-4B-Base
  checkpoints:
    math: runs/experts/math/full
    code: runs/experts/code/full
  density: 0.5
  dtype: bfloat16

The generated YAML follows mergekit's standard models/base_model schema. Always
record the installed mergekit version in run metadata before producing results.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import yaml


def parse_list_json(x):
    try:return list(json.loads(x)) if str(x).strip() else []
    except Exception:return []


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--map',required=True);ap.add_argument('--out-dir',default='configs/mergekit/generated');args=ap.parse_args()
    mp=yaml.safe_load(Path(args.map).read_text(encoding='utf-8')); ck=mp.get('checkpoints',{}); base=mp.get('base_model'); density=float(mp.get('density',0.5)); dtype=mp.get('dtype','bfloat16')
    df=pd.read_csv(args.manifest,keep_default_na=False); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);n=0
    for _,r in df.iterrows():
        method=str(r['construction'])
        if method not in {'ties','dare_ties'}: continue
        parents=[x for x in str(r['parents']).split(';') if x]
        weights=parse_list_json(r['weights'])
        if len(weights)!=len(parents): raise SystemExit(f"weight mismatch for {r['run_id']}")
        missing=[x for x in parents if x not in ck]
        if missing: raise SystemExit(f"checkpoint map missing {missing} for {r['run_id']}")
        cfg={'merge_method':method,'base_model':base,'models':[],'parameters':{'normalize':True},'tokenizer_source':'base','dtype':dtype}
        for name,w in zip(parents,weights):
            pars={'weight':float(w),'density':density}
            cfg['models'].append({'model':str(ck[name]),'parameters':pars})
        fn=out/f"{r['run_id']}.yaml";fn.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8');n+=1
    print(f'wrote {n} mergekit configs -> {out}')

if __name__=='__main__':main()
