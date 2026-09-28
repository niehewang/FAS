#!/usr/bin/env python3
"""Collect selective FAS JSON outputs into the row-level paper metric schema.

This removes the final manual transcription step between `decompose_target.py`
and `evaluate_decompositions.py`.  A truth CSV supplies experiment semantics;
the JSON files supply predictions and selective states.

Truth CSV columns:
  target_id, method, setting, parent_names, true_support
Optional:
  true_pi, seed, queries, run_id, bank_complete, signal_bearing

JSON inputs are discovered recursively from --input-dir and matched by either
`target_id` inside the JSON or by filename stem before `.decomposition.json`.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def enc_list(x):
    if x is None: return ""
    if isinstance(x, str): return x
    return json.dumps(list(x), ensure_ascii=False)


def load_jsons(root: Path):
    out={}
    for p in sorted(root.rglob('*.json')):
        try: obj=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        if not isinstance(obj,dict): continue
        tid=str(obj.get('target_id') or p.name.replace('.decomposition.json','').replace('.json',''))
        if 'state' in obj or 'reported_pi' in obj or 'pi' in obj:
            if tid in out: raise ValueError(f'duplicate target_id={tid}: {out[tid][0]} and {p}')
            out[tid]=(p,obj)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--truth', required=True)
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output', default='results/raw/decomposition_predictions.csv')
    ap.add_argument('--strict', action='store_true')
    args=ap.parse_args()
    with open(args.truth,encoding='utf-8',newline='') as f: truth=list(csv.DictReader(f))
    preds=load_jsons(Path(args.input_dir)); rows=[]; missing=[]
    for t in truth:
        tid=t['target_id']
        if tid not in preds:
            missing.append(tid); continue
        path,obj=preds[tid]
        pi=obj.get('reported_pi', None)
        support=obj.get('reported_support', None)
        if pi is None and str(obj.get('state','')).lower()=='decomposable': pi=obj.get('pi')
        if support is None and str(obj.get('state','')).lower()=='decomposable': support=obj.get('support_names',obj.get('support'))
        names=obj.get('parent_names') or obj.get('ancestor_names') or t.get('parent_names','')
        row=dict(t)
        row.update({
            'pred_pi': enc_list(pi),
            'pred_support': enc_list(support),
            'parent_names': enc_list(names),
            'state': obj.get('state',''),
            'rho': obj.get('rho',''),
            'p_open': obj.get('open_p',obj.get('p_open','')),
            'p_signal': obj.get('signal_p',obj.get('p_signal','')),
            'calibration_regime': obj.get('calibration_regime',''),
            'finite_sample_guarantee': obj.get('finite_sample_guarantee',''),
            'source_json': str(path),
        })
        rows.append(row)
    if args.strict and missing: raise SystemExit(f'missing {len(missing)} targets: {missing[:20]}')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f'collected {len(rows)}/{len(truth)} targets -> {out}')
    if missing: print(f'WARNING missing={len(missing)}')

if __name__=='__main__': main()
