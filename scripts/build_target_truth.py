#!/usr/bin/env python3
"""Build a row-level truth manifest from the deterministic experiment manifest.

The file is consumed by collect_decomposition_jsons.py. Real merge/KD runs get
structural ground truth only by default; no construction weights are silently
converted into FAS coordinate truth.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

LABEL={'math':'Math','code':'Code','medical':'Medical','science':'Science'}

def setting(scenario):
    if scenario.startswith(('L1-','L2-')): return 'clean'
    if scenario.startswith(('L3-','L4-')): return 'sft'
    if scenario.startswith(('L5-','L6-')): return 'kd'
    if scenario.startswith('L7-'): return 'deep'
    if scenario.startswith('L8-'): return 'open'
    return scenario

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--output',default='data/metadata/target_truth.csv');ap.add_argument('--method',default='FAS');ap.add_argument('--selected',default='runs/selected_probes.json');ap.add_argument('--samples',type=int,default=4);args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')));out=[]
    budget=''
    sp=Path(args.selected)
    if sp.exists():
        sel=json.loads(sp.read_text(encoding='utf-8')).get('selected',[])
        budget=str(2*len(sel)*args.samples)  # suspect calls: base+edited prompt for each stochastic replicate
    for r in rows:
        ps=[LABEL.get(x,x) for x in r['parents'].split(';') if x]
        # Open-set unknown-A is bank-incomplete; known support remains useful but no coordinate truth.
        out.append({'target_id':r['run_id'],'run_id':r['run_id'],'method':args.method,'setting':setting(r['scenario']),
                    'parent_names':json.dumps(list(LABEL.values()),ensure_ascii=False),
                    'true_support':json.dumps(ps,ensure_ascii=False),'true_pi':'','seed':r['seed'],'queries':budget,
                    'scenario':r['scenario'],'bank_complete':'0' if r.get('unseen_parents','').strip() else '1',
                    'signal_bearing':'1'})
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(out[0].keys()) if out else []
    with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    print(f'{len(out)} rows -> {p}')
if __name__=='__main__':main()
