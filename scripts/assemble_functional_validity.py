#!/usr/bin/env python3
"""Join domain-wise exact Shapley with FAS/construction/baseline coordinates.

Use repeated ``--fas DOMAIN=JSON`` arguments so every functional domain is
compared against coordinates extracted from probes selected within that same
domain.  This enforces the paper's definition pi^F(D), rather than comparing a
global coordinate vector to domain-specific counterfactual contributions.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read_coord(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    names=obj.get('parent_names') or obj.get('ancestor_names') or obj.get('names')
    pi=obj.get('reported_pi') or obj.get('coordinates') or obj.get('pi')
    if names is None or pi is None: raise ValueError(f'{path}: missing names/coordinates')
    return {str(n):float(v) for n,v in zip(names,pi)}

def parse_map(items):
    out={}
    for x in items:
        if '=' not in x: raise ValueError(f'expected DOMAIN=PATH, got {x}')
        d,p=x.split('=',1);out[d]=read_coord(p)
    return out

def parse_weights(s):
    if not s:return {}
    p=Path(s)
    if p.exists(): return {str(k):float(v) for k,v in json.loads(p.read_text()).items()}
    return {str(k):float(v) for k,v in json.loads(s).items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shapley',default='results/derived/shapley.csv');ap.add_argument('--case-id',default='adapter_merge_triplet');ap.add_argument('--fas',action='append',required=True,help='DOMAIN=coordinate.json');ap.add_argument('--construction-weights',default='{}');ap.add_argument('--absolute',action='append',default=[],help='optional DOMAIN=coordinate.json');ap.add_argument('--output',default='results/raw/functional_vectors.csv');args=ap.parse_args()
    sh=list(csv.DictReader(open(args.shapley,encoding='utf-8',newline=''))); fas=parse_map(args.fas); absd=parse_map(args.absolute) if args.absolute else {}; cons=parse_weights(args.construction_weights);rows=[]
    for r in sh:
        dom=r['domain'];p=r['parent'];sv=float(r['shapley'])
        if dom=='ALL' or dom not in fas: continue
        for q,mp in [('FAS coordinates',fas[dom]),('Merge weights',cons),('Absolute-output decomposition',absd.get(dom,{}))]:
            if p in mp: rows.append({'case_id':args.case_id,'domain':dom,'quantity':q,'parent':p,'value':mp[p],'shapley':sv})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','domain','quantity','parent','value','shapley']);w.writeheader();w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {out}')
if __name__=='__main__':main()
