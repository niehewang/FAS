#!/usr/bin/env python3
"""End-to-end Random-vs-Active probe benchmark from precomputed response NPZs.

This is the recommended first "go/no-go" experiment. It consumes:
  * response_bank.npz with task_fields [N,K,d] and ancestor_names;
  * base response NPZ;
  * a target manifest CSV with either target_npz or target_centered_npz, plus true_support,true_pi;
and directly writes paper-shaped active_probe_curve.csv plus row-level
geometry_runs.csv.

Target-manifest list columns accept semicolon or JSON arrays. `true_pi` aligns
with response_bank ancestor_names.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'code'))
from fas_core.response import probe_blocks,global_ancestor_norms,stack_selected
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry,information_matrix
from fas_core.decompose import fas_decompose
from fas_core.metrics import support_metrics,l1_coordinate_error


def parse_list(s):
    s=(s or '').strip()
    if not s:return []
    if s.startswith('['):return list(json.loads(s))
    return [x for x in s.split(';') if x]

def response(path):
    z=np.load(path); b=np.asarray(z['base_embeddings'],float); e=np.asarray(z['edited_embeddings'],float)
    return e-b if b.ndim==2 else e.mean(axis=1)-b.mean(axis=1)

def y_for(target_r,base_r,selected): return np.concatenate([(target_r-base_r)[i] for i in selected],axis=0)
def centered_target(row, base_r):
    p=row.get('target_centered_npz','').strip()
    if p:
        z=np.load(ROOT/p if not Path(p).is_absolute() else p,allow_pickle=False)
        y=np.asarray(z['centered_response'],float)
        if y.shape!=base_r.shape: raise ValueError(f'centered target shape {y.shape} != base response shape {base_r.shape}')
        return y
    p=row.get('target_npz','').strip()
    if not p: raise ValueError('target row requires target_npz or target_centered_npz')
    pp=ROOT/p if not Path(p).is_absolute() else Path(p)
    return response(pp)-base_r

def y_selected(centered,selected): return np.concatenate([centered[i] for i in selected],axis=0)
def logdet_info(blocks,selected,ridge=1e-6):
    H=information_matrix(blocks,selected,ridge); sign,v=np.linalg.slogdet(H); return float(v) if sign>0 else float('-inf')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',required=True,help='evaluation/estimation response bank')
    ap.add_argument('--selection-bank',help='optional independent bank used only for active selection; defaults to --bank')
    ap.add_argument('--base',required=True); ap.add_argument('--targets',required=True)
    ap.add_argument('--budgets',default='8,16,32,64,128'); ap.add_argument('--random-repeats',type=int,default=10); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--support-threshold',type=float,default=0.01); ap.add_argument('--output-curve',default='results/raw/active_probe_curve.csv'); ap.add_argument('--output-geometry',default='results/raw/geometry_runs.csv'); args=ap.parse_args()
    bank=np.load(args.bank); fields=np.asarray(bank['task_fields'],float); names=[str(x) for x in bank['ancestor_names']]
    sel_bank=np.load(args.selection_bank or args.bank); sel_fields=np.asarray(sel_bank['task_fields'],float); sel_names=[str(x) for x in sel_bank['ancestor_names']]
    if sel_fields.shape!=fields.shape or sel_names!=names:
        raise ValueError('selection/evaluation banks must have identical probe/ancestor layout')
    selection_blocks,_=probe_blocks(sel_fields); evaluation_blocks,norms=probe_blocks(fields); base_r=response(args.base)
    targets=list(csv.DictReader(open(args.targets,encoding='utf-8',newline=''))); budgets=[int(x) for x in args.budgets.split(',')]
    rng=np.random.default_rng(args.seed); detailed=[]
    active_cache={}
    for B in budgets:
        if B>len(selection_blocks):continue
        sel,gains=greedy_logdet_select(selection_blocks,B); active_cache[B]=(sel,gains)
        for tr in targets:
            centered=centered_target(tr,base_r); true_support=set(parse_list(tr.get('true_support',''))); tp=np.asarray([float(x) for x in parse_list(tr.get('true_pi',''))],float) if tr.get('true_pi','').strip() else None
            for strategy in ['Active','Random']:
                reps=1 if strategy=='Active' else args.random_repeats
                for rep in range(reps):
                    selected=sel if strategy=='Active' else rng.choice(len(evaluation_blocks),size=B,replace=False).tolist()
                    A=stack_selected(fields,selected,norms); y=y_selected(centered,selected); dec=fas_decompose(A,y,support_threshold=args.support_threshold); geom=dictionary_geometry(A)
                    sm=support_metrics(true_support,{names[i] for i in dec['support']}) if true_support else {'f1':np.nan,'exact_support':np.nan}
                    l1=l1_coordinate_error(dec['pi'],tp) if tp is not None and len(tp)==len(names) else np.nan
                    tid=tr.get('target_id','').strip() or Path((tr.get('target_npz') or tr.get('target_centered_npz'))).stem
                    detailed.append({'target_id':tid,'strategy':strategy,'budget':B,'repeat':rep,'sigma_min_raw':geom['sigma_min_raw'],'sigma_min_unit':geom['sigma_min_unit'],'coherence':geom['coherence'],'condition_number_raw':geom['condition_number_raw'],'logdet':logdet_info(evaluation_blocks,selected),'coordinate_l1':l1,'parent_f1':sm['f1'],'exact_support':sm['exact_support'],'rho':dec['rho']})
    gp=Path(args.output_geometry);gp.parent.mkdir(parents=True,exist_ok=True)
    with gp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detailed[0].keys()));w.writeheader();w.writerows(detailed)
    agg=[]
    for strategy in ['Random','Active']:
        for B in budgets:
            rr=[r for r in detailed if r['strategy']==strategy and r['budget']==B]
            if not rr:continue
            def mn(k):
                v=np.asarray([float(r[k]) for r in rr if np.isfinite(float(r[k]))]);return float(v.mean()) if len(v) else ''
            agg.append({'method':strategy,'budget':B,'sigma_min_raw':mn('sigma_min_raw'),'sigma_min_unit':mn('sigma_min_unit'),'coherence':mn('coherence'),'l1_error':mn('coordinate_l1')})
    cp=Path(args.output_curve);cp.parent.mkdir(parents=True,exist_ok=True)
    with cp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','budget','sigma_min_raw','sigma_min_unit','coherence','l1_error']);w.writeheader();w.writerows(agg)
    print(f'Wrote {cp} and {gp} ({len(detailed)} row-level evaluations)')
if __name__=='__main__':main()
