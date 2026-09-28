#!/usr/bin/env python3
"""Evaluate online repeat-sampling budget while keeping the ancestor bank fixed.

The ancestor dictionary and cached common-base responses are treated as offline
high-precision objects. Only target base/edited generations are subsampled to m,
so reported online calls are 2*B*m, matching the paper protocol.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.response import probe_blocks,global_ancestor_norms,stack_selected
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry
from fas_core.decompose import fas_decompose
from fas_core.metrics import l1_coordinate_error


def parse_list(s):
    s=(s or '').strip();
    if not s:return []
    if s.startswith('['):return list(json.loads(s))
    return [x for x in s.split(';') if x]

def response_full(path):
    z=np.load(path);b=np.asarray(z['base_embeddings'],float);e=np.asarray(z['edited_embeddings'],float)
    return e-b if b.ndim==2 else e.mean(1)-b.mean(1)
def target_response_subset(path,idx):
    z=np.load(path);b=np.asarray(z['base_embeddings'],float);e=np.asarray(z['edited_embeddings'],float)
    if b.ndim!=3: raise ValueError('sampling-budget benchmark requires target arrays [N,M,d]')
    return e[:,idx,:].mean(1)-b[:,idx,:].mean(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',required=True);ap.add_argument('--base',required=True);ap.add_argument('--targets',required=True);ap.add_argument('--budget',type=int,default=64);ap.add_argument('--m-values',default='1,2,4,8,16');ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output',default='results/templates/sampling_budget.csv');args=ap.parse_args()
    bank=np.load(args.bank);fields=np.asarray(bank['task_fields'],float);blocks,norms=probe_blocks(fields);base_r=response_full(args.base);active,_=greedy_logdet_select(blocks,args.budget);rng=np.random.default_rng(args.seed); targets=list(csv.DictReader(open(args.targets,encoding='utf-8',newline='')));mvals=[int(x) for x in args.m_values.split(',')]
    rows=[]
    for strategy in ['Random','Active']:
        selected=active if strategy=='Active' else rng.choice(len(blocks),size=args.budget,replace=False).tolist();A=stack_selected(fields,selected,norms);geom=dictionary_geometry(A)
        for m in mvals:
            errs=[]
            for tr in targets:
                z=np.load(tr['target_npz']);M=np.asarray(z['base_embeddings']).shape[1] if np.asarray(z['base_embeddings']).ndim==3 else 1
                if m>M:continue
                tp=np.asarray([float(x) for x in parse_list(tr.get('true_pi',''))],float) if tr.get('true_pi','').strip() else None
                if tp is None:continue
                for _ in range(args.repeats):
                    idx=rng.choice(M,size=m,replace=False);rt=target_response_subset(tr['target_npz'],idx);yall=rt-base_r;y=np.concatenate([yall[i] for i in selected]);dec=fas_decompose(A,y);errs.append(l1_coordinate_error(dec['pi'],tp))
            rows.append({'m':m,'strategy':strategy,'sigma_min_raw':geom['sigma_min_raw'],'l1_error':float(np.mean(errs)) if errs else '','online_calls':2*args.budget*m})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['m','strategy','sigma_min_raw','l1_error','online_calls']);w.writeheader();w.writerows(rows)
    print(f'Wrote {out}')
if __name__=='__main__':main()
