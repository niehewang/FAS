#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
DOMS=['Math','Code','Medical','Science']; EPS=1e-12

def avg(a):
    a=np.asarray(a,float);return a.mean(1) if a.ndim==3 else a

def delta(z): return avg(z['edited_embeddings'])-avg(z['base_embeddings'])
def gids(z): return [int(x) for x in np.asarray(z['global_probe_indices']).reshape(-1)] if 'global_probe_indices' in z else list(range(avg(z['base_embeddings']).shape[0]))
def solve(F,y,norms):
    A=np.concatenate([(F[i]/norms[:,None]).T for i in range(F.shape[0])],axis=0); yy=np.asarray(y,float).reshape(-1); c,_=nnls(A,yy);pi=c/(c.sum()+EPS);pred=A@c;res=float(np.linalg.norm(pred-yy)/(np.linalg.norm(yy)+EPS));return pi,res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage3',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--base-selected',required=True);ap.add_argument('--target-selected',required=True);ap.add_argument('--selection',action='append',required=True,help='DOMAIN=selected.json');ap.add_argument('--output',required=True);a=ap.parse_args();S=Path(a.stage3)
    zb=np.load(S/'shared/base_full.npz');db=delta(zb);bb=avg(zb['base_embeddings']);be=avg(zb['edited_embeddings']);gb=gids(zb);mb={g:i for i,g in enumerate(gb)}
    banks={d:np.load(S/f's{a.seed}/banks/{d.lower()}.npz') for d in DOMS};maps={d:{g:i for i,g in enumerate(gids(banks[d]))} for d in DOMS};de={d:delta(banks[d]) for d in DOMS};eb={d:avg(banks[d]['base_embeddings']) for d in DOMS}
    F=np.stack([np.stack([de[d][maps[d][g]]-db[mb[g]] for d in DOMS]) for g in gb]);N=np.maximum(np.sqrt(np.sum(F**2,axis=(0,2))),EPS)
    # Absolute-output baseline follows the formal LLM clean-merge baseline: unedited/base-query output displacement relative to the shared base.
    Fa=np.stack([np.stack([eb[d][maps[d][g]]-bb[mb[g]] for d in DOMS]) for g in gb]);Na=np.maximum(np.sqrt(np.sum(Fa**2,axis=(0,2))),EPS)
    zbs=np.load(a.base_selected);zts=np.load(a.target_selected);mbs={g:i for i,g in enumerate(gids(zbs))};mts={g:i for i,g in enumerate(gids(zts))};dselb=delta(zbs);dselt=delta(zts);bsel=avg(zbs['base_embeddings']);tsel=avg(zts['base_embeddings'])
    sels={}
    for x in a.selection:
        d,p=x.split('=',1);sels[d]=[int(i) for i in json.loads(Path(p).read_text())['selected']]
    out={'seed':a.seed,'ancestor_names':DOMS,'domains':{}}
    for d,gs in sels.items():
        if not all(g in mbs and g in mts and g in mb for g in gs):raise ValueError(f'{d}: missing selected gid')
        pf=[mb[g] for g in gs];pt=[mts[g] for g in gs];pb=[mbs[g] for g in gs]
        y=np.stack([dselt[i]-dselb[j] for i,j in zip(pt,pb)]);pi,res=solve(F[pf],y,N)
        ya=np.stack([tsel[i]-bsel[j] for i,j in zip(pt,pb)]);pia,resa=solve(Fa[pf],ya,Na)
        out['domains'][d]={'selected_global_probe_indices':gs,'fas_pi':pi.tolist(),'fas_relative_residual':res,'absolute_pi':pia.tolist(),'absolute_relative_residual':resa,'fas_false_candidate_code_mass':float(pi[1]),'absolute_false_candidate_code_mass':float(pia[1])}
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2),encoding='utf-8');print(p)
if __name__=='__main__':main()
