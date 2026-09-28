#!/usr/bin/env python3
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry
from fas_core.decompose import fas_decompose

def make_fields(n=400,k=4,d=8,seed=0):
    rng=np.random.default_rng(seed)
    shared=rng.normal(size=(n,1,d))
    unique=rng.normal(size=(n,k,d))
    info=rng.lognormal(-0.4,0.9,size=(n,1,1))
    latent=.80*shared+info*unique
    return latent

def noisy(latent,seed,sd=.03):
    rng=np.random.default_rng(seed); return latent+sd*rng.normal(size=latent.shape)

def blocks(fields):
    norms=np.sqrt(np.sum(fields**2,axis=(0,2)))
    return [fields[p].T/np.maximum(norms,1e-12)[None,:] for p in range(fields.shape[0])], norms

def stack(fields,ids,norms):
    return np.concatenate([fields[p].T/norms[None,:] for p in ids],axis=0)

def main():
    out=ROOT/'results/proxy'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    budgets=[8,16,32,64,128]
    for seed in range(20):
        latent=make_fields(seed=seed)
        sel_fields=noisy(latent,1000+seed,.05); eval_fields=noisy(latent,2000+seed,.05)
        sel_blocks,_=blocks(sel_fields); _,norms=blocks(eval_fields)
        rng=np.random.default_rng(3000+seed)
        for B in budgets:
            act,_=greedy_logdet_select(sel_blocks,B)
            for strategy in ['Active','Random']:
                reps=1 if strategy=='Active' else 10
                for r in range(reps):
                    ids=act if strategy=='Active' else rng.choice(len(sel_blocks),B,replace=False).tolist()
                    A=stack(eval_fields,ids,norms)
                    geom=dictionary_geometry(A)
                    for t in range(20):
                        supp=rng.choice(4,size=3,replace=False)
                        w=rng.dirichlet(np.ones(3)); gamma=np.zeros(4); gamma[supp]=w
                        y=A@gamma+0.02*rng.normal(size=A.shape[0])
                        dec=fas_decompose(A,y,support_threshold=.05)
                        l1=float(np.abs(dec['pi']-gamma/gamma.sum()).sum())
                        pred=set(dec['support']); true=set(supp.tolist())
                        tp=len(pred&true); prec=tp/max(len(pred),1); rec=tp/len(true); f1=2*prec*rec/max(prec+rec,1e-12)
                        rows.append(dict(seed=seed,budget=B,strategy=strategy,repeat=r,target=t,
                            sigma_min=geom['sigma_min_raw'],coherence=geom['coherence'],l1=l1,f1=f1,rho=dec['rho']))
    p=out/'mechanism_proxy_rows.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    agg=[]
    for B in budgets:
        for s in ['Active','Random']:
            rr=[x for x in rows if x['budget']==B and x['strategy']==s]
            agg.append(dict(method=s,budget=B,
                sigma_min=np.mean([x['sigma_min'] for x in rr]),
                coherence=np.mean([x['coherence'] for x in rr]),
                l1_error=np.mean([x['l1'] for x in rr]),
                parent_f1=np.mean([x['f1'] for x in rr]),
                rho=np.mean([x['rho'] for x in rr])))
    q=out/'mechanism_proxy_summary.csv'
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
    print(q)
    for x in agg: print(x)
if __name__=='__main__': main()
