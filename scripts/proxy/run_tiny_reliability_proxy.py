#!/usr/bin/env python3
# reuse definitions from run_tiny_neural_proxy
from pathlib import Path
import sys,random,copy,csv
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'scripts/proxy')); sys.path.insert(0,str(ROOT/'code'))
from run_tiny_neural_proxy import LM,domain_lines,make_examples,train,probs,probe_pool,task_fields,norm_and_blocks,stack,merge
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)

def main():
 out=ROOT/'results/proxy'; rows=[]
 for seed in [11,23,47]:
  rng=random.Random(seed); torch.manual_seed(seed)
  base=LM(); X,Y=make_examples(sum([domain_lines(d,80,rng) for d in ['math','code','medical','science']],[])); train(base,X,Y,80,seed,3e-3)
  experts=[]
  for j,d in enumerate(['math','code','medical','science']):
   m=copy.deepcopy(base); xx,yy=make_examples(domain_lines(d,120,rng)); train(m,xx,yy,50,seed*10+j,4e-3);experts.append(m)
  pairs=probe_pool(); F=task_fields(base,experts,pairs); norms,blocks=norm_and_blocks(F)
  q=[a for a,b in pairs]; qe=[b for a,b in pairs]; bq=probs(base,q); bqe=probs(base,qe); br=bqe-bq
  # calibration composition residual per probe
  calw=[[.5,.5,0,0],[0,.5,.5,0],[0,0,.5,.5],[.5,0,0,.5],[.4,.3,.3,0]]
  resid=[]
  for w in calw:
   desc=merge(base,experts,w); df=(probs(desc,qe)-probs(desc,q))-br; pred=sum(float(w[i])*F[:,i,:] for i in range(4)); resid.append(df-pred)
  resid=np.stack(resid) # C,N,V
  res_rms=np.sqrt(np.mean(resid**2,axis=(0,2)) + 1e-10)
  field_rms=np.sqrt(np.mean(F**2,axis=(1,2)) + 1e-10)
  ratio=res_rms/field_rms
  rel=1.0/np.sqrt(1.0+4.0*ratio**2)
  rel=np.clip(rel,0.35,1.0)
  # independent selection noise, and reliability-weighted blocks
  rngnp=np.random.default_rng(seed); selF=F+0.002*rngnp.normal(size=F.shape); _,selblocks=norm_and_blocks(selF)
  relblocks=[selblocks[p]*rel[p] for p in range(len(selblocks))]
  targets=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]
  desc_fields=[]
  for w in targets:
   d=merge(base,experts,w); desc_fields.append((probs(d,qe)-probs(d,q))-br)
  for B in [8,16,32,64]:
   ids_active,_=greedy_logdet_select(selblocks,B); ids_rel,_=greedy_logdet_select(relblocks,B)
   for strat in ['Active','RelActive','Random']:
    reps=1 if strat!='Random' else 5
    for rep in range(reps):
     ids=ids_active if strat=='Active' else ids_rel if strat=='RelActive' else rngnp.choice(len(pairs),B,replace=False).tolist()
     A=stack(F,ids,norms); geom=dictionary_geometry(A)
     wvec=np.concatenate([np.full(F.shape[2],rel[p]) for p in ids]) if strat=='RelActive' else None
     for ti,w in enumerate(targets):
      y=np.concatenate([desc_fields[ti][p] for p in ids]); dec=fas_decompose(A,y,weights=wvec,support_threshold=.05)
      ref=np.array(w)*norms; ref=ref/ref.sum(); l1=float(np.abs(dec['pi']-ref).sum())
      true=set(np.flatnonzero(np.array(w)>0));pred=set(dec['support']);tp=len(true&pred);pr=tp/max(1,len(pred));rc=tp/max(1,len(true));f1=2*pr*rc/max(pr+rc,1e-12)
      rows.append(dict(seed=seed,budget=B,strategy=strat,repeat=rep,target=ti,sigma_min=geom['sigma_min_raw'],coherence=geom['coherence'],l1=l1,f1=f1,rho=dec['rho']))
 p=out/'tiny_reliability_proxy_rows.csv';
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 agg=[]
 for B in [8,16,32,64]:
  for s in ['Active','RelActive','Random']:
   rr=[x for x in rows if x['budget']==B and x['strategy']==s]
   agg.append(dict(method=s,budget=B,sigma_min=np.mean([x['sigma_min'] for x in rr]),coherence=np.mean([x['coherence'] for x in rr]),l1_error=np.mean([x['l1'] for x in rr]),parent_f1=np.mean([x['f1'] for x in rr]),rho=np.mean([x['rho'] for x in rr])))
 q=out/'tiny_reliability_proxy_summary.csv';
 with q.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
 print(q);[print(x) for x in agg]
if __name__=='__main__': main()
