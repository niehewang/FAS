#!/usr/bin/env python3
from pathlib import Path
import sys, random, copy, csv, json
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/proxy')); sys.path.insert(0,str(ROOT/'code'))
from run_tiny_neural_proxy import LM,domain_lines,make_examples,train,probs,probe_pool,task_fields,norm_and_blocks,stack,merge
from run_tiny_balanced_proxy import balanced_select
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
DOMS=['math','code','medical','science']

def f1(pi,w,thr=.05):
    t=set(np.flatnonzero(np.asarray(w)>0)); p=set(np.flatnonzero(np.asarray(pi)>=thr)); tp=len(t&p)
    pr=tp/max(1,len(p)); rc=tp/max(1,len(t)); return 2*pr*rc/max(pr+rc,1e-12)

def main():
    out=ROOT/'results/proxy'; out.mkdir(parents=True,exist_ok=True); rows=[]
    targets=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]
    for seed in [11,23,47]:
        rng=random.Random(seed); torch.manual_seed(seed); rngnp=np.random.default_rng(seed)
        base=LM(); X,Y=make_examples(sum([domain_lines(d,80,rng) for d in DOMS],[])); train(base,X,Y,80,seed,3e-3)
        experts=[]
        for j,d in enumerate(DOMS):
            m=copy.deepcopy(base); xx,yy=make_examples(domain_lines(d,120,rng)); train(m,xx,yy,50,seed*10+j,4e-3); experts.append(m)
        pairs=probe_pool(); groups=[DOMS[i%4] for i in range(len(pairs))]
        F=task_fields(base,experts,pairs); norms,blocks=norm_and_blocks(F)
        selF=F+0.002*rngnp.normal(size=F.shape); _,selblocks=norm_and_blocks(selF)
        ids=balanced_select(selblocks,16,groups)
        q=[a for a,b in pairs]; qe=[b for a,b in pairs]
        bq=probs(base,q); bqe=probs(base,qe); br=bqe-bq
        # absolute-output dictionary at edited query only
        Fabs=np.stack([probs(e,qe)-bqe for e in experts],axis=1); nabs,babs=norm_and_blocks(Fabs); ids_abs=balanced_select(babs,16,groups); Aabs=stack(Fabs,ids_abs,nabs)
        A=stack(F,ids,norms); Afull=stack(F,range(len(pairs)),norms)
        # deterministic within-domain pair permutation
        perm=np.arange(len(pairs))
        for g in DOMS:
            gg=np.array([i for i,x in enumerate(groups) if x==g]); sh=gg.copy(); rngnp.shuffle(sh)
            if len(gg)>1 and np.any(gg==sh): sh=np.roll(sh,1)
            perm[gg]=sh
        e0r=probs(experts[0],qe)-probs(experts[0],q)
        for ti,w in enumerate(targets):
            d=merge(base,experts,w)
            dq=probs(d,q); dqe=probs(d,qe); df=(dqe-dq)-br
            oracle=fas_decompose(Afull,np.concatenate(df),support_threshold=.05)['pi']
            controls={}
            controls['Correct IFRF']=fas_decompose(A,np.concatenate([df[p] for p in ids]),support_threshold=.05)['pi']
            controls['Pair-shuffled target']=fas_decompose(A,np.concatenate([df[perm[p]] for p in ids]),support_threshold=.05)['pi']
            wrong=(dqe-dq)-e0r
            controls['Wrong anchor']=fas_decompose(A,np.concatenate([wrong[p] for p in ids]),support_threshold=.05)['pi']
            yabs=dqe-bqe
            controls['Absolute output']=fas_decompose(Aabs,np.concatenate([yabs[p] for p in ids_abs]),support_threshold=.05)['pi']
            # Random-domain metadata: balanced selection on shuffled domain labels, correct IFRF afterward.
            sg=groups.copy(); rngnp.shuffle(sg); ids_s=balanced_select(selblocks,16,sg); As=stack(F,ids_s,norms)
            controls['Shuffled domain labels']=fas_decompose(As,np.concatenate([df[p] for p in ids_s]),support_threshold=.05)['pi']
            for name,pi in controls.items():
                rows.append(dict(seed=seed,target=ti,control=name,l1_oracle=float(np.abs(pi-oracle).sum()),parent_f1=f1(pi,w)))
    p=out/'tiny_negative_control_rows.csv'
    with p.open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    agg=[]
    for name in sorted(set(r['control'] for r in rows)):
        rr=[r for r in rows if r['control']==name]
        agg.append(dict(control=name,l1_oracle=float(np.mean([r['l1_oracle'] for r in rr])),parent_f1=float(np.mean([r['parent_f1'] for r in rr])),n=len(rr)))
    qout=out/'tiny_negative_control_summary.csv'
    with qout.open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=agg[0].keys());wr.writeheader();wr.writerows(agg)
    print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
