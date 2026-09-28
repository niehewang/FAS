#!/usr/bin/env python3
from pathlib import Path
import sys, random, copy, csv, json, itertools, math, time
import numpy as np, torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/proxy')); sys.path.insert(0,str(ROOT/'code'))
from run_tiny_transformer_proxy import TinyTransformerLM, domain_lines, make_examples, train, probs, probe_pool, task_fields, norm_blocks, stack, merge, DOMS
from fas_core.active_probe import greedy_logdet_select_partitioned
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)

def quantize8(model):
    m=copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            mx=p.abs().max()
            if mx>0:
                s=mx/127.; p.copy_(torch.clamp(torch.round(p/s),-127,127)*s)
    return m

def utility(model,domain,seed):
    rng=random.Random(seed+{'math':1,'code':2,'medical':3,'science':4}[domain])
    X,Y=make_examples(domain_lines(domain,80,rng))
    model.eval()
    with torch.no_grad():
        total=0.;n=0
        for i in range(0,len(X),96):
            logits=model(X[i:i+96]);total+=nn.functional.cross_entropy(logits,Y[i:i+96],reduction='sum').item();n+=len(X[i:i+96])
    return -total/n

def shapley3(vals):
    out=np.zeros(3);N={0,1,2}
    for i in range(3):
        for r in range(3):
            for S in itertools.combinations(sorted(N-{i}),r):
                S=tuple(S);T=tuple(sorted(S+(i,)));coef=math.factorial(r)*math.factorial(2-r)/math.factorial(3);out[i]+=coef*(vals[T]-vals[S])
    return out

def fpr95(y,s):
    fpr,tpr,_=roc_curve(y,s);idx=np.where(tpr>=.95)[0];return float(fpr[idx[0]]) if len(idx) else 1.0

def main():
    t0=time.time();out=ROOT/'results/proxy';out.mkdir(parents=True,exist_ok=True);open_rows=[];valid_rows=[]
    for seed in [11,23,47]:
        rng=random.Random(seed);torch.manual_seed(seed);rngnp=np.random.default_rng(seed)
        base=TinyTransformerLM();X,Y=make_examples(sum([domain_lines(d,55,rng) for d in DOMS],[]));train(base,X,Y,45,seed,2e-3)
        experts=[]
        for j,d in enumerate(DOMS):
            m=copy.deepcopy(base);xx,yy=make_examples(domain_lines(d,80,rng));train(m,xx,yy,28,seed*10+j,2.5e-3);experts.append(m)
        pairs,groups=probe_pool();F=task_fields(base,experts,pairs);norms,blocks=norm_blocks(F)
        q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q)
        F3=F[:,:3,:];n3,b3=norm_blocks(F3);ids3,_=greedy_logdet_select_partitioned(b3,16,groups);A3=stack(F3,ids3,n3)
        def centered(m):return (probs(m,qe)-probs(m,q))-br
        def residual(m):
            df=centered(m);y=np.concatenate([df[p] for p in ids3]);return fas_decompose(A3,y,support_threshold=.05)['relative_residual']
        cal=[]
        for c in range(30):
            w=rngnp.dirichlet(np.ones(3));cal.append(residual(merge(base,experts,[*w,0.])))
        for typ in ['Known','UnknownA','UnknownB']:
            for j in range(12):
                if typ=='UnknownA':
                    ws=rngnp.dirichlet(np.ones(3));sci=float(rngnp.uniform(.3,.55));w=[(1-sci)*ws[0],(1-sci)*ws[1],(1-sci)*ws[2],sci];m=merge(base,experts,w)
                else:
                    w3=rngnp.dirichlet(np.ones(3));m=merge(base,experts,[*w3,0.]);m=quantize8(m) if typ=='UnknownB' else m
                r=residual(m);p=(1+sum(x>=r for x in cal))/(len(cal)+1);open_rows.append(dict(seed=seed,type=typ,residual=r,p_open=p,declare_incomplete=int(p<.05)))
        wfull=np.array([.4,.35,.25]);subs={}
        for bits in itertools.product([0,1],repeat=3):
            S=tuple(i for i,b in enumerate(bits) if b);w=[wfull[i] if bits[i] else 0. for i in range(3)]+[0.];subs[S]=merge(base,experts,w)
        full=subs[(0,1,2)];fdf=centered(full)
        for dom in DOMS[:3]:
            vals={S:utility(m,dom,seed+700) for S,m in subs.items()};sh=shapley3(vals);ids=[i for i,g in enumerate(groups) if g==dom];Fd=F3[ids];nd=np.sqrt(np.sum(Fd**2,axis=(0,2)));A=np.concatenate([Fd[j].T/np.maximum(nd,1e-12)[None,:] for j in range(len(ids))],0);y=np.concatenate([fdf[i] for i in ids]);pi=fas_decompose(A,y,support_threshold=.05)['pi'];sf=float(spearmanr(pi,sh).statistic) if np.std(sh)>1e-12 else np.nan;sw=float(spearmanr(wfull,sh).statistic) if np.std(sh)>1e-12 else np.nan;valid_rows.append(dict(seed=seed,domain=dom,spearman_fas=sf,spearman_weight=sw,shapley=json.dumps(sh.tolist()),fas_pi=json.dumps(pi.tolist()),construction=json.dumps(wfull.tolist())))
    for name,data in [('transformer_proxy_open_set_rows.csv',open_rows),('transformer_proxy_functional_validity_rows.csv',valid_rows)]:
        with (out/name).open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=data[0].keys());wr.writeheader();wr.writerows(data)
    y=np.array([1 if r['type']=='UnknownA' else 0 for r in open_rows]);sc=np.array([r['residual'] for r in open_rows]);stats={}
    for typ in ['Known','UnknownA','UnknownB']:
        rr=[r for r in open_rows if r['type']==typ];stats[typ]={'mean_residual':float(np.mean([r['residual'] for r in rr])),'incomplete_rate':float(np.mean([r['declare_incomplete'] for r in rr]))}
    fas=np.array([r['spearman_fas'] for r in valid_rows],float);wt=np.array([r['spearman_weight'] for r in valid_rows],float)
    summary={'seconds':time.time()-t0,'open_set_auc':float(roc_auc_score(y,sc)),'open_set_fpr95':fpr95(y,sc),'open_set':stats,'functional_validity':{'fas_spearman_mean':float(np.nanmean(fas)),'construction_spearman_mean':float(np.nanmean(wt)),'n_seed_domain':len(valid_rows)}}
    (out/'transformer_proxy_suite_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
