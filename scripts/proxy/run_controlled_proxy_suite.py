#!/usr/bin/env python3
from pathlib import Path
import sys,random,copy,csv,itertools,json
import numpy as np, torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/proxy'));sys.path.insert(0,str(ROOT/'code'))
from run_tiny_neural_proxy import LM,domain_lines,make_examples,train,probs,probe_pool,task_fields,norm_and_blocks,stack,merge
from fas_core.active_probe import greedy_logdet_select_partitioned
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
DOMS=['math','code','medical','science']

def quantize8(model):
    m=copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            mx=p.abs().max()
            if mx>0:
                s=mx/127.; p.copy_(torch.clamp(torch.round(p/s),-127,127)*s)
    return m

def utility(model,domain,seed):
    rng=random.Random(seed+hash(domain)%1000)
    X,Y=make_examples(domain_lines(domain,100,rng))
    model.eval()
    with torch.no_grad():
        total=0.; n=0
        for i in range(0,len(X),128):
            logits=model(X[i:i+128]); total += nn.functional.cross_entropy(logits,Y[i:i+128],reduction='sum').item(); n+=len(X[i:i+128])
    return -total/n

def shapley3(values):
    # values dict keyed tuple sorted subset, n=3
    import math
    out=np.zeros(3)
    N={0,1,2}
    for i in range(3):
        for r in range(3):
            for S in itertools.combinations(sorted(N-{i}),r):
                S=tuple(S); T=tuple(sorted(S+(i,)))
                coef=math.factorial(r)*math.factorial(2-r)/math.factorial(3)
                out[i]+=coef*(values[T]-values[S])
    return out

def fpr95(y_true,scores):
    fpr,tpr,_=roc_curve(y_true,scores)
    idx=np.where(tpr>=.95)[0]
    return float(fpr[idx[0]]) if len(idx) else 1.0

def main():
    out=ROOT/'results/proxy';out.mkdir(parents=True,exist_ok=True)
    open_rows=[]; valid_rows=[]
    for seed in [11,23,47]:
        rng=random.Random(seed);torch.manual_seed(seed);rngnp=np.random.default_rng(seed)
        base=LM();X,Y=make_examples(sum([domain_lines(d,80,rng) for d in DOMS],[]));train(base,X,Y,80,seed,3e-3)
        experts=[]
        for j,d in enumerate(DOMS):
            m=copy.deepcopy(base);xx,yy=make_examples(domain_lines(d,120,rng));train(m,xx,yy,50,seed*10+j,4e-3);experts.append(m)
        pairs=probe_pool();groups=[DOMS[i%4] for i in range(len(pairs))];F=task_fields(base,experts,pairs);norms,blocks=norm_and_blocks(F)
        q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q)
        # balanced selection for full 4-parent and known 3-parent banks
        ids4,_=greedy_logdet_select_partitioned(blocks,32,groups)
        F3=F[:,:3,:]; norms3,blocks3=norm_and_blocks(F3); ids3,_=greedy_logdet_select_partitioned(blocks3,32,groups)
        A3=stack(F3,ids3,norms3)
        def centered(model): return (probs(model,qe)-probs(model,q))-br
        def residual(model):
            df=centered(model); y=np.concatenate([df[p] for p in ids3]); return fas_decompose(A3,y,support_threshold=.05)['relative_residual']
        # conformal calibration from bank-complete known blends
        cal=[]
        for c in range(40):
            w=rngnp.dirichlet(np.ones(3)); cal.append(residual(merge(base,experts,[*w,0.])))
        # known, unknown-A, unknown-B (8-bit quantized known blends)
        for typ in ['Known','UnknownA','UnknownB']:
            for j in range(15):
                if typ=='UnknownA':
                    ws=rngnp.dirichlet(np.ones(3)); sci=float(rngnp.uniform(.3,.55)); w=[(1-sci)*ws[0],(1-sci)*ws[1],(1-sci)*ws[2],sci]; m=merge(base,experts,w)
                else:
                    w3=rngnp.dirichlet(np.ones(3)); m=merge(base,experts,[*w3,0.]); m=quantize8(m) if typ=='UnknownB' else m
                r=residual(m); p=(1+sum(x>=r for x in cal))/(len(cal)+1)
                open_rows.append(dict(seed=seed,type=typ,residual=r,p_open=p,declare_incomplete=int(p<.05)))
        # exact Shapley for first 3 parents, fixed coefficients
        wfull=np.array([.4,.35,.25])
        subset_models={}
        for bits in itertools.product([0,1],repeat=3):
            subset=tuple(i for i,b in enumerate(bits) if b); w=[wfull[i] if bits[i] else 0. for i in range(3)]+[0.]; subset_models[subset]=merge(base,experts,w)
        full=subset_models[(0,1,2)]
        full_df=centered(full)
        for dom in DOMS[:3]:
            vals={S:utility(m,dom,seed+700) for S,m in subset_models.items()}; shap=shapley3(vals)
            # FAS coordinate from all probes in this domain, first 3 ancestors
            ids=[i for i,g in enumerate(groups) if g==dom]; Fd=F3[ids]; nd=np.sqrt(np.sum(Fd**2,axis=(0,2))); A=np.concatenate([Fd[j].T/np.maximum(nd,1e-12)[None,:] for j in range(len(ids))],axis=0); y=np.concatenate([full_df[i] for i in ids]); dec=fas_decompose(A,y,support_threshold=.05); pi=dec['pi']
            sr_f=float(spearmanr(pi,shap).statistic) if np.std(shap)>1e-12 else np.nan; sr_w=float(spearmanr(wfull,shap).statistic) if np.std(shap)>1e-12 else np.nan
            valid_rows.append(dict(seed=seed,domain=dom,spearman_fas=sr_f,spearman_weight=sr_w,shapley=json.dumps(shap.tolist()),fas_pi=json.dumps(pi.tolist()),construction=json.dumps(wfull.tolist())))
    # outputs
    with (out/'proxy_open_set_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=open_rows[0].keys());w.writeheader();w.writerows(open_rows)
    with (out/'proxy_functional_validity_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=valid_rows[0].keys());w.writeheader();w.writerows(valid_rows)
    # open set aggregate
    y=np.array([1 if r['type']=='UnknownA' else 0 for r in open_rows]); sc=np.array([r['residual'] for r in open_rows]); auc=roc_auc_score(y,sc); fp=fpr95(y,sc)
    stats={}
    for typ in ['Known','UnknownA','UnknownB']:
        rr=[r for r in open_rows if r['type']==typ]; stats[typ]={'mean_residual':float(np.mean([r['residual'] for r in rr])),'incomplete_rate':float(np.mean([r['declare_incomplete'] for r in rr]))}
    fas=np.array([r['spearman_fas'] for r in valid_rows],float); wt=np.array([r['spearman_weight'] for r in valid_rows],float)
    summary={'open_set_auc':float(auc),'open_set_fpr95':float(fp),'open_set':stats,'functional_validity':{'fas_spearman_mean':float(np.nanmean(fas)),'construction_spearman_mean':float(np.nanmean(wt)),'n_seed_domain':len(valid_rows)}}
    (out/'proxy_suite_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
