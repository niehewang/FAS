#!/usr/bin/env python3
from pathlib import Path
import sys,random,copy,csv,json,time,argparse
import numpy as np,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/proxy'));sys.path.insert(0,str(ROOT/'code'))
from run_tiny_transformer_proxy import TinyTransformerLM,domain_lines,make_examples,train,probs,probe_pool,task_fields,norm_blocks,stack,merge,DOMS
from fas_core.active_probe import greedy_logdet_select_partitioned
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
SPECS=[('48k',48,2,96,4),('246k',96,3,192,4),('558k',128,4,256,8)]

def quant8(m):
 z=copy.deepcopy(m)
 with torch.no_grad():
  for p in z.parameters():
   mx=p.abs().max()
   if mx>0:
    s=mx/127.;p.copy_(torch.clamp(torch.round(p/s),-127,127)*s)
 return z

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--specs',default='48k,246k,558k');ap.add_argument('--seeds',default='11,23,47');args=ap.parse_args()
 want=set(args.specs.split(','));seeds=[int(x) for x in args.seeds.split(',') if x]
 out=ROOT/'results/proxy';out.mkdir(parents=True,exist_ok=True);rows=[];t0=time.time()
 for label,d,l,ff,h in SPECS:
  if label not in want: continue
  for seed in seeds:
   rng=random.Random(seed);torch.manual_seed(seed);rngnp=np.random.default_rng(seed)
   base=TinyTransformerLM(d=d,nhead=h,layers=l,ff=ff);nparam=sum(p.numel() for p in base.parameters())
   X,Y=make_examples(sum([domain_lines(dom,45,rng) for dom in DOMS],[]));train(base,X,Y,40,seed,2e-3)
   experts=[]
   for j,dom in enumerate(DOMS):
    m=copy.deepcopy(base);xx,yy=make_examples(domain_lines(dom,65,rng));train(m,xx,yy,24,seed*10+j,2.5e-3);experts.append(m)
   pairs,groups=probe_pool();F=task_fields(base,experts,pairs);norms,blocks=norm_blocks(F);ids,_=greedy_logdet_select_partitioned(blocks,16,groups);A=stack(F,ids,norms);Afull=stack(F,range(len(pairs)),norms)
   q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q)
   # ancestry coordinate recovery, three parent-count targets
   errs=[];f1s=[]
   for w in [[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]:
    m=merge(base,experts,w);df=(probs(m,qe)-probs(m,q))-br;oracle=fas_decompose(Afull,np.concatenate(df),support_threshold=.05)['pi'];pi=fas_decompose(A,np.concatenate([df[p] for p in ids]),support_threshold=.05)['pi'];errs.append(float(np.abs(pi-oracle).sum()));t=set(np.flatnonzero(np.asarray(w)>0));p=set(np.flatnonzero(pi>=.05));tp=len(t&p);pr=tp/max(1,len(p));rc=tp/max(1,len(t));f1s.append(2*pr*rc/max(pr+rc,1e-12))
   # open-set: known 3-bank vs hidden 4th, and quantized known hard negative
   F3=F[:,:3,:];n3,b3=norm_blocks(F3);ids3,_=greedy_logdet_select_partitioned(b3,16,groups);A3=stack(F3,ids3,n3)
   def residual(m):
    df=(probs(m,qe)-probs(m,q))-br;return fas_decompose(A3,np.concatenate([df[p] for p in ids3]),support_threshold=.05)['relative_residual']
   cal=[residual(merge(base,experts,[*rngnp.dirichlet(np.ones(3)),0.])) for _ in range(20)]
   scores=[];labs=[];qneg=[]
   for j in range(8):
    w3=rngnp.dirichlet(np.ones(3));scores.append(residual(merge(base,experts,[*w3,0.])));labs.append(0)
    ws=rngnp.dirichlet(np.ones(3));sci=float(rngnp.uniform(.3,.55));scores.append(residual(merge(base,experts,[(1-sci)*ws[0],(1-sci)*ws[1],(1-sci)*ws[2],sci])));labs.append(1)
    qneg.append(residual(quant8(merge(base,experts,[*rngnp.dirichlet(np.ones(3)),0.]))))
   rows.append(dict(size=label,params=nparam,seed=seed,l1_oracle=float(np.mean(errs)),parent_f1=float(np.mean(f1s)),open_auc=float(roc_auc_score(labs,scores)),known_residual=float(np.mean([s for s,y in zip(scores,labs) if y==0])),unknown_residual=float(np.mean([s for s,y in zip(scores,labs) if y==1])),quantized_known_residual=float(np.mean(qneg))))
   # checkpoint partial results after every seed
   pp=out/'transformer_capacity_rows_partial.csv'
   with pp.open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
 p=out/'transformer_capacity_rows.csv';
 with p.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
 agg=[]
 for label,*_ in SPECS:
  if label not in want: continue
  rr=[r for r in rows if r['size']==label];agg.append(dict(size=label,params=int(rr[0]['params']),l1_oracle=float(np.mean([r['l1_oracle'] for r in rr])),parent_f1=float(np.mean([r['parent_f1'] for r in rr])),open_auc=float(np.mean([r['open_auc'] for r in rr])),known_residual=float(np.mean([r['known_residual'] for r in rr])),unknown_residual=float(np.mean([r['unknown_residual'] for r in rr])),quantized_known_residual=float(np.mean([r['quantized_known_residual'] for r in rr]))))
 qout=out/'transformer_capacity_summary.csv';
 with qout.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=agg[0].keys());wr.writeheader();wr.writerows(agg)
 print(json.dumps({'seconds':time.time()-t0,'summary':agg},indent=2))
if __name__=='__main__':main()
