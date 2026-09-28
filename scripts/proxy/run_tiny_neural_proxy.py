#!/usr/bin/env python3
from pathlib import Path
import sys,csv,random,copy
import numpy as np, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
CHARS=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*/=():,.;_[]{}<>?\n")
stoi={c:i+2 for i,c in enumerate(CHARS)}; PAD=0; UNK=1; V=len(stoi)+2

def enc(s,L=64):
    a=[stoi.get(c,UNK) for c in s[-L:]]; return [PAD]*(L-len(a))+a

def domain_lines(domain,n,rng):
    out=[]
    for _ in range(n):
        a,b=rng.randint(1,30),rng.randint(1,30)
        if domain=='math': out.append(f"Math problem: {a}+{b}={a+b}. {a}*{b}={a*b}.")
        elif domain=='code': out.append(f"Code: def add_{a}(x): return x+{a}. list_{b}=[i for i in range({b})].")
        elif domain=='medical': out.append(rng.choice(["Medical: aspirin is an analgesic used for pain.","Medical: fever may accompany infection.","Medical: hypertension means high blood pressure.","Medical: insulin helps regulate blood glucose."]))
        else: out.append(rng.choice(["Science: force equals mass times acceleration.","Science: water is H2O and freezes near 0 C.","Science: energy can be measured in joules.","Science: DNA stores genetic information."]))
    return out

def make_examples(lines,L=64):
    xs=[]; ys=[]
    for s in lines:
        ids=[stoi.get(c,UNK) for c in s]
        for j in range(4,len(ids),max(1,len(ids)//8)):
            pre=ids[max(0,j-L):j]; x=[PAD]*(L-len(pre))+pre; xs.append(x); ys.append(ids[j])
    return torch.tensor(xs),torch.tensor(ys)

class LM(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(V,16,padding_idx=PAD); self.gru=nn.GRU(16,32,batch_first=True); self.head=nn.Linear(32,V)
    def forward(self,x):
        z=self.emb(x); _,h=self.gru(z); return self.head(h[-1])

def train(model,X,Y,steps,seed,lr=3e-3):
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=lr); n=len(X); model.train()
    for t in range(steps):
        idx=torch.randint(0,n,(64,)); logits=model(X[idx]); loss=nn.functional.cross_entropy(logits,Y[idx]); opt.zero_grad();loss.backward();opt.step()
    return model

def probs(model,prompts):
    model.eval(); X=torch.tensor([enc(x) for x in prompts]);
    with torch.no_grad(): return torch.softmax(model(X),-1).numpy()

def probe_pool(n=160):
    pairs=[]
    for i in range(n//4):
        a=2+i%20;b=3+(i*3)%20; pairs.append((f"Math problem: {a}+{b}=",f"Math problem: {a+1}+{b}="))
        pairs.append((f"Code: def add_{a}(x): return x+",f"Code: def add_{a+1}(x): return x+"))
        med=[("Medical: aspirin is an", "Medical: insulin is a"),("Medical: fever may accompany", "Medical: hypertension may involve")][i%2];pairs.append(med)
        sci=[("Science: force equals mass times", "Science: energy is measured in"),("Science: water is", "Science: DNA stores")][i%2];pairs.append(sci)
    return pairs[:n]

def task_fields(base,experts,pairs):
    q=[a for a,b in pairs]; qe=[b for a,b in pairs]; br=probs(base,qe)-probs(base,q)
    fs=[]
    for m in experts: fs.append((probs(m,qe)-probs(m,q))-br)
    return np.stack(fs,axis=1) # N,K,V

def norm_and_blocks(F):
    norms=np.sqrt(np.sum(F**2,axis=(0,2))); blocks=[F[p].T/np.maximum(norms,1e-12)[None,:] for p in range(len(F))]; return norms,blocks

def stack(F,ids,norms): return np.concatenate([F[p].T/norms[None,:] for p in ids],0)
def merge(base,experts,weights):
    m=copy.deepcopy(base); sd=m.state_dict(); bsd=base.state_dict(); es=[e.state_dict() for e in experts]
    for k in sd: sd[k]=bsd[k].clone()+sum(float(w)*(es[i][k]-bsd[k]) for i,w in enumerate(weights))
    m.load_state_dict(sd); return m

def main():
    out=ROOT/'results/proxy'; out.mkdir(parents=True,exist_ok=True); rows=[]
    for seed in [11,23,47]:
        rng=random.Random(seed); torch.manual_seed(seed)
        base=LM(); base_lines=sum([domain_lines(d,80,rng) for d in ['math','code','medical','science']],[]); X,Y=make_examples(base_lines);train(base,X,Y,80,seed,3e-3)
        experts=[]
        for j,d in enumerate(['math','code','medical','science']):
            m=copy.deepcopy(base); xx,yy=make_examples(domain_lines(d,120,rng));train(m,xx,yy,50,seed*10+j,4e-3);experts.append(m)
        pairs=probe_pool(); F=task_fields(base,experts,pairs); norms,blocks=norm_and_blocks(F)
        # independent selection noise to mimic bank-select / estimate
        rngnp=np.random.default_rng(seed); selF=F+0.002*rngnp.normal(size=F.shape); _,selblocks=norm_and_blocks(selF)
        weights_list=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]
        q=[a for a,b in pairs]; qe=[b for a,b in pairs]; bq=probs(base,q); bqe=probs(base,qe)
        desc_fields=[]
        for w in weights_list:
            desc=merge(base,experts,w); desc_fields.append((probs(desc,qe)-probs(desc,q))-(bqe-bq))
        for B in [8,16,32,64]:
            act,_=greedy_logdet_select(selblocks,B)
            for strat in ['Active','Random']:
                reps=1 if strat=='Active' else 5
                for rep in range(reps):
                    ids=act if strat=='Active' else rngnp.choice(len(pairs),B,replace=False).tolist();A=stack(F,ids,norms);geom=dictionary_geometry(A)
                    for ti,w in enumerate(weights_list):
                        DF=desc_fields[ti]; y=np.concatenate([DF[p] for p in ids]); dec=fas_decompose(A,y,support_threshold=.05)
                        # exact functional coordinate induced by global normalization
                        ref=np.array(w)*norms; ref=ref/ref.sum()
                        l1=float(np.abs(dec['pi']-ref).sum()); true=set(np.flatnonzero(np.array(w)>0)); pred=set(dec['support']);tp=len(true&pred);pr=tp/max(1,len(pred));rc=tp/max(1,len(true));f1=2*pr*rc/max(pr+rc,1e-12)
                        rows.append(dict(seed=seed,budget=B,strategy=strat,repeat=rep,target=ti,sigma_min=geom['sigma_min_raw'],coherence=geom['coherence'],l1=l1,f1=f1,rho=dec['rho']))
    p=out/'tiny_neural_proxy_rows.csv';
    with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    agg=[]
    for B in [8,16,32,64]:
        for s in ['Active','Random']:
            rr=[x for x in rows if x['budget']==B and x['strategy']==s]
            agg.append(dict(method=s,budget=B,sigma_min=np.mean([x['sigma_min'] for x in rr]),coherence=np.mean([x['coherence'] for x in rr]),l1_error=np.mean([x['l1'] for x in rr]),parent_f1=np.mean([x['f1'] for x in rr]),rho=np.mean([x['rho'] for x in rr])))
    q=out/'tiny_neural_proxy_summary.csv';
    with q.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
    print(q); [print(x) for x in agg]
if __name__=='__main__': main()
