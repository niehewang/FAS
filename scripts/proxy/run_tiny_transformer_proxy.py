#!/usr/bin/env python3
from pathlib import Path
import sys, random, copy, csv, json, time
import numpy as np, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from fas_core.active_probe import greedy_logdet_select, greedy_logdet_select_partitioned, dictionary_geometry
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
CHARS=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*/=():,.;_[]{}<>?\n")
stoi={c:i+2 for i,c in enumerate(CHARS)}; PAD=0; UNK=1; V=len(stoi)+2
DOMS=['math','code','medical','science']

def enc(s,L=48):
    a=[stoi.get(c,UNK) for c in s[-L:]]; return [PAD]*(L-len(a))+a

def domain_lines(domain,n,rng):
    med=["Medical: aspirin is an analgesic used for pain.","Medical: fever may accompany infection.","Medical: hypertension means high blood pressure.","Medical: insulin helps regulate blood glucose.","Medical: antibiotics target susceptible bacteria.","Medical: dehydration may cause dizziness."]
    sci=["Science: force equals mass times acceleration.","Science: water is H2O and freezes near 0 C.","Science: energy can be measured in joules.","Science: DNA stores genetic information.","Science: light travels faster than sound.","Science: electrons carry negative charge."]
    out=[]
    for _ in range(n):
        a,b=rng.randint(1,30),rng.randint(1,30)
        if domain=='math': out.append(f"Math problem: {a}+{b}={a+b}. {a}*{b}={a*b}.")
        elif domain=='code': out.append(f"Code: def add_{a}(x): return x+{a}. vals_{b}=[i*i for i in range({b})].")
        elif domain=='medical': out.append(rng.choice(med))
        else: out.append(rng.choice(sci))
    return out

def make_examples(lines,L=48):
    xs=[];ys=[]
    for s in lines:
        ids=[stoi.get(c,UNK) for c in s]
        stride=max(1,len(ids)//8)
        for j in range(4,len(ids),stride):
            pre=ids[max(0,j-L):j];xs.append([PAD]*(L-len(pre))+pre);ys.append(ids[j])
    return torch.tensor(xs),torch.tensor(ys)

class TinyTransformerLM(nn.Module):
    def __init__(self,d=48,nhead=4,layers=2,ff=96,L=48):
        super().__init__(); self.L=L
        self.tok=nn.Embedding(V,d,padding_idx=PAD); self.pos=nn.Embedding(L,d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=ff,dropout=0.0,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,num_layers=layers); self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,V)
    def forward(self,x):
        B,L=x.shape; pos=torch.arange(L,device=x.device)[None,:].expand(B,L); z=self.tok(x)+self.pos(pos)
        mask=(x==PAD); h=self.enc(z,src_key_padding_mask=mask); return self.head(self.ln(h[:,-1]))

def train(m,X,Y,steps,seed,lr=2e-3):
    torch.manual_seed(seed); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01);n=len(X);m.train()
    for _ in range(steps):
        idx=torch.randint(0,n,(48,));loss=nn.functional.cross_entropy(m(X[idx]),Y[idx]);opt.zero_grad();loss.backward();opt.step()
    return m

def probs(m,prompts):
    m.eval();X=torch.tensor([enc(x) for x in prompts])
    with torch.no_grad():return torch.softmax(m(X),-1).numpy()

def probe_pool(n_per=24):
    pairs=[];groups=[]
    meds=[("Medical: aspirin is an","Medical: insulin is a"),("Medical: fever may accompany","Medical: hypertension may involve"),("Medical: antibiotics target","Medical: dehydration may cause")]
    scis=[("Science: force equals mass times","Science: energy is measured in"),("Science: water is","Science: DNA stores"),("Science: light travels","Science: electrons carry")]
    for i in range(n_per):
        a=2+i%20;b=3+(i*3)%20
        pairs.append((f"Math problem: {a}+{b}=",f"Math problem: {a+1}+{b}="));groups.append('math')
        pairs.append((f"Code: def add_{a}(x): return x+",f"Code: def add_{a+1}(x): return x+"));groups.append('code')
        pairs.append(meds[i%len(meds)]);groups.append('medical')
        pairs.append(scis[i%len(scis)]);groups.append('science')
    return pairs,groups

def task_fields(base,experts,pairs):
    q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q)
    return np.stack([(probs(e,qe)-probs(e,q))-br for e in experts],axis=1)

def norm_blocks(F):
    norms=np.sqrt(np.sum(F**2,axis=(0,2)));blocks=[F[p].T/np.maximum(norms,1e-12)[None,:] for p in range(len(F))];return norms,blocks

def stack(F,ids,norms):return np.concatenate([F[p].T/np.maximum(norms,1e-12)[None,:] for p in ids],axis=0)

def merge(base,experts,w):
    m=copy.deepcopy(base);sd=m.state_dict();b=base.state_dict();es=[e.state_dict() for e in experts]
    with torch.no_grad():
        for k in sd: sd[k]=b[k].clone()+sum(float(wi)*(es[i][k]-b[k]) for i,wi in enumerate(w))
    m.load_state_dict(sd);return m

def support_f1(pi,w,thr=.05):
    t=set(np.flatnonzero(np.asarray(w)>0));p=set(np.flatnonzero(np.asarray(pi)>=thr));tp=len(t&p);pr=tp/max(1,len(p));rc=tp/max(1,len(t));return 2*pr*rc/max(1e-12,pr+rc)

def main():
    t0=time.time();out=ROOT/'results/proxy';out.mkdir(parents=True,exist_ok=True);rows=[]
    targets=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]
    for seed in [11,23,47]:
        rng=random.Random(seed);torch.manual_seed(seed);rngnp=np.random.default_rng(seed)
        base=TinyTransformerLM();X,Y=make_examples(sum([domain_lines(d,55,rng) for d in DOMS],[]));train(base,X,Y,45,seed,2e-3)
        experts=[]
        for j,d in enumerate(DOMS):
            m=copy.deepcopy(base);xx,yy=make_examples(domain_lines(d,80,rng));train(m,xx,yy,28,seed*10+j,2.5e-3);experts.append(m)
        pairs,groups=probe_pool();F=task_fields(base,experts,pairs);norms,blocks=norm_blocks(F)
        selF=F+0.001*rngnp.normal(size=F.shape);_,selblocks=norm_blocks(selF)
        q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q);Afull=stack(F,range(len(pairs)),norms)
        desc=[];oracle=[]
        for w in targets:
            d=merge(base,experts,w);df=(probs(d,qe)-probs(d,q))-br;desc.append(df);oracle.append(fas_decompose(Afull,np.concatenate(df),support_threshold=.05)['pi'])
        for B in [8,16,32]:
            dopt,_=greedy_logdet_select(selblocks,B);bal,_=greedy_logdet_select_partitioned(selblocks,B,groups)
            for strat in ['DOpt','BalancedDOpt','Random']:
                reps=1 if strat!='Random' else 6
                for rep in range(reps):
                    ids=dopt if strat=='DOpt' else bal if strat=='BalancedDOpt' else rngnp.choice(len(pairs),B,replace=False).tolist();A=stack(F,ids,norms);geom=dictionary_geometry(A)
                    for ti,w in enumerate(targets):
                        pi=fas_decompose(A,np.concatenate([desc[ti][p] for p in ids]),support_threshold=.05)['pi']
                        rows.append(dict(seed=seed,budget=B,strategy=strat,repeat=rep,target=ti,parent_count=sum(np.asarray(w)>0),l1_oracle=float(np.abs(pi-oracle[ti]).sum()),parent_f1=support_f1(pi,w),sigma_min=geom['sigma_min_raw'],coherence=geom['coherence']))
    p=out/'tiny_transformer_proxy_rows.csv'
    with p.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    agg=[]
    for B in [8,16,32]:
        for s in ['DOpt','BalancedDOpt','Random']:
            rr=[x for x in rows if x['budget']==B and x['strategy']==s]
            agg.append(dict(strategy=s,budget=B,l1_oracle=float(np.mean([x['l1_oracle'] for x in rr])),parent_f1=float(np.mean([x['parent_f1'] for x in rr])),sigma_min=float(np.mean([x['sigma_min'] for x in rr])),coherence=float(np.mean([x['coherence'] for x in rr]))))
    qout=out/'tiny_transformer_proxy_summary.csv'
    with qout.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=agg[0].keys());wr.writeheader();wr.writerows(agg)
    print(json.dumps({'seconds':time.time()-t0,'summary':agg},indent=2))
if __name__=='__main__':main()
