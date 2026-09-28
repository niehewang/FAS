#!/usr/bin/env python3
from pathlib import Path
import sys, random, copy, csv, json, time
import numpy as np, torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'code'))
from fas_core.active_probe import greedy_logdet_select,dictionary_geometry
from fas_core.decompose import fas_decompose

torch.set_num_threads(5)
CHARS=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-*/=():,.;_[]{}<>?\n")
stoi={c:i+2 for i,c in enumerate(CHARS)}; PAD=0; UNK=1; V=len(stoi)+2
DOMS=['math','code','medical','science']

def enc(s,L=64):
    a=[stoi.get(c,UNK) for c in s[-L:]]; return [PAD]*(L-len(a))+a

def domain_lines(domain,n,rng):
    out=[]
    med=["Medical: aspirin is an analgesic used for pain.","Medical: fever may accompany infection.","Medical: hypertension means high blood pressure.","Medical: insulin helps regulate blood glucose.","Medical: antibiotics target susceptible bacteria.","Medical: dehydration may cause dizziness."]
    sci=["Science: force equals mass times acceleration.","Science: water is H2O and freezes near 0 C.","Science: energy can be measured in joules.","Science: DNA stores genetic information.","Science: light travels faster than sound.","Science: electrons carry negative charge."]
    for _ in range(n):
        a,b=rng.randint(1,40),rng.randint(1,40)
        if domain=='math': out.append(f"Math problem: {a}+{b}={a+b}. {a}*{b}={a*b}. {a+b}-{a}={b}.")
        elif domain=='code': out.append(f"Code: def add_{a}(x): return x+{a}. vals_{b}=[i*i for i in range({b})].")
        elif domain=='medical': out.append(rng.choice(med))
        else: out.append(rng.choice(sci))
    return out

def make_examples(lines,L=64):
    xs=[];ys=[]
    for s in lines:
        ids=[stoi.get(c,UNK) for c in s]
        stride=max(1,len(ids)//10)
        for j in range(4,len(ids),stride):
            pre=ids[max(0,j-L):j];xs.append([PAD]*(L-len(pre))+pre);ys.append(ids[j])
    return torch.tensor(xs),torch.tensor(ys)

class LM(nn.Module):
    def __init__(self,emb=48,hid=96):
        super().__init__();self.emb=nn.Embedding(V,emb,padding_idx=PAD);self.gru=nn.GRU(emb,hid,batch_first=True);self.head=nn.Linear(hid,V)
    def forward(self,x):
        z=self.emb(x);_,h=self.gru(z);return self.head(h[-1])

def train(m,X,Y,steps,seed,lr):
    torch.manual_seed(seed);opt=torch.optim.AdamW(m.parameters(),lr=lr);n=len(X);m.train()
    for _ in range(steps):
        idx=torch.randint(0,n,(64,));loss=nn.functional.cross_entropy(m(X[idx]),Y[idx]);opt.zero_grad();loss.backward();opt.step()
    return m

def probs(m,prompts):
    m.eval();X=torch.tensor([enc(x) for x in prompts])
    with torch.no_grad():return torch.softmax(m(X),-1).numpy()

def probe_pool(n_per=48):
    pairs=[];groups=[]
    for i in range(n_per):
        a=2+i%25;b=3+(i*3)%25
        pairs.append((f"Math problem: {a}+{b}=",f"Math problem: {a+1}+{b}="));groups.append('math')
        pairs.append((f"Code: def add_{a}(x): return x+",f"Code: def add_{a+1}(x): return x+"));groups.append('code')
        meds=[("Medical: aspirin is an","Medical: insulin is a"),("Medical: fever may accompany","Medical: hypertension may involve"),("Medical: antibiotics target","Medical: dehydration may cause")]
        pairs.append(meds[i%len(meds)]);groups.append('medical')
        scis=[("Science: force equals mass times","Science: energy is measured in"),("Science: water is","Science: DNA stores"),("Science: light travels","Science: electrons carry")]
        pairs.append(scis[i%len(scis)]);groups.append('science')
    return pairs,groups

def task_fields(base,experts,pairs):
    q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q)
    return np.stack([(probs(e,qe)-probs(e,q))-br for e in experts],axis=1)

def norms_blocks(F):
    norms=np.sqrt(np.sum(F**2,axis=(0,2)));blocks=[F[p].T/np.maximum(norms,1e-12)[None,:] for p in range(len(F))];return norms,blocks

def stack(F,ids,norms):return np.concatenate([F[p].T/np.maximum(norms,1e-12)[None,:] for p in ids],0)

def merge(base,experts,w):
    m=copy.deepcopy(base);sd=m.state_dict();b=base.state_dict();es=[e.state_dict() for e in experts]
    for k in sd:sd[k]=b[k].clone()+sum(float(wi)*(es[i][k]-b[k]) for i,wi in enumerate(w))
    m.load_state_dict(sd);return m

def balanced_select(blocks,B,groups):
    k=blocks[0].shape[1];H=1e-6*np.eye(k);sel=[];rem=set(range(len(blocks)));uniq=DOMS;base=B//4;extra=B%4;quota={g:base+(i<extra) for i,g in enumerate(uniq)};cnt={g:0 for g in uniq}
    def ld(M):
        s,v=np.linalg.slogdet(M);return v if s>0 else -1e99
    for _ in range(B):
        cur=ld(H);best=None;gain=-1e99
        for i in rem:
            g=groups[i]
            if cnt[g]>=quota[g]:continue
            x=ld(H+blocks[i].T@blocks[i])-cur
            if x>gain:best=i;gain=x
        if best is None:break
        sel.append(best);rem.remove(best);cnt[groups[best]]+=1;H+=blocks[best].T@blocks[best]
    return sel

def f1_support(pi,w,thr=.05):
    true=set(np.flatnonzero(np.asarray(w)>0));pred=set(np.flatnonzero(np.asarray(pi)>=thr));tp=len(true&pred);pr=tp/max(1,len(pred));rc=tp/max(1,len(true));return 2*pr*rc/max(1e-12,pr+rc)

def main():
    out=ROOT/'results/proxy';out.mkdir(parents=True,exist_ok=True);rows=[];neg=[];t0=time.time()
    targets=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]
    for seed in [11,23,47]:
        rng=random.Random(seed);torch.manual_seed(seed)
        base=LM();X,Y=make_examples(sum([domain_lines(d,100,rng) for d in DOMS],[]));train(base,X,Y,100,seed,2.5e-3)
        experts=[]
        for j,d in enumerate(DOMS):
            m=copy.deepcopy(base);xx,yy=make_examples(domain_lines(d,160,rng));train(m,xx,yy,70,seed*10+j,3e-3);experts.append(m)
        pairs,groups=probe_pool();F=task_fields(base,experts,pairs);norms,blocks=norms_blocks(F);rngnp=np.random.default_rng(seed);selF=F+0.0015*rngnp.normal(size=F.shape);_,selblocks=norms_blocks(selF)
        q=[a for a,b in pairs];qe=[b for a,b in pairs];br=probs(base,qe)-probs(base,q);desc=[];oracle=[];Afull=stack(F,range(len(pairs)),norms)
        for w in targets:
            d=merge(base,experts,w);df=(probs(d,qe)-probs(d,q))-br;desc.append(df);oracle.append(fas_decompose(Afull,np.concatenate(df),support_threshold=.05)['pi'])
        for B in [8,16,32,64]:
            act,_=greedy_logdet_select(selblocks,B);bal=balanced_select(selblocks,B,groups)
            for strat in ['DOpt','BalancedDOpt','Random']:
                reps=1 if strat!='Random' else 8
                for rep in range(reps):
                    ids=act if strat=='DOpt' else bal if strat=='BalancedDOpt' else rngnp.choice(len(pairs),B,replace=False).tolist();A=stack(F,ids,norms);geom=dictionary_geometry(A)
                    for ti,w in enumerate(targets):
                        y=np.concatenate([desc[ti][p] for p in ids]);pi=fas_decompose(A,y,support_threshold=.05)['pi'];rows.append(dict(seed=seed,budget=B,strategy=strat,repeat=rep,parent_count=int(np.sum(np.asarray(w)>0)),target=ti,l1_oracle=float(np.abs(pi-oracle[ti]).sum()),parent_f1=f1_support(pi,w),sigma_min=geom['sigma_min_raw'],coherence=geom['coherence']))
        # Negative controls at B=16 Balanced
        ids=balanced_select(selblocks,16,groups);A=stack(F,ids,norms)
        # wrong pair: permute edited probe responses for target
        perm=ids.copy();rngnp.shuffle(perm)
        # wrong anchor: use expert0 as anchor for descendant response instead of base
        e0r=probs(experts[0],qe)-probs(experts[0],q)
        for ti,w in enumerate(targets):
            y_good=np.concatenate([desc[ti][p] for p in ids]);pi_good=fas_decompose(A,y_good,support_threshold=.05)['pi']
            y_pair=np.concatenate([desc[ti][p] for p in perm]);pi_pair=fas_decompose(A,y_pair,support_threshold=.05)['pi']
            dmodel=merge(base,experts,w);dr=(probs(dmodel,qe)-probs(dmodel,q))-e0r;y_anchor=np.concatenate([dr[p] for p in ids]);pi_anchor=fas_decompose(A,y_anchor,support_threshold=.05)['pi']
            for name,pi in [('Correct',pi_good),('PairShuffle',pi_pair),('WrongAnchor',pi_anchor)]:neg.append(dict(seed=seed,control=name,target=ti,l1_oracle=float(np.abs(pi-oracle[ti]).sum()),parent_f1=f1_support(pi,w)))
    # save
    for fname,data in [('medium_proxy_rows.csv',rows),('medium_proxy_negative_controls.csv',neg)]:
        with (out/fname).open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=data[0].keys());wr.writeheader();wr.writerows(data)
    agg=[]
    for B in [8,16,32,64]:
        for s in ['DOpt','BalancedDOpt','Random']:
            rr=[x for x in rows if x['budget']==B and x['strategy']==s];agg.append(dict(strategy=s,budget=B,l1_oracle=np.mean([x['l1_oracle'] for x in rr]),parent_f1=np.mean([x['parent_f1'] for x in rr]),sigma_min=np.mean([x['sigma_min'] for x in rr]),coherence=np.mean([x['coherence'] for x in rr])))
    with (out/'medium_proxy_summary.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=agg[0].keys());wr.writeheader();wr.writerows(agg)
    nagg=[]
    for c in ['Correct','PairShuffle','WrongAnchor']:
        rr=[x for x in neg if x['control']==c];nagg.append(dict(control=c,l1_oracle=np.mean([x['l1_oracle'] for x in rr]),parent_f1=np.mean([x['parent_f1'] for x in rr])))
    with (out/'medium_proxy_negative_summary.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=nagg[0].keys());wr.writeheader();wr.writerows(nagg)
    print(json.dumps({'seconds':time.time()-t0,'summary':agg,'negative':nagg},indent=2))
if __name__=='__main__':main()
