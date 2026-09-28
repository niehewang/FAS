#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, random, re
from collections import defaultdict
from pathlib import Path

DOMAINS=("Math","Medical","Science")
WS=re.compile(r"\s+")
NONWORD=re.compile(r"[^\w]+",re.UNICODE)

def normalize_text(x):
    x=str(x or '').lower().strip(); x=NONWORD.sub(' ',x); return WS.sub(' ',x).strip()
def stable_text_id(domain,prompt):
    s='\n'.join(normalize_text(p) for p in (domain,prompt)); return 'seed_'+hashlib.sha256(s.encode()).hexdigest()[:16]
def ngrams(text,n=8):
    t=normalize_text(text).split()
    if not t:return set()
    if len(t)<n:return {' '.join(t)}
    return {' '.join(t[i:i+n]) for i in range(len(t)-n+1)}
def build_ref(rows,n=8):
    ref=[]; inv=defaultdict(set)
    for j,r in enumerate(rows):
        txt=f"{r.get('prompt','')}\n{r.get('response','')}"; ng=ngrams(txt,n); norm=normalize_text(txt); ref.append((ng,norm))
        for g in ng:inv[g].add(j)
    return ref,inv
def max_overlap(text,ref,inv,n=8):
    norm=normalize_text(text); ng=ngrams(norm,n); cand=set()
    for g in ng:cand.update(inv.get(g,()))
    best=(False,0.0,0.0)
    for j in cand:
        rg,rnorm=ref[j]; exact=(norm==rnorm); inter=len(ng&rg); union=len(ng|rg) or 1; denom=min(len(ng),len(rg)) or 1
        cur=(exact,inter/union,inter/denom)
        if exact or (cur[1],cur[2])>(best[1],best[2]):best=cur
    return {'exact':best[0],'jaccard':best[1],'containment':best[2]}
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def last_boxed(s):
    s=str(s or ''); starts=[s.rfind('\\boxed{'),s.rfind('\\fbox{')]; i=max(starts)
    if i<0:return None
    j=s.find('{',i); depth=0
    for k in range(j,len(s)):
        if s[k]=='{':depth+=1
        elif s[k]=='}':
            depth-=1
            if depth==0:return s[j+1:k]
    return None

def simple_number(x):
    if x is None:return None
    s=str(x).strip().replace('$','').replace(',','').replace('\\!','').replace('\\,','').replace(' ','')
    # simple LaTeX fraction only; intentionally skip symbolic/radical answers
    m=re.fullmatch(r'[-+]?\\(?:d|t)?frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}',s)
    if m:
        b=float(m.group(2)); return None if b==0 else float(m.group(1))/b
    m=re.fullmatch(r'([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)',s)
    if m:
        b=float(m.group(2)); return None if b==0 else float(m.group(1))/b
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',s):return float(s)
    return None

def science_formal_and_utility(x):
    q=str(x.get('question','')).strip(); c=x.get('choices',{}) or {}; labels=list(c.get('label',[]) or []); texts=list(c.get('text',[]) or []); key=str(x.get('answerKey','')).strip()
    if not q or not labels or len(labels)!=len(texts) or key not in labels:return None
    formal=q+'\n'+'\n'.join(f'{a}. {b}' for a,b in zip(labels,texts))
    idx=labels.index(key); letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'; answer=letters[idx]
    util=q+'\n'+'\n'.join(f'{letters[i]}. {b}' for i,b in enumerate(texts))+'\nAnswer with the option letter only.'
    return formal,util,answer

def medical_formal_and_utility(x):
    q=str(x.get('question','')).strip(); ctx=x.get('context',{}) or {}; cs=ctx.get('contexts',[]) if isinstance(ctx,dict) else []
    formal=(q+'\nContext: '+' '.join(map(str,cs))).strip(); ans=str(x.get('final_decision','')).strip().lower()
    if not q or ans not in {'yes','no','maybe'}:return None
    return formal, formal+'\nAnswer only yes, no, or maybe.', ans

def math_formal_and_utility(x):
    p=str(x.get('problem','')).strip(); ans=simple_number(last_boxed(x.get('solution','')))
    if not p or ans is None or not math.isfinite(ans):return None
    return p, p+'\nGive only the final numeric value (use a decimal or simple fraction if needed).', repr(float(ans))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed-pool',required=True); ap.add_argument('--probe-manifest',required=True); ap.add_argument('--data-lock',required=True); ap.add_argument('--training-dir',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--n-per-domain',type=int,default=96); ap.add_argument('--selection-seed',default='20260926_stage6_v1_1')
    a=ap.parse_args()
    seed_pool=read_jsonl(a.seed_pool); used={d:set() for d in DOMAINS}
    for r in seed_pool:
        d=str(r.get('domain','')); 
        if d in used:used[d].add(str(r.get('seed_id','')))
    pm=json.loads(Path(a.probe_manifest).read_text()); expected=pm.get('output_sha256'); actual=sha256(a.seed_pool)
    if expected and expected!=actual:raise SystemExit(f'probe seed SHA mismatch: {actual} != {expected}')
    lock=json.loads(Path(a.data_lock).read_text()); train={}
    for d in DOMAINS:
        fn=d.lower()+'.jsonl'; p=Path(a.training_dir)/fn
        if not p.exists():raise SystemExit(f'missing training file {p}')
        exp=lock['domains'][d.lower()]['sha256']; act=sha256(p)
        if exp!=act:raise SystemExit(f'training SHA mismatch {d}: {act} != {exp}')
        train[d]=read_jsonl(p)
    refs={d:build_ref(train[d]) for d in DOMAINS}
    entries=pm['sources']
    def revision(dataset,config=None):
        vals={e['revision'] for e in entries if e['dataset']==dataset and (config is None or e.get('config')==config)}
        if len(vals)!=1:raise SystemExit(f'cannot resolve frozen revision for {dataset}/{config}: {vals}')
        return next(iter(vals))
    from datasets import load_dataset
    candidates={d:{} for d in DOMAINS}; stats={d:defaultdict(int) for d in DOMAINS}; provenance=[]
    # Math: all seven frozen MATH configs, exact frozen revision.
    mcfg=['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']; mrev=revision('EleutherAI/hendrycks_math')
    for cfg in mcfg:
        ds=load_dataset('EleutherAI/hendrycks_math',cfg,split='test',revision=mrev)
        provenance.append({'dataset':'EleutherAI/hendrycks_math','config':cfg,'split':'test','revision':mrev})
        for idx,x in enumerate(ds):
            stats['Math']['raw']+=1; z=math_formal_and_utility(x)
            if z is None:stats['Math']['unscorable']+=1; continue
            formal,prompt,ans=z; sid=stable_text_id('Math',formal)
            if sid in used['Math']:stats['Math']['probe_overlap']+=1; continue
            ov=max_overlap(formal+'\n'+ans,*refs['Math'])
            if ov['exact'] or ov['jaccard']>=.45 or ov['containment']>=.70:stats['Math']['training_overlap']+=1; continue
            candidates['Math'][sid]={'id':sid,'domain':'Math','prompt':prompt,'metric':'numeric','answer':ans,'source_dataset':'EleutherAI/hendrycks_math','source_config':cfg,'source_split':'test','source_revision':mrev,'source_row_index':idx}
    # Medical
    drev=revision('qiaojin/PubMedQA','pqa_labeled'); ds=load_dataset('qiaojin/PubMedQA','pqa_labeled',split='train',revision=drev); provenance.append({'dataset':'qiaojin/PubMedQA','config':'pqa_labeled','split':'train','revision':drev})
    for idx,x in enumerate(ds):
        stats['Medical']['raw']+=1; z=medical_formal_and_utility(x)
        if z is None:stats['Medical']['unscorable']+=1; continue
        formal,prompt,ans=z; sid=stable_text_id('Medical',formal)
        if sid in used['Medical']:stats['Medical']['probe_overlap']+=1; continue
        ov=max_overlap(formal+'\n'+ans,*refs['Medical'])
        if ov['exact'] or ov['jaccard']>=.45 or ov['containment']>=.70:stats['Medical']['training_overlap']+=1; continue
        candidates['Medical'][sid]={'id':sid,'domain':'Medical','prompt':prompt,'metric':'choice','answer':ans,'source_dataset':'qiaojin/PubMedQA','source_config':'pqa_labeled','source_split':'train','source_revision':drev,'source_row_index':idx}
    # Science, both ARC configs.
    srev=revision('allenai/ai2_arc')
    for cfg in ['ARC-Easy','ARC-Challenge']:
        ds=load_dataset('allenai/ai2_arc',cfg,split='test',revision=srev); provenance.append({'dataset':'allenai/ai2_arc','config':cfg,'split':'test','revision':srev})
        for idx,x in enumerate(ds):
            stats['Science']['raw']+=1; z=science_formal_and_utility(x)
            if z is None:stats['Science']['unscorable']+=1; continue
            formal,prompt,ans=z; sid=stable_text_id('Science',formal)
            if sid in used['Science']:stats['Science']['probe_overlap']+=1; continue
            ov=max_overlap(formal+'\n'+ans,*refs['Science'])
            if ov['exact'] or ov['jaccard']>=.45 or ov['containment']>=.70:stats['Science']['training_overlap']+=1; continue
            candidates['Science'][sid]={'id':sid,'domain':'Science','prompt':prompt,'metric':'choice','answer':ans,'source_dataset':'allenai/ai2_arc','source_config':cfg,'source_split':'test','source_revision':srev,'source_row_index':idx}
    selected=[]; domains={}
    for d in DOMAINS:
        arr=list(candidates[d].values()); stats[d]['eligible_unique']=len(arr)
        if len(arr)<a.n_per_domain:raise SystemExit(f'{d}: only {len(arr)} eligible held-out utility rows; need {a.n_per_domain}')
        arr.sort(key=lambda r:hashlib.sha256((a.selection_seed+'|'+d+'|'+r['id']).encode()).hexdigest()); take=arr[:a.n_per_domain]; selected.extend(take)
        domains[d]={**dict(stats[d]),'selected':len(take),'selected_ids':[r['id'] for r in take]}
        if any(r['id'] in used[d] for r in take):raise AssertionError('probe overlap survived')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    man={'protocol':'FAS_STAGE6_HELDOUT_UTILITY_V1','status':'frozen_before_stage6_model_evaluation','selection_seed':a.selection_seed,'n_per_domain':a.n_per_domain,'probe_seed_sha256':actual,'formal_data_lock_sha256':sha256(a.data_lock),'training_overlap_thresholds':{'ngram':8,'jaccard':.45,'containment':.70},'domains':domains,'sources':provenance,'output':str(out),'output_sha256':sha256(out)}
    Path(a.manifest).write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'sha256':man['output_sha256'],'domains':{d:{k:v for k,v in domains[d].items() if k!='selected_ids'} for d in DOMAINS}},indent=2))
if __name__=='__main__':main()
