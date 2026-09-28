#!/usr/bin/env python3
"""Build a deterministic candidate intervention pool from frozen formal probe seeds.

The pool is intentionally larger than the online budget. All rows remain auditable;
semantic edits that can be brittle are marked needs_review and never silently
promoted to a final locked pool without validate_formal_interventions.py.
"""
from __future__ import annotations
import argparse, json, re, hashlib, random
from pathlib import Path

NUM=re.compile(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)')
AGE=re.compile(r'\b(\d{1,2})[- ]year[- ]old\b',re.I)
GENDER=[(re.compile(r'\bmale\b',re.I),'female'),(re.compile(r'\bfemale\b',re.I),'male'),(re.compile(r'\bman\b',re.I),'woman'),(re.compile(r'\bwoman\b',re.I),'man')]
REL=[('before','after'),('after','before'),('increase','decrease'),('decrease','increase'),('greater than','less than'),('less than','greater than')]

def numeric_edit(q:str,step:int=1):
    m=NUM.search(q)
    if not m:return None
    raw=m.group(1)
    try:
        x=float(raw)
        # Avoid label years / huge identifiers; produce a small but visible local edit.
        if abs(x)>10000:return None
        y=x+step if abs(x)<100 else x*1.05
        rep=str(int(round(y))) if raw.lstrip('-').isdigit() else f'{y:.4g}'
    except Exception:return None
    out=q[:m.start(1)]+rep+q[m.end(1):]
    return out if out!=q else None

def relation_edit(q:str):
    low=q.lower()
    for a,b in REL:
        i=low.find(a)
        if i>=0:return q[:i]+b+q[i+len(a):]
    return None

def medical_edit(q:str,idx:int):
    a=AGE.search(q)
    if a:
        old=int(a.group(1)); new=min(90,max(18,old+(10 if idx%2==0 else -10)))
        if new!=old:return q[:a.start(1)]+str(new)+q[a.end(1):], 'patient_age', True
    for pat,repl in GENDER:
        m=pat.search(q)
        if m:return q[:m.start()]+repl+q[m.end():], 'patient_sex', True
    suffix='\nAdditional patient context: assume normal renal function.' if idx%2==0 else '\nAdditional patient context: assume no known drug allergies.'
    return q+suffix,'clinical_context',True

def code_edit(q:str,idx:int):
    suffixes=[
      ('\nAdditional constraint: do not use recursion.','constraint_no_recursion'),
      ('\nAdditional constraint: use O(1) auxiliary memory when feasible.','constraint_memory'),
      ('\nAdditional constraint: handle empty input explicitly.','constraint_edge_case'),
      ('\nAdditional constraint: preserve the original function signature.','constraint_signature'),
    ]
    s,t=suffixes[idx%len(suffixes)];return q+s,t,True

def edit(row,idx):
    q=str(row['prompt']);d=str(row.get('domain','')).lower()
    if d=='math':
        e=numeric_edit(q,1 if idx%2==0 else -1)
        if e:return e,'value',False
        e=relation_edit(q)
        if e:return e,'relation',True
        return q+'\nAdditional constraint: solve under the same assumptions but with one boundary case included.','constraint',True
    if d=='code': return code_edit(q,idx)
    if d=='medical': return medical_edit(q,idx)
    if d=='science':
        e=numeric_edit(q,1 if idx%2==0 else -1)
        if e:return e,'value',False
        e=relation_edit(q)
        if e:return e,'relation',True
        suffix='\nAdditional condition: assume standard Earth surface conditions.' if idx%2==0 else '\nAdditional condition: reason about the nearest contrasting case before answering.'
        return q+suffix,'context',True
    return q+'\nApply one explicit counterfactual condition before answering.','constraint',True

def pid(seed_id,typ,edited):
    h=hashlib.sha256((seed_id+'\n'+typ+'\n'+edited).encode('utf-8')).hexdigest()[:16]
    return 'formal_'+h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='data/probes/formal_seed_pool.jsonl');ap.add_argument('--output',default='data/probes/formal_interventions_candidate.jsonl');ap.add_argument('--max-per-domain',type=int,default=400);ap.add_argument('--seed',type=int,default=20260923);args=ap.parse_args()
    inp=Path(args.input); rows=[json.loads(x) for x in inp.read_text(encoding='utf-8').splitlines() if x.strip()]
    rng=random.Random(args.seed);rng.shuffle(rows);count={};out=[]
    for i,r in enumerate(rows):
        d=str(r['domain']);
        if count.get(d,0)>=args.max_per_domain:continue
        qe,typ,review=edit(r,i)
        if not qe or qe.strip()==str(r['prompt']).strip():continue
        rec={'probe_id':pid(r['seed_id'],typ,qe),'domain':d,'type':typ,'changed_factor':typ,'base_query':r['prompt'],'edited_query':qe,'source_seed_id':r['seed_id'],'source_dataset':r.get('source_dataset'),'source_config':r.get('source_config'),'source_split':r.get('source_split'),'source_license':r.get('license'),'generation_rule':'formal_rules_v1','needs_review':bool(review),'review_status':'pending' if review else 'auto_pass_candidate','notes':'candidate pool; freeze only after review/audit'}
        out.append(rec);count[d]=count.get(d,0)+1
    out.sort(key=lambda x:(x['domain'],x['probe_id']))
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(p),'n':len(out),'by_domain':count,'needs_review':sum(r['needs_review'] for r in out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
