#!/usr/bin/env python3
"""Build ~1k deterministic, held-out intervention probes for the first mechanism test.

This helper operates on the disjoint pilot prompt split produced by
prepare_pilot_datasets.py. The transformations are intentionally simple and
fully auditable; they are a *pilot* pool, not the final TPAMI intervention set.
Every row records its intervention type and whether manual review is advised.

Final experiments should replace/augment these with the locked, contamination-
audited intervention taxonomy described in the paper.
"""
from __future__ import annotations
import argparse,json,re,hashlib
from pathlib import Path

NUM=re.compile(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)')
CHOICE=re.compile(r'(?m)^([A-D])\.\s+(.+)$')

def numeric_edit(q:str):
    m=NUM.search(q)
    if not m:return None
    raw=m.group(1)
    try:
        x=float(raw); y=x+1 if abs(x)<10 else x*1.1
        rep=str(int(round(y))) if raw.lstrip('-').isdigit() else f'{y:.3g}'
    except Exception:return None
    return q[:m.start(1)]+rep+q[m.end(1):]

def science_choice_relabel(q:str):
    ms=list(CHOICE.finditer(q))
    if len(ms)<2:return None
    # Swap the text of A/B while keeping labels fixed: exactly one answer-option
    # assignment changes, which is easy to audit and independent of an LLM.
    a,b=ms[0],ms[1]
    if a.group(1)!='A' or b.group(1)!='B':return None
    qa=q[:a.start(2)]+b.group(2)+q[a.end(2):]
    # Recompute B span after first replacement rather than use stale offsets.
    ms2=list(CHOICE.finditer(qa)); b2=ms2[1]
    return qa[:b2.start(2)]+a.group(2)+qa[b2.end(2):]

def edit(row,idx):
    q=str(row['prompt']); domain=str(row.get('domain','')).lower()
    if domain=='math':
        e=numeric_edit(q)
        return (e,'value',False) if e and e!=q else (q+'\nAssume the final numeric quantity is increased by one unit.','constraint',True)
    if domain=='code':
        suffix='\nAdditional constraint: do not use recursion.' if idx%2==0 else '\nAdditional constraint: use O(1) auxiliary memory when feasible.'
        return q+suffix,'constraint',True
    if domain=='medical':
        suffix='\nAdditional patient context: assume normal renal function.' if idx%2==0 else '\nAdditional patient context: assume no known drug allergies.'
        return q+suffix,'clinical_context',True
    if domain=='science':
        e=numeric_edit(q)
        if e and e!=q:return e,'value',False
        e=science_choice_relabel(q)
        if e and e!=q:return e,'choice_assignment',True
        return q+'\nAnswer under standard Earth conditions.','context',True
    return q+'\nGive the answer under one additional explicit constraint.','constraint',True

def stable_id(row,i):
    base=str(row.get('id',i)); h=hashlib.sha256(base.encode()).hexdigest()[:10]
    return f'pilot_{i:05d}_{h}'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/distillation/pilot_prompts_with_domains.jsonl'); ap.add_argument('--output',default='data/probes/interventions_pilot.jsonl'); ap.add_argument('--per-domain',type=int,default=250); args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    counts={}; out=[]
    for i,r in enumerate(rows):
        d=str(r.get('domain','Unknown')); key=d.lower()
        if counts.get(key,0)>=args.per_domain:continue
        qe,typ,review=edit(r,i)
        if qe==r['prompt']:continue
        rec={'probe_id':stable_id(r,i),'domain':d,'type':typ,'changed_factor':typ,'base_query':r['prompt'],'edited_query':qe,'source_id':r.get('id',''),'source':'deterministic_pilot_v1','notes':'pilot auto-generation; manual review advised' if review else 'pilot auto-generation','needs_review':review}
        out.append(rec); counts[key]=counts.get(key,0)+1
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(p),'n':len(out),'by_domain':counts,'manual_review_advised':sum(bool(r['needs_review']) for r in out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
