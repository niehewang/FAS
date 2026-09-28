from __future__ import annotations
import hashlib, re
from collections import defaultdict
from typing import Iterable

WS=re.compile(r"\s+")
NONWORD=re.compile(r"[^\w]+",re.UNICODE)


def normalize_text(x:str)->str:
    x=str(x or '').lower().strip()
    x=NONWORD.sub(' ',x)
    return WS.sub(' ',x).strip()


def stable_text_id(*parts:str, prefix:str='ex')->str:
    s='\n'.join(normalize_text(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]}"


def word_ngrams(text:str,n:int=8)->set[str]:
    toks=normalize_text(text).split()
    if not toks:return set()
    if len(toks)<n:return {' '.join(toks)}
    return {' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)}


def contamination_candidates(reference_rows:Iterable[dict], n:int=8):
    ref=[]; inv=defaultdict(set)
    for j,r in enumerate(reference_rows):
        text=f"{r.get('base_query',r.get('prompt',''))}\n{r.get('edited_query','')}"
        ng=word_ngrams(text,n)
        ref.append((r,ng,normalize_text(text)))
        for g in ng: inv[g].add(j)
    return ref,inv


def max_overlap(text:str, ref, inv, n:int=8):
    norm=normalize_text(text); ng=word_ngrams(norm,n)
    if not norm:return {'exact':False,'jaccard':0.0,'containment':0.0,'ref_index':None}
    cand=set()
    for g in ng:cand.update(inv.get(g,()))
    best={'exact':False,'jaccard':0.0,'containment':0.0,'ref_index':None}
    for j in cand:
        _,rg,rnorm=ref[j]
        exact=(norm==rnorm)
        inter=len(ng & rg); union=len(ng | rg) or 1; denom=min(len(ng),len(rg)) or 1
        jac=inter/union; cont=inter/denom
        if exact or (jac,cont)>(best['jaccard'],best['containment']):
            best={'exact':exact,'jaccard':jac,'containment':cont,'ref_index':j}
    return best


def format_mcq(question, choices, answer_idx):
    labels='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    choices=[str(x) for x in (choices or [])]
    opts='\n'.join(f'{labels[i]}. {x}' for i,x in enumerate(choices))
    try: idx=int(answer_idx); label=labels[idx]; answer=choices[idx]
    except Exception: label=str(answer_idx); answer=''
    prompt=f"{str(question).strip()}\n{opts}".strip()
    response=f"{label}. {answer}".strip('. ')
    return prompt,response
