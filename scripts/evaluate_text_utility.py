#!/usr/bin/env python3
"""Evaluate a text model on a small machine-readable utility benchmark.

Input JSONL fields:
  id, domain, prompt, metric, answer
Supported metrics: numeric, choice, exact, contains.

The evaluator is deliberately simple and deterministic.  It is intended for
counterfactual/Shapley utility measurement where every subset model must be
scored by exactly the same rule.  Code-execution benchmarks are intentionally
not executed here; use a sandboxed external evaluator and import its domain
utility through collect_subset_utilities.py.
"""
from __future__ import annotations
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def last_number(s: str):
    xs=re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", str(s))
    if not xs: return None
    try: return float(xs[-1].replace(',',''))
    except Exception: return None


def choice_token(s: str) -> str:
    t=norm_text(s)
    # yes/no/maybe tasks
    for w in ('yes','no','maybe'):
        if re.search(rf'\b{w}\b',t): return w
    # multiple-choice tasks: prefer an early standalone A-E token
    m=re.search(r'(^|\s|\()([a-e])([\s\).,:]|$)', t)
    return m.group(2) if m else ''


def score_prediction(pred: str, answer: str, metric: str, tol: float=1e-6) -> float:
    metric=metric.lower()
    if metric=='numeric':
        p=last_number(pred); a=last_number(answer)
        if p is None or a is None: return 0.0
        return float(abs(p-a) <= tol*max(1.0,abs(a)))
    if metric=='choice': return float(choice_token(pred)==choice_token(answer) and choice_token(answer)!='')
    if metric=='exact': return float(norm_text(pred)==norm_text(answer))
    if metric=='contains': return float(norm_text(answer) in norm_text(pred))
    raise ValueError(f'unsupported metric={metric!r}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--raw-output')
    ap.add_argument('--max-new-tokens',type=int,default=256)
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--load-in-4bit',action='store_true')
    ap.add_argument('--numeric-tol',type=float,default=1e-6)
    args=ap.parse_args()

    rows=[json.loads(x) for x in Path(args.data).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows: raise SystemExit('empty evaluation JSONL')

    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=True)
    if args.adapter: model=PeftModel.from_pretrained(model,args.adapter)
    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    raw=[]; sums=defaultdict(float); counts=defaultdict(int)
    for i,r in enumerate(rows):
        q=str(r['prompt']); x=tok(q,return_tensors='pt').to(model.device)
        with torch.inference_mode(): y=model.generate(**x,max_new_tokens=args.max_new_tokens,do_sample=False)
        pred=tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        sc=score_prediction(pred,str(r['answer']),str(r['metric']),args.numeric_tol)
        dom=str(r.get('domain','overall'))
        sums[dom]+=sc; counts[dom]+=1
        raw.append({'id':r.get('id',i),'domain':dom,'metric':r['metric'],'answer':r['answer'],'prediction':pred,'score':sc})
        if (i+1)%50==0: print(f'{i+1}/{len(rows)}')
    by_domain={d:{'utility':sums[d]/counts[d],'n':counts[d]} for d in sorted(counts)}
    overall=sum(x['score'] for x in raw)/len(raw)
    obj={'model':args.model,'adapter':args.adapter or '', 'overall_utility':overall,'n':len(raw),'by_domain':by_domain,'data':args.data}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    rp=Path(args.raw_output) if args.raw_output else out.with_suffix('.predictions.csv')
    with rp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','domain','metric','answer','prediction','score']);w.writeheader();w.writerows(raw)
    print(json.dumps(obj,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
