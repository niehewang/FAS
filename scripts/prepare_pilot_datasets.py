#!/usr/bin/env python3
"""Prepare small *pilot* sibling-expert corpora and disjoint KD prompts.

This is only for the first mechanism/sanity experiment, not the final TPAMI
training corpus. For each public benchmark *training* split, examples are
partitioned deterministically into an expert-training portion and a disjoint
pilot distillation-prompt portion. No benchmark test split is used.

Final experiments must lock exact revisions/licenses and use a contamination-
audited training corpus; this helper is intentionally a low-cost pilot path.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def dump(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(path,len(rows))

def split_rows3(rows, expert_fraction, distill_fraction):
    n=len(rows)
    if n<3:
        return rows, [], []
    a=max(1,min(n-2,int(round(n*expert_fraction))))
    b=max(a+1,min(n-1,a+int(round(n*distill_fraction))))
    return rows[:a], rows[a:b], rows[b:]

def gsm8k_final(answer):
    text=str(answer)
    return text.split('####')[-1].strip() if '####' in text else text.strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='data/training/pilot');ap.add_argument('--distill-output',default='data/distillation/pilot_prompts_with_domains.jsonl');ap.add_argument('--eval-output',default='data/evaluation/pilot_utility.jsonl');ap.add_argument('--max-total-per-domain',type=int,default=5000);ap.add_argument('--expert-fraction',type=float,default=.70);ap.add_argument('--distill-fraction',type=float,default=.15);args=ap.parse_args()
    from datasets import load_dataset
    out=Path(args.output_dir);n=args.max_total_per_domain;all_distill=[];all_eval=[]

    specs=[]
    ds=load_dataset('openai/gsm8k','main',split='train'); specs.append(('math','Math',[{'id':f'gsm8k_{i}','domain':'Math','prompt':x['question'],'response':x['answer']} for i,x in enumerate(ds.select(range(min(n,len(ds)))))]))
    ds=load_dataset('google-research-datasets/mbpp',split='train'); rows=[]
    for i,x in enumerate(ds.select(range(min(n,len(ds))))): rows.append({'id':f'mbpp_{i}','domain':'Code','prompt':x.get('text') or x.get('prompt') or '','response':x.get('code') or x.get('completion') or ''})
    specs.append(('code','Code',rows))
    ds=load_dataset('qiaojin/PubMedQA','pqa_labeled',split='train'); rows=[]
    for i,x in enumerate(ds.select(range(min(n,len(ds))))): rows.append({'id':f'pubmedqa_{i}','domain':'Medical','prompt':x.get('question',''),'response':str(x.get('long_answer') or x.get('answer') or x.get('final_decision') or ''),'label':str(x.get('final_decision') or x.get('answer') or '')})
    specs.append(('medical','Medical',rows))
    ds=load_dataset('allenai/ai2_arc','ARC-Challenge',split='train'); rows=[]
    for i,x in enumerate(ds.select(range(min(n,len(ds))))):
        choices=x.get('choices',{}); labels=choices.get('label',[]);texts=choices.get('text',[]);opts='\n'.join(f'{a}. {b}' for a,b in zip(labels,texts));key=str(x.get('answerKey',''))
        try: answer=f'{key}. {texts[labels.index(key)]}'
        except Exception: answer=key
        rows.append({'id':f'arc_{i}','domain':'Science','prompt':x['question']+'\n'+opts,'response':answer})
    specs.append(('science','Science',rows))

    for fn,domain,rows in specs:
        expert,distill,eval_rows=split_rows3(rows,args.expert_fraction,args.distill_fraction); dump(out/f'{fn}.jsonl',expert)
        for r in distill: all_distill.append({'id':r['id'],'domain':domain,'prompt':r['prompt']})
        # Cheap pilot utility excludes code execution.  Code remains part of ancestry
        # recovery, but exact Shapley pilot uses Math/Medical/Science where scoring
        # can be deterministic without executing untrusted generated programs.
        for r in eval_rows:
            if domain=='Math':
                all_eval.append({'id':r['id'],'domain':domain,'prompt':r['prompt']+'\nGive the final numeric answer.', 'metric':'numeric','answer':gsm8k_final(r['response'])})
            elif domain=='Medical':
                all_eval.append({'id':r['id'],'domain':domain,'prompt':r['prompt']+'\nAnswer only yes, no, or maybe.', 'metric':'choice','answer':str(r.get('label') or r['response']).split()[0]})
            elif domain=='Science':
                all_eval.append({'id':r['id'],'domain':domain,'prompt':r['prompt']+'\nAnswer with the option letter only.', 'metric':'choice','answer':str(r['response']).split('.')[0]})
    dump(Path(args.distill_output),all_distill)
    dump(Path(args.eval_output),all_eval)
    # Convenience mixture-KD prompts without domain routing metadata.
    dump(Path(args.distill_output).with_name('pilot_prompts.jsonl'),[{'id':r['id'],'prompt':r['prompt']} for r in all_distill])

if __name__=='__main__':main()
