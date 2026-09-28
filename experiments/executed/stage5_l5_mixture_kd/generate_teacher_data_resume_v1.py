#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, random, gc
from pathlib import Path
TEACHERS=['Math','Code','Medical','Science']; WEIGHTS=[.4,.3,.2,.1]
def choose(pid,seed):
    h=int(hashlib.sha256(f'{seed}:{pid}'.encode()).hexdigest()[:16],16); return random.Random(h).choices(TEACHERS,weights=WEIGHTS,k=1)[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompts',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--base',default='Qwen/Qwen3-4B-Base'); ap.add_argument('--adapter-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--max-new-tokens',type=int,default=512); ap.add_argument('--batch-size',type=int,default=4); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.prompts).read_text(encoding='utf-8').splitlines() if x.strip()]
    for i,r in enumerate(rows): r['_idx']=i; r['teacher']=choose(r['id'],a.seed)
    work=Path(a.work_dir); work.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    for t in TEACHERS:
        assigned=[r for r in rows if r['teacher']==t]; p=work/f'{t.lower()}.jsonl'; done={}
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    q=json.loads(line); done[q['id']]=q
        todo=[r for r in assigned if r['id'] not in done]
        print(f'{t}: assigned={len(assigned)} done={len(done)} todo={len(todo)}')
        if todo:
            tok=AutoTokenizer.from_pretrained(a.base,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
            model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,device_map='auto',trust_remote_code=True,low_cpu_mem_usage=True)
            model=PeftModel.from_pretrained(model,str(Path(a.adapter_root)/t.lower()/'adapter'),is_trainable=False); model.eval()
            with p.open('a',encoding='utf-8') as f:
                for st in range(0,len(todo),a.batch_size):
                    b=todo[st:st+a.batch_size]; texts=[r['prompt'] for r in b]; x=tok(texts,return_tensors='pt',padding=True,truncation=True,max_length=4096).to(model.device); inlen=x['input_ids'].shape[1]
                    with torch.inference_mode(): y=model.generate(**x,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
                    outs=tok.batch_decode(y[:,inlen:],skip_special_tokens=True)
                    for r,o in zip(b,outs):
                        rec={k:v for k,v in r.items() if k!='_idx'}; rec['response']=o.strip(); rec['distill_mode']='mixture'; rec['genealogy_seed']=a.seed; f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush(); done[r['id']]=rec
                    print(f'{t}: {min(st+a.batch_size,len(todo))}/{len(todo)} new')
            del model,tok; gc.collect(); torch.cuda.empty_cache()
    merged=[]
    byid={}
    for t in TEACHERS:
        p=work/f'{t.lower()}.jsonl'
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip(): q=json.loads(line); byid[q['id']]=q
    missing=[r['id'] for r in rows if r['id'] not in byid]
    if missing: raise SystemExit(f'missing teacher outputs: {missing[:10]}')
    merged=[byid[r['id']] for r in rows]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in merged:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    counts={t:sum(r['teacher']==t for r in merged) for t in TEACHERS}; meta={'seed':a.seed,'n':len(merged),'counts':counts,'weights':WEIGHTS,'decode':'greedy','max_new_tokens':a.max_new_tokens}; (out.with_suffix(out.suffix+'.meta.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
