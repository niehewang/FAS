#!/usr/bin/env python3
"""Generate response-distillation data from multiple local teachers.

Input JSONL rows require ``prompt``; optional ``domain`` is used by router
mode.  The implementation deliberately *streams teachers one at a time* so a
4-teacher experiment does not keep four 4B checkpoints resident on GPU.

Modes:
  mixture: deterministically assign each example to one teacher using exposure weights;
  router: choose teacher by ``domain_to_teacher``;
  all: query all teachers and emit one row per teacher (analysis only).
"""
from __future__ import annotations
import argparse, gc, hashlib, json, random
from pathlib import Path
import yaml


def weighted_choice(names, weights, key, seed):
    h=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16],16)
    r=random.Random(h)
    return r.choices(names,weights=weights,k=1)[0]


def load_model(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    base=spec.get('base') or spec['model']
    tok=AutoTokenizer.from_pretrained(base,trust_remote_code=True)
    dtype=getattr(torch,spec.get('dtype','bfloat16'))
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type=spec.get('bnb_4bit_quant_type','nf4'),bnb_4bit_compute_dtype=dtype) if spec.get('load_in_4bit',False) else None
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=dtype,quantization_config=quant,device_map='auto',trust_remote_code=True)
    if spec.get('adapter'):
        model=PeftModel.from_pretrained(model,spec['adapter'])
    model.eval()
    return model,tok


def unload_model(model, tok):
    del model, tok
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass


def generate(model,tok,prompt,gen,seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    x=tok(prompt,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        y=model.generate(**x,**gen)
    new=y[0,x['input_ids'].shape[1]:]
    return tok.decode(new,skip_special_tokens=True).strip()


def assignment_plan(rows, cfg):
    """Return list[(row_index, teacher_name)] without loading any model."""
    teachers={x['name']:x for x in cfg['teachers']}
    names=list(teachers)
    weights=[float(teachers[n].get('weight',1.0)) for n in names]
    domain_map=cfg.get('domain_to_teacher',{})
    mode=cfg.get('mode','mixture'); seed=int(cfg.get('seed',42))
    plan=[]
    for idx,row in enumerate(rows):
        if mode=='mixture':
            chosen=[weighted_choice(names,weights,row.get('id',idx),seed)]
        elif mode=='router':
            domain=row.get('domain','')
            if domain not in domain_map: raise KeyError(f'no teacher route for domain={domain!r}')
            chosen=[domain_map[domain]]
        elif mode=='all':
            chosen=names
        else:
            raise ValueError(mode)
        for name in chosen: plan.append((idx,name))
    return plan


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    teachers={x['name']:x for x in cfg['teachers']}
    mode=cfg.get('mode','mixture'); seed=int(cfg.get('seed',42)); gen=cfg.get('generation',{'max_new_tokens':512,'do_sample':False})
    rows=[json.loads(x) for x in Path(cfg['input_jsonl']).read_text(encoding='utf-8').splitlines() if x.strip()]
    plan=assignment_plan(rows,cfg)
    by_teacher={name:[] for name in teachers}
    for idx,name in plan:
        if name not in teachers: raise KeyError(f'assignment references unknown teacher {name!r}')
        by_teacher[name].append(idx)

    # Generate one teacher at a time and keep only strings in CPU memory.
    records=[]
    for name,spec in teachers.items():
        inds=by_teacher.get(name,[])
        if not inds: continue
        print(f'loading teacher {name}: {len(inds)} examples')
        model,tok=load_model(spec)
        for pos,idx in enumerate(inds):
            row=rows[idx]; prompt=row['prompt']
            response=generate(model,tok,prompt,gen,seed+idx)
            rec={**row,'teacher':name,'response':response,'distill_mode':mode,'source_index':idx}
            records.append((idx, name, rec))
            if (pos+1)%50==0: print(f'  {name}: {pos+1}/{len(inds)}')
        unload_model(model,tok)

    # Restore deterministic input order (and teacher order for mode=all).
    order={n:i for i,n in enumerate(teachers)}
    records.sort(key=lambda x:(x[0],order[x[1]]))
    out=Path(cfg['output_jsonl']); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for _,_,rec in records: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    summary={n:len(by_teacher[n]) for n in teachers}
    (out.with_suffix(out.suffix+'.meta.json')).write_text(json.dumps({'mode':mode,'seed':seed,'counts':summary,'n_outputs':len(records)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'counts':summary,'n_outputs':len(records)},ensure_ascii=False))
if __name__=='__main__': main()
