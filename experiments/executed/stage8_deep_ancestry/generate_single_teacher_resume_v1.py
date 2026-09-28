#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,gc
from pathlib import Path
from jsonl_utils_v1 import strict_read_jsonl,recover_generated_jsonl_tail

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prompts',required=True);ap.add_argument('--model',required=True);ap.add_argument('--adapter');ap.add_argument('--output',required=True);ap.add_argument('--max-new-tokens',type=int,default=256);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--load-in-4bit',action='store_true');ap.add_argument('--seed',type=int,default=20260927);a=ap.parse_args()
 rows=strict_read_jsonl(a.prompts); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); old,repaired=recover_generated_jsonl_tail(out); done={r['id']:r for r in old}; todo=[r for r in rows if r['id'] not in done]
 print(f'generate model={a.model} total={len(rows)} done={len(done)} todo={len(todo)} int4={a.load_in_4bit}')
 if todo:
  import torch
  from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
  from peft import PeftModel
  tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
  quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True) if a.load_in_4bit else None
  model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=True,low_cpu_mem_usage=True)
  if a.adapter:model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
  model.eval()
  with out.open('a',encoding='utf-8') as f:
   for st in range(0,len(todo),a.batch_size):
    b=todo[st:st+a.batch_size]; txt=[r['prompt'] for r in b]; x=tok(txt,return_tensors='pt',padding=True,truncation=True,max_length=4096).to(model.device); il=x['input_ids'].shape[1]
    torch.manual_seed(a.seed+st)
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
    oo=tok.batch_decode(y[:,il:],skip_special_tokens=True)
    for r,o in zip(b,oo):
     q=dict(r);q['response']=o.strip();q['teacher_model']=a.model;q['teacher_int4']=bool(a.load_in_4bit);f.write(json.dumps(q,ensure_ascii=False)+'\n');done[q['id']]=q
    f.flush();os.fsync(f.fileno()); print(min(st+a.batch_size,len(todo)),'/',len(todo))
  del model,tok;gc.collect();torch.cuda.empty_cache()
 merged=[done[r['id']] for r in rows]
 tmp=out.with_name(out.name+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for r in merged:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,out)
 meta={'n':len(merged),'model':a.model,'adapter':a.adapter or '', 'load_in_4bit':bool(a.load_in_4bit),'max_new_tokens':a.max_new_tokens,'decode':'greedy'}
 out.with_suffix(out.suffix+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
