#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np

def rows(path):
 o=[]
 for i,l in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
  if l.strip():
   r=json.loads(l);r.setdefault('global_probe_index',i);o.append(r)
 return o

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter');ap.add_argument('--probes',required=True);ap.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2');ap.add_argument('--max-new-tokens',type=int,default=96);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--load-in-4bit',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
 R=rows(a.probes);import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from sentence_transformers import SentenceTransformer
 from peft import PeftModel
 tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True) if a.load_in_4bit else None
 m=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,quantization_config=q,device_map='auto',trust_remote_code=True,low_cpu_mem_usage=True)
 if a.adapter:m=PeftModel.from_pretrained(m,a.adapter,is_trainable=False)
 m.eval();enc=SentenceTransformer(a.encoder,device='cuda' if torch.cuda.is_available() else 'cpu')
 def gen(ts):
  x=tok(ts,return_tensors='pt',padding=True,truncation=True,max_length=4096).to(m.device);il=x['input_ids'].shape[1]
  with torch.inference_mode():y=m.generate(**x,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
  return tok.batch_decode(y[:,il:],skip_special_tokens=True)
 b=[];e=[]
 for st in range(0,len(R),a.batch_size):
  z=R[st:st+a.batch_size];bo=gen([r.get('base_query') or r.get('q') or r.get('prompt') for r in z]);eo=gen([r.get('edited_query') or r.get('q_edit') for r in z]);b.extend(enc.encode(bo,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False));e.extend(enc.encode(eo,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False));print(min(st+a.batch_size,len(R)),'/',len(R))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,base_embeddings=np.asarray(b,np.float32),edited_embeddings=np.asarray(e,np.float32),probe_ids=np.asarray([str(r.get('probe_id',i)) for i,r in enumerate(R)]),global_probe_indices=np.asarray([int(r.get('global_probe_index',i)) for i,r in enumerate(R)],np.int64),model=np.asarray([a.model]),adapter=np.asarray([a.adapter or '']),encoder=np.asarray([a.encoder]),load_in_4bit=np.asarray([int(a.load_in_4bit)]))
 del m,enc,tok;gc.collect();torch.cuda.empty_cache();print(out)
if __name__=='__main__':main()
