#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--probes',required=True);ap.add_argument('--encoder',required=True);ap.add_argument('--base-output',required=True);ap.add_argument('--target-output',required=True);ap.add_argument('--max-new-tokens',type=int,default=96);ap.add_argument('--batch-size',type=int,default=4);a=ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    from peft import PeftModel
    rows=[json.loads(x) for x in Path(a.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=q,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True)
    m=PeftModel.from_pretrained(base,a.adapter);m.eval();enc=SentenceTransformer(a.encoder,device='cpu')
    def gen(prompts):
        out=[]
        for i in range(0,len(prompts),a.batch_size):
            pp=prompts[i:i+a.batch_size];z=tok(pp,return_tensors='pt',padding=True,truncation=True,max_length=1536).to(m.device)
            with torch.inference_mode():g=m.generate(**z,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            n=z['input_ids'].shape[1];out.extend(tok.decode(x[n:],skip_special_tokens=True).strip() for x in g)
        return out
    qb=[str(r.get('base_query') or r.get('q') or r.get('prompt')) for r in rows];qe=[str(r.get('edited_query') or r.get('q_edit')) for r in rows]
    def run(disabled):
        if disabled:m.disable_adapter_layers()
        else:m.enable_adapter_layers()
        b=gen(qb);e=gen(qe)
        return np.asarray(enc.encode(b,batch_size=64,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32),np.asarray(enc.encode(e,batch_size=64,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
    bb,be=run(True);tb,te=run(False); ids=np.asarray([str(r.get('probe_id','')) for r in rows]);gs=np.asarray([int(r.get('global_probe_index',i)) for i,r in enumerate(rows)],dtype=np.int64)
    for path,b,e,tag in [(a.base_output,bb,be,'base'),(a.target_output,tb,te,'target')]:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,base_embeddings=b[:,None,:],edited_embeddings=e[:,None,:],probe_ids=ids,global_probe_indices=gs,role=np.asarray([tag]))
        print(p)
if __name__=='__main__':main()
