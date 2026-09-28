#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path

def norm_text(s): return re.sub(r'\s+',' ',str(s).strip().lower())
def last_number(s):
    t=str(s).replace(',', '')
    fs=re.findall(r'([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)',t)
    if fs:
        try:
            a,b=map(float,fs[-1]);
            if b!=0:return a/b
        except Exception:pass
    lf=re.findall(r'\\(?:d|t)?frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}',t)
    if lf:
        try:
            a,b=map(float,lf[-1]);
            if b!=0:return a/b
        except Exception:pass
    xs=re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',t)
    if not xs:return None
    try:return float(xs[-1])
    except:return None
def choice_token(s):
    t=norm_text(s)
    for w in ('yes','no','maybe'):
        if re.search(rf'\b{w}\b',t):return w
    m=re.search(r'(^|\s|\()([a-e])([\s\).,:]|$)',t); return m.group(2) if m else ''
def score(pred,ans,metric,tol=1e-6):
    metric=metric.lower()
    if metric=='numeric':
        p=last_number(pred); q=last_number(ans); return float(p is not None and q is not None and abs(p-q)<=tol*max(1.0,abs(q)))
    if metric=='choice': return float(choice_token(pred)==choice_token(ans) and choice_token(ans)!='')
    if metric=='exact': return float(norm_text(pred)==norm_text(ans))
    if metric=='contains': return float(norm_text(ans) in norm_text(pred))
    raise ValueError(metric)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--adapter-root',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); ap.add_argument('--details-dir',required=True); ap.add_argument('--max-new-tokens',type=int,default=96); ap.add_argument('--batch-size',type=int,default=8); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.data).read_text(encoding='utf-8').splitlines() if x.strip()]; plan=list(csv.DictReader(open(a.plan,encoding='utf-8',newline='')))
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(a.base,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(a.base,quantization_config=q,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True); base.eval(); base.config.use_cache=True
    def generate(model,prompts):
        out=[]
        for i in range(0,len(prompts),a.batch_size):
            pp=prompts[i:i+a.batch_size]; z=tok(pp,return_tensors='pt',padding=True,truncation=True,max_length=1536).to(model.device)
            with torch.inference_mode(): g=model.generate(**z,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            n=z['input_ids'].shape[1]
            out.extend(tok.decode(x[n:],skip_special_tokens=True).strip() for x in g)
        return out
    prompts=[str(r['prompt']) for r in rows]; details=Path(a.details_dir);details.mkdir(parents=True,exist_ok=True); allsum=[]
    def eval_preds(label,subset,preds):
        sums=defaultdict(float);cnt=defaultdict(int); raw=[]
        for r,p in zip(rows,preds):
            d=str(r['domain']); sc=score(p,str(r['answer']),str(r['metric'])); sums[d]+=sc;cnt[d]+=1; raw.append({'id':r.get('id',''),'domain':d,'metric':r['metric'],'answer':r['answer'],'prediction':p,'score':sc})
        for d in sorted(cnt): allsum.append({'calibration_id':label,'subset':subset,'domain':d,'utility':sums[d]/cnt[d],'n':cnt[d]})
        with (details/f'{label}.predictions.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','domain','metric','answer','prediction','score']);w.writeheader();w.writerows(raw)
    eval_preds('empty','',generate(base,prompts))
    first=plan[0]['calibration_id']; m=PeftModel.from_pretrained(base,str(Path(a.adapter_root)/first),adapter_name=first);m.eval()
    for r in plan[1:]: m.load_adapter(str(Path(a.adapter_root)/r['calibration_id']),adapter_name=r['calibration_id'])
    for r in plan:
        cid=r['calibration_id'];m.set_adapter(cid);eval_preds(cid,r['subset'],generate(m,prompts));print('evaluated',cid)
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['calibration_id','subset','domain','utility','n']);w.writeheader();w.writerows(allsum)
    print(p)
if __name__=='__main__':main()
