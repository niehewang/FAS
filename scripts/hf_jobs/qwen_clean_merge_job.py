#!/usr/bin/env python3
# Self-contained GPU feasibility/clean-merge job for FAS.
# This is intentionally separate from the paper's formal 54-run pipeline: it is a
# reproducible GPU gate that trains four siblings from a shared Qwen base and
# verifies black-box multi-parent recovery before launching expensive KD chains.
from __future__ import annotations
import argparse, copy, gc, json, math, os, random, re, time
from dataclasses import dataclass
import numpy as np

DOMAINS=("math","code","medical","science")

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parent_f1(pi, true, thr=.05):
    p=set(np.flatnonzero(np.asarray(pi)>=thr)); t=set(np.flatnonzero(np.asarray(true)>0))
    tp=len(p&t); pr=tp/max(1,len(p)); rc=tp/max(1,len(t)); return 2*pr*rc/max(pr+rc,1e-12)

def nnls_fas(A,y):
    from scipy.optimize import nnls
    g,_=nnls(A,y); fit=A@g; resid=np.linalg.norm(y-fit); yn=np.linalg.norm(y)
    rho=max(0.,min(1.,1-resid**2/(yn**2+1e-12))); pi=g/(g.sum()+1e-12)
    return {"gamma":g,"pi":pi,"rho":rho,"relative_residual":resid/(yn+1e-12)}

def logdet_select(blocks,budget,groups=None,balanced=True,ridge=1e-4):
    # blocks: list [d,K], Gram increment B^T B.
    K=blocks[0].shape[1]; H=np.eye(K)*ridge; chosen=[]
    if balanced and groups is not None:
        ug=sorted(set(groups)); base=budget//len(ug); rem=budget%len(ug)
        quota={g:base+(i<rem) for i,g in enumerate(ug)}; used={g:0 for g in ug}
    else: quota=used=None
    for _ in range(min(budget,len(blocks))):
        best=None; bg=-1e100
        sign0,ld0=np.linalg.slogdet(H)
        for i,B in enumerate(blocks):
            if i in chosen: continue
            if quota is not None and used[groups[i]]>=quota[groups[i]]: continue
            _,ld=np.linalg.slogdet(H+B.T@B); gain=ld-ld0
            if gain>bg: best=i;bg=gain
        if best is None: break
        chosen.append(best); H=H+blocks[best].T@blocks[best]
        if used is not None: used[groups[best]]+=1
    return chosen

def transform_training(row,domain):
    if domain=='math':
        p=str(row.get('problem','')).strip(); a=str(row.get('generated_solution','')).strip()
    elif domain=='code':
        msgs=row.get('messages') or []; u=[m.get('content','') for m in msgs if str(m.get('role','')).lower()=='user']; v=[m.get('content','') for m in msgs if str(m.get('role','')).lower()=='assistant']; p=(u[0] if u else row.get('query','')).strip(); a=(v[-1] if v else row.get('answer','')).strip()
    elif domain=='medical':
        p=str(row.get('question','')).strip(); ops=[row.get('opa',''),row.get('opb',''),row.get('opc',''),row.get('opd','')]; idx=int(row.get('cop',0) or 0); a=f"Option {chr(65+max(0,min(3,idx)))}: {ops[max(0,min(3,idx))]}"; exp=str(row.get('exp') or '').strip(); a+=('\\nExplanation: '+exp) if exp else ''
    else:
        if str(row.get('subject','')).lower()!='science' or bool(row.get('pic_prob')) or bool(row.get('pic_choice')): return None
        p=str(row.get('problem','')).strip(); ch=row.get('choices') or []; idx=int(row.get('answer_idx',0) or 0); a=f"Option {chr(65+idx)}: {ch[idx] if idx<len(ch) else ''}"
    if not p or not a:return None
    return {'prompt':p,'response':a}

def load_train(domain,n,seed):
    from datasets import load_dataset
    if domain=='math': repo,conf,split='nvidia/OpenMathInstruct-2',None,'train_1M'
    elif domain=='code': repo,conf,split='m-a-p/Code-Feedback',None,'train'
    elif domain=='medical': repo,conf,split='openlifescienceai/medmcqa',None,'train'
    else: repo,conf,split='stemdataset/STEM',None,'train'
    ds=load_dataset(repo,conf,split=split,streaming=True).shuffle(seed=seed,buffer_size=10000)
    out=[]
    for r in ds:
        z=transform_training(r,domain)
        if z and len(z['prompt'])<10000 and len(z['response'])<12000: out.append(z)
        if len(out)>=n:break
    if len(out)<n: raise RuntimeError(f'{domain}: only {len(out)} samples')
    return out

def make_probe_pool(per_domain=64):
    from datasets import load_dataset
    pool=[]
    # Math: one config is enough for feasibility; formal project uses all configs.
    ds=load_dataset('EleutherAI/hendrycks_math','algebra',split='test')
    for i,r in enumerate(ds.select(range(min(per_domain,len(ds))))):
        q=str(r['problem']); m=re.search(r'(?<![A-Za-z])(-?\\d+(?:\\.\\d+)?)',q); qe=q
        if m:
            try: x=float(m.group(1)); rep=str(int(x+1)) if m.group(1).lstrip('-').isdigit() else f'{x+1:.4g}'; qe=q[:m.start(1)]+rep+q[m.end(1):]
            except: pass
        if qe==q: qe=q+'\\nAlso consider the nearest boundary case before answering.'
        pool.append(('math',q,qe))
    ds=load_dataset('openai/openai_humaneval','openai_humaneval',split='test')
    for r in list(ds)[:per_domain]:
        q=str(r['prompt']); pool.append(('code',q,q+'\\nAdditional constraint: handle empty input explicitly and preserve the function signature.'))
    ds=load_dataset('qiaojin/PubMedQA','pqa_labeled',split='train')
    for r in ds.select(range(min(per_domain,len(ds)))):
        q=str(r.get('question','')); pool.append(('medical',q,q+'\\nAdditional patient context: assume normal renal function and no known drug allergies.'))
    ds=load_dataset('allenai/ai2_arc','ARC-Challenge',split='test')
    for r in ds.select(range(min(per_domain,len(ds)))):
        q=str(r['question']); pool.append(('science',q,q+'\\nBefore answering, reason about the nearest contrasting case under standard Earth conditions.'))
    return pool

def prepare_sft_dataset(rows,tok,max_len):
    from datasets import Dataset
    ds=Dataset.from_list(rows)
    def f(ex):
        prefix=f"Question: {ex['prompt']}\\nAnswer:"
        text=prefix+' '+ex['response']
        z=tok(text,truncation=True,max_length=max_len,add_special_tokens=True)
        p=tok(prefix,truncation=True,max_length=max_len,add_special_tokens=True)['input_ids']; labels=list(z['input_ids']); labels[:min(len(p),len(labels))]=[-100]*min(len(p),len(labels)); z['labels']=labels; return z
    return ds.map(f,remove_columns=ds.column_names)

class Collator:
    def __init__(self,tok):self.tok=tok
    def __call__(self,features):
        import torch
        labels=[f.pop('labels') for f in features]; b=self.tok.pad(features,padding=True,return_tensors='pt'); L=b['input_ids'].shape[1]; b['labels']=torch.tensor([x+[-100]*(L-len(x)) for x in labels]); return b

def train_one(base_model,rows,outdir,seed,max_len=512,max_steps=-1):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    tok=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='right'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    m=AutoModelForCausalLM.from_pretrained(base_model,quantization_config=q,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True)
    m=prepare_model_for_kbit_training(m); m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM')); m.gradient_checkpointing_enable(); m.config.use_cache=False
    ds=prepare_sft_dataset(rows,tok,max_len)
    args=TrainingArguments(output_dir=outdir,num_train_epochs=1,max_steps=max_steps,per_device_train_batch_size=1,gradient_accumulation_steps=16,learning_rate=2e-4,warmup_ratio=.03,lr_scheduler_type='cosine',bf16=True,optim='paged_adamw_8bit',logging_steps=20,save_strategy='no',report_to='none',seed=seed,data_seed=seed,remove_unused_columns=False)
    Trainer(model=m,args=args,train_dataset=ds,data_collator=Collator(tok)).train(); m.save_pretrained(outdir); tok.save_pretrained(outdir)
    del m; gc.collect(); torch.cuda.empty_cache(); return tok

def load_multi_adapter(base_model,dirs):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
    from peft import PeftModel
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(base_model,quantization_config=q,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True)
    tok=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    peft=PeftModel.from_pretrained(base,dirs['math'],adapter_name='math')
    for d in DOMAINS[1:]: peft.load_adapter(dirs[d],adapter_name=d)
    peft.eval(); return peft,tok

def set_weighted(model,name,weights):
    if name in model.peft_config:
        model.delete_adapter(name)
    adapters=[d for d,w in zip(DOMAINS,weights) if abs(w)>1e-10]; ws=[float(w) for w in weights if abs(w)>1e-10]
    if len(adapters)==1:
        model.set_adapter(adapters[0]); return adapters[0]
    model.add_weighted_adapter(adapters,ws,name,combination_type='linear'); model.set_adapter(name); return name

def generate_outputs(model,tok,prompts,max_new=48,batch=8):
    import torch
    out=[]
    model.eval(); model.config.use_cache=True
    for i in range(0,len(prompts),batch):
        pp=[f"Question: {x}\\nAnswer:" for x in prompts[i:i+batch]]; z=tok(pp,return_tensors='pt',padding=True,truncation=True,max_length=768).to(model.device)
        with torch.inference_mode(): g=model.generate(**z,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id)
        input_len=z['input_ids'].shape[1]
        for row in g: out.append(tok.decode(row[input_len:],skip_special_tokens=True))
    return out

_ENC=None
def embed_texts(texts):
    global _ENC
    from sentence_transformers import SentenceTransformer
    if _ENC is None:
        _ENC=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
    return np.asarray(_ENC.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)

def load_one_adapter(base_model, adapter_dir):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer
    from peft import PeftModel
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(base_model,quantization_config=q,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True)
    tok=AutoTokenizer.from_pretrained(base_model,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    m=PeftModel.from_pretrained(base,adapter_dir,adapter_name='student');m.eval();return m,tok

def build_teacher_rows(model,tok,train_seen,seed,n_each=160):
    rows=[]
    for j,d in enumerate(DOMAINS[:3]):
        cand=load_train(d,max(n_each*2,n_each+32),seed+2000+j)
        prompts=[]
        for r in cand:
            if r['prompt'] in train_seen.get(d,set()): continue
            prompts.append(r['prompt'])
            if len(prompts)>=n_each: break
        if len(prompts)<n_each: raise RuntimeError(f'{d}: insufficient distill prompts')
        model.set_adapter(d); answers=generate_outputs(model,tok,prompts,max_new=64,batch=8)
        rows.extend({'prompt':p,'response':a,'teacher_domain':d} for p,a in zip(prompts,answers))
    random.Random(seed).shuffle(rows);return rows

def audit_student(student_base,adapter_dir,qb,qe,R0_4b,base_b_4b,A_act,A_rand,A_abs,ids_act,ids_rand,true_support):
    import torch
    m,tok=load_one_adapter(student_base,adapter_dir)
    union=sorted(set(ids_act)|set(ids_rand)); uq=[qb[i] for i in union]; ue=[qe[i] for i in union];pos={p:j for j,p in enumerate(union)}
    m.disable_adapter_layers(); b0=embed_texts(generate_outputs(m,tok,uq)); e0=embed_texts(generate_outputs(m,tok,ue)); r0=e0-b0
    m.enable_adapter_layers(); b=embed_texts(generate_outputs(m,tok,uq)); e=embed_texts(generate_outputs(m,tok,ue)); r=e-b
    def take(arr,ids): return np.stack([arr[pos[i]] for i in ids],0)
    ya=np.concatenate(take(r-r0,ids_act),0);yr=np.concatenate(take(r-r0,ids_rand),0);yabs=np.concatenate(take(b-b0,ids_act),0)
    da=nnls_fas(A_act,ya);dr=nnls_fas(A_rand,yr);dx=nnls_fas(A_abs,yabs)
    def pack(z):return {'pi':[float(x) for x in z['pi']],'parent_f1':parent_f1(z['pi'],true_support),'rho':z['rho'],'relative_residual':z['relative_residual']}
    out={'FAS':pack(da),'Random-IFRF':pack(dr),'Absolute Output + NNLS':pack(dx)}
    del m;gc.collect();torch.cuda.empty_cache();return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11);ap.add_argument('--base-model',default='Qwen/Qwen3-4B-Base');ap.add_argument('--student-model',default='Qwen/Qwen3-1.7B-Base');ap.add_argument('--train-n',type=int,default=3000);ap.add_argument('--probe-per-domain',type=int,default=64);ap.add_argument('--budget',type=int,default=64);ap.add_argument('--max-steps',type=int,default=-1);ap.add_argument('--distill-n-each',type=int,default=160);ap.add_argument('--stage',choices=['smoke','clean','clean_kd'],default='clean');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    import torch,tempfile
    seed_all(args.seed);t0=time.time();print(json.dumps({'event':'env','cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'seed':args.seed,'base':args.base_model}))
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    smoke=args.smoke or args.stage=='smoke'; train_n=64 if smoke else args.train_n; max_steps=3 if smoke else args.max_steps; probe_n=8 if smoke else args.probe_per_domain; budget=min(16 if smoke else args.budget,4*probe_n)
    root=tempfile.mkdtemp(prefix=f'fas_s{args.seed}_'); dirs={}; seen={}
    for j,d in enumerate(DOMAINS):
        print(json.dumps({'event':'load_data','domain':d,'n':train_n})); rows=load_train(d,train_n,args.seed+100*j); seen[d]={r['prompt'] for r in rows}; out=os.path.join(root,d); train_one(args.base_model,rows,out,args.seed*10+j,max_steps=max_steps);dirs[d]=out;print(json.dumps({'event':'trained','domain':d}))
    model,tok=load_multi_adapter(args.base_model,dirs); probes=make_probe_pool(probe_n); groups=[x[0] for x in probes]; qb=[x[1] for x in probes]; qe=[x[2] for x in probes]
    # Generate base and all ancestor outputs.
    model.disable_adapter_layers(); base_b=embed_texts(generate_outputs(model,tok,qb)); base_e=embed_texts(generate_outputs(model,tok,qe)); model.enable_adapter_layers(); R0=base_e-base_b
    fields=[]; abs_fields=[]
    for d in DOMAINS:
        model.set_adapter(d); eb=embed_texts(generate_outputs(model,tok,qb)); ee=embed_texts(generate_outputs(model,tok,qe)); fields.append((ee-eb)-R0); abs_fields.append(eb-base_b); print(json.dumps({'event':'bank','domain':d}))
    F=np.stack(fields,axis=1); Fabs=np.stack(abs_fields,axis=1) # P,K,D
    norms=np.sqrt(np.sum(F**2,axis=(0,2)))+1e-12; blocks=[F[p].T/norms[None,:] for p in range(len(probes))]
    abs_norms=np.sqrt(np.sum(Fabs**2,axis=(0,2)))+1e-12; abs_blocks=[Fabs[p].T/abs_norms[None,:] for p in range(len(probes))]
    ids=logdet_select(blocks,budget,groups,balanced=True); rng=np.random.default_rng(args.seed+999)
    # Balanced-random baseline uses the same per-domain quota as FAS.
    ids_rand=[]
    for g in sorted(set(groups)):
        cand=[i for i,x in enumerate(groups) if x==g]; qn=sum(groups[i]==g for i in ids); ids_rand.extend(rng.choice(cand,size=qn,replace=False).tolist())
    A=np.concatenate([blocks[p] for p in ids],axis=0); Ar=np.concatenate([blocks[p] for p in ids_rand],axis=0); Aabs=np.concatenate([abs_blocks[p] for p in ids],axis=0)
    targets=[[.5,.5,0,0],[.4,.35,.25,0],[.15,.25,.25,.35]]; rows=[]
    union=sorted(set(ids)|set(ids_rand)); pos={p:j for j,p in enumerate(union)}
    for ti,w in enumerate(targets):
        set_weighted(model,f'target{ti}',w); tb=embed_texts(generate_outputs(model,tok,[qb[p] for p in union])); te=embed_texts(generate_outputs(model,tok,[qe[p] for p in union])); rr=(te-tb)-R0[union]; aa=tb-base_b[union]
        def take(arr,ids2): return np.stack([arr[pos[i]] for i in ids2],0)
        fa=nnls_fas(A,np.concatenate(take(rr,ids),0)); fr=nnls_fas(Ar,np.concatenate(take(rr,ids_rand),0)); fx=nnls_fas(Aabs,np.concatenate(take(aa,ids),0))
        def pack(z):return {'pi':[float(x) for x in z['pi']],'parent_f1':parent_f1(z['pi'],w),'rho':z['rho'],'relative_residual':z['relative_residual']}
        rows.append({'weights':w,'FAS':pack(fa),'Random-IFRF':pack(fr),'Absolute Output + NNLS':pack(fx)})
    result={'event':'FAS_GPU_RESULT','seed':args.seed,'base_model':args.base_model,'train_n':train_n,'probe_n':len(probes),'budget':budget,'seconds':time.time()-t0,'clean_targets':rows,'clean_mean_parent_f1':{m:float(np.mean([r[m]['parent_f1'] for r in rows])) for m in ['FAS','Random-IFRF','Absolute Output + NNLS']},'clean_mean_residual':{m:float(np.mean([r[m]['relative_residual'] for r in rows])) for m in ['FAS','Random-IFRF','Absolute Output + NNLS']},'selected_by_domain':{g:sum(groups[i]==g for i in ids) for g in sorted(set(groups))}}
    if args.stage=='clean_kd' and not smoke:
        print(json.dumps({'event':'distill_generate','n_each':args.distill_n_each}))
        teacher_rows=build_teacher_rows(model,tok,seen,args.seed+5000,args.distill_n_each)
        del model; gc.collect(); torch.cuda.empty_cache()
        student_dir=os.path.join(root,'student_mixkd'); train_one(args.student_model,teacher_rows,student_dir,args.seed+777,max_len=512,max_steps=-1)
        result['multi_teacher_kd']=audit_student(args.student_model,student_dir,qb,qe,R0,base_b,A,Ar,Aabs,ids,ids_rand,[1,1,1,0])
        result['student_model']=args.student_model
    result['seconds']=time.time()-t0
    print('FAS_RESULT_JSON='+json.dumps(result,separators=(',',':')))

if __name__=='__main__': main()
