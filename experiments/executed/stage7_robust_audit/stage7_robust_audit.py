#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, csv, itertools, json, math, os, re, sys, time
from pathlib import Path
import numpy as np

SEEDS=[23,47,71]
ANCESTORS=['Math','Code','Medical','Science']
KNOWN=['Math','Code','Medical']
TRUE_L1={'Math','Code'}
ENCODERS=[
    ('mpnet','sentence-transformers/all-mpnet-base-v2'),
    ('minilm','sentence-transformers/all-MiniLM-L6-v2'),
    ('bge','BAAI/bge-base-en-v1.5'),
]

def jdump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def delta_npz(z):
    b=np.asarray(z['base_embeddings'],dtype=float); e=np.asarray(z['edited_embeddings'],dtype=float)
    if b.ndim==3: b=b.mean(axis=1)
    if e.ndim==3: e=e.mean(axis=1)
    return e-b

def conformal_p(cal,score):
    a=np.asarray(cal,dtype=float).reshape(-1)
    return float((1+np.sum(a>=float(score)))/(len(a)+1))

def load_score_csv(p,col='score'):
    with open(p,encoding='utf-8',newline='') as f:
        rr=list(csv.DictReader(f))
    return np.asarray([float(r[col]) for r in rr if str(r.get(col,'')).strip()],dtype=float)

def find_one(root,name):
    hits=sorted(Path(root).rglob(name),key=lambda p:(len(p.parts),str(p)))
    if not hits: raise FileNotFoundError(f'{name} under {root}')
    return hits[0]

def nnls_solve(A,y):
    from scipy.optimize import nnls
    c,r=nnls(np.asarray(A,float),np.asarray(y,float).reshape(-1))
    s=float(c.sum()); pi=c/s if s>0 else np.zeros_like(c)
    rel=float(np.linalg.norm(A@c-y)/(np.linalg.norm(y)+1e-12))
    return c,pi,rel

def f1(pred,true):
    pred=set(pred); true=set(true)
    if not pred and not true: return 1.0
    if not pred or not true: return 0.0
    tp=len(pred&true); prec=tp/len(pred); rec=tp/len(true)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def topk_names(pi,names,k=2):
    idx=np.argsort(-np.asarray(pi))[:k]
    return [names[int(i)] for i in idx]

def jaccard(a,b):
    a=set(a);b=set(b); u=a|b
    return 1.0 if not u else len(a&b)/len(u)

def spearman(a,b):
    from scipy.stats import spearmanr
    x=float(spearmanr(a,b).statistic)
    return 0.0 if not np.isfinite(x) else x

def auc_rank(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    wins=0.0
    for p in pos:
        wins += np.sum(p>neg)+0.5*np.sum(p==neg)
    return float(wins/(len(pos)*len(neg)))

def select_rows(stage3_seed):
    sel=read_json(stage3_seed/'selections/active_B32.json')['selected']
    union=rows_jsonl(stage3_seed/'selections/union_probes.jsonl')
    by={int(r['global_probe_index']):r for r in union}
    miss=[i for i in sel if int(i) not in by]
    if miss: raise RuntimeError(f'active_B32 global indices absent from union_probes: {miss[:5]}')
    return [by[int(i)] for i in sel], [int(i) for i in sel]

def load_stage3_global_norms(stage3,seed,base_full):
    norms={}
    for anc in ANCESTORS:
        p=stage3/f's{seed}'/'banks'/f'{anc.lower()}.npz'
        if not p.exists(): raise FileNotFoundError(p)
        z=np.load(p,allow_pickle=False)
        f=delta_npz(z)-base_full
        norms[anc]=float(np.linalg.norm(f.reshape(-1)))+1e-12
    return norms

def generate_seed(seed, probes, plan, project, stage3, run_seed):
    rawp=run_seed/'raw_outputs.json'
    required=['base']+ANCESTORS+['l1']+[x['id'] for group in ['cal_open_candidates','test_known','test_unknown'] for x in plan[group]]
    if rawp.exists():
        obj=read_json(rawp)
        if all(k in obj.get('outputs',{}) for k in required):
            print(f'SKIP raw generation seed {seed}: complete',flush=True); return obj
    print(f'GENERATE raw deterministic texts seed {seed}',flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model_id='Qwen/Qwen3-4B-Base'
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True)
    adapter_paths={a:project/'runs'/'experts'/f's{seed}'/a.lower()/'adapter' for a in ANCESTORS}
    for a,p in adapter_paths.items():
        if not p.exists(): raise FileNotFoundError(p)
    model=PeftModel.from_pretrained(base,str(adapter_paths['Math']),adapter_name='Math',is_trainable=False)
    for a in ['Code','Medical','Science']:
        model.load_adapter(str(adapter_paths[a]),adapter_name=a,is_trainable=False)
    model.eval()
    device=next(model.parameters()).device
    qs=[r['base_query'] for r in probes]; qe=[r['edited_query'] for r in probes]
    @torch.inference_mode()
    def generate_texts(prompts,batch=4):
        out=[]
        for st in range(0,len(prompts),batch):
            pp=prompts[st:st+batch]
            x=tok(pp,return_tensors='pt',padding=True,truncation=True).to(device)
            n=x['input_ids'].shape[1]
            y=model.generate(**x,max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)
            for j in range(y.shape[0]):
                out.append(tok.decode(y[j,n:],skip_special_tokens=True).strip())
        return out
    outputs={}
    with model.disable_adapter():
        outputs['base']={'base':generate_texts(qs),'edit':generate_texts(qe)}
    for a in ANCESTORS:
        model.set_adapter(a); outputs[a]={'base':generate_texts(qs),'edit':generate_texts(qe)}
    def weighted(spec):
        name='tmp_'+re.sub(r'[^A-Za-z0-9_]','_',spec['id'])
        pars=list(spec['parents']); w=[float(spec['weights'][p]) for p in pars]
        model.add_weighted_adapter(pars,w,name,combination_type='cat')
        model.set_adapter(name)
        val={'base':generate_texts(qs),'edit':generate_texts(qe)}
        model.set_adapter('Math')
        if hasattr(model,'delete_adapter'): model.delete_adapter(name)
        return val
    l1={'id':'l1','parents':['Math','Code'],'weights':{'Math':0.5,'Code':0.5}}
    outputs['l1']=weighted(l1)
    for group in ['cal_open_candidates','test_known','test_unknown']:
        for spec in plan[group]: outputs[spec['id']]=weighted(spec)
    obj={'seed':seed,'probe_ids':[r['probe_id'] for r in probes],'global_probe_indices':[int(r['global_probe_index']) for r in probes],
         'domains':[r['domain'] for r in probes],'outputs':outputs}
    jdump(rawp,obj)
    del model,base
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return obj

def encode_seed(seed,raw,run_seed):
    names=list(raw['outputs'].keys()); emb={}
    for tag,enc_id in ENCODERS:
        p=run_seed/f'embeddings_{tag}.npz'
        if p.exists():
            z=np.load(p,allow_pickle=False)
            if all(f'{n}__base' in z and f'{n}__edit' in z for n in names):
                print(f'SKIP encoding {tag} seed {seed}',flush=True)
                emb[tag]={n:(np.asarray(z[f'{n}__base']),np.asarray(z[f'{n}__edit'])) for n in names}; continue
        print(f'ENCODE seed {seed} with {enc_id}',flush=True)
        from sentence_transformers import SentenceTransformer
        enc=SentenceTransformer(enc_id)
        save={}
        for n in names:
            b=enc.encode(raw['outputs'][n]['base'],normalize_embeddings=True,convert_to_numpy=True,batch_size=32,show_progress_bar=False)
            e=enc.encode(raw['outputs'][n]['edit'],normalize_embeddings=True,convert_to_numpy=True,batch_size=32,show_progress_bar=False)
            save[f'{n}__base']=np.asarray(b,dtype=np.float32); save[f'{n}__edit']=np.asarray(e,dtype=np.float32)
        np.savez_compressed(p,**save)
        emb[tag]={n:(save[f'{n}__base'],save[f'{n}__edit']) for n in names}
        del enc
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception: pass
    return emb

def fields_from_emb(embtag,norms=None,names=ANCESTORS):
    bb,be=embtag['base']; db=be-bb
    fields=[]
    for a in names:
        b,e=embtag[a]; f=(e-b)-db
        den=(norms[a] if norms else float(np.linalg.norm(f.reshape(-1)))+1e-12)
        fields.append((f/den).reshape(-1))
    A=np.stack(fields,axis=1)
    return A,db

def target_y(embtag,name):
    bb,be=embtag['base']; tb,te=embtag[name]
    return ((te-tb)-(be-bb)).reshape(-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',required=True); ap.add_argument('--stage3',required=True); ap.add_argument('--stage4',required=True); ap.add_argument('--run-root',required=True); ap.add_argument('--plan',required=True); ap.add_argument('--freeze',required=True)
    args=ap.parse_args()
    project=Path(args.project); stage3=Path(args.stage3); stage4=Path(args.stage4); run=Path(args.run_root); run.mkdir(parents=True,exist_ok=True)
    plan=read_json(args.plan); freeze=read_json(args.freeze)
    enc_rows=[]; open_rows=[]; neg_rows=[]; per_seed={}
    all_open_y=[]; all_open_s=[]
    pair_collapse=0; zero_false=0
    for seed in SEEDS:
        print(f'=== seed {seed} ===',flush=True)
        s3=stage3/f's{seed}'; rs=run/f's{seed}'; rs.mkdir(parents=True,exist_ok=True)
        probes,sel=select_rows(s3)
        raw=generate_seed(seed,probes,plan['seeds'][str(seed)],project,stage3,rs)
        emb=encode_seed(seed,raw,rs)
        # C6 sensitivity: local B32 normalization in each encoder, L1 only.
        enc_res={}
        for tag,_ in ENCODERS:
            A,_=fields_from_emb(emb[tag],norms=None)
            y=target_y(emb[tag],'l1')
            c,pi,rel=nnls_solve(A,y)
            supp=topk_names(pi,ANCESTORS,2)
            row={'seed':seed,'encoder':tag,'pi':[float(x) for x in pi],'top2_support':supp,'parent_f1':f1(supp,TRUE_L1),'relative_residual':rel}
            enc_res[tag]=row
        prim=enc_res['mpnet']
        for tag,_ in ENCODERS:
            r=enc_res[tag]
            r['support_jaccard_vs_primary']=1.0 if tag=='mpnet' else jaccard(r['top2_support'],prim['top2_support'])
            r['coordinate_spearman_vs_primary']=1.0 if tag=='mpnet' else spearman(r['pi'],prim['pi'])
            enc_rows.append(r)
        # C8 primary geometry with Stage3 global norms.
        bz=np.load(stage3/'shared/base_full.npz',allow_pickle=False); base_full=delta_npz(bz)
        norms=load_stage3_global_norms(stage3,seed,base_full)
        A_full,_=fields_from_emb(emb['mpnet'],norms=norms,names=ANCESTORS)
        A_known=A_full[:,:3]
        sigp=find_one(stage4/f's{seed}','signal_scores.csv'); openp=find_one(stage4/f's{seed}','open_scores.csv'); taup=find_one(stage4/f's{seed}','support_threshold.json')
        sigcal=load_score_csv(sigp); full_open_cal=load_score_csv(openp)
        tau=float(read_json(taup)['selected']['threshold'])
        # Stage7 restricted-bank open calibration: first 20 signal-eligible of 24 pre-frozen known targets.
        cal=[]
        for spec in plan['seeds'][str(seed)]['cal_open_candidates']:
            y=target_y(emb['mpnet'],spec['id']); sig=float(np.linalg.norm(y)); ps=conformal_p(sigcal,sig)
            _,pi,rel=nnls_solve(A_known,y)
            if ps<0.05: cal.append({'id':spec['id'],'residual':rel,'signal_p':ps})
        if len(cal)<20: raise RuntimeError(f'seed {seed}: only {len(cal)} signal-eligible Stage7 cal-open targets; need 20')
        cal=cal[:20]; open_cal=np.asarray([x['residual'] for x in cal])
        seed_open=[]
        for rolekey,isunk in [('test_known',0),('test_unknown',1)]:
            for spec in plan['seeds'][str(seed)][rolekey]:
                y=target_y(emb['mpnet'],spec['id']); sig=float(np.linalg.norm(y)); ps=conformal_p(sigcal,sig)
                _,pi,rel=nnls_solve(A_known,y)
                po=None; state='Low-Signal'
                if ps<0.05:
                    po=conformal_p(open_cal,rel); state='Bank-Insufficient' if po<0.05 else 'Decomposable'
                row={'seed':seed,'target_id':spec['id'],'is_unknown':isunk,'role':rolekey,'signal_score':sig,'signal_p':ps,'relative_residual':rel,'open_p':po,'state':state,'pi_known':[float(x) for x in pi], 'parents':spec['parents'],'weights':spec['weights']}
                open_rows.append(row); seed_open.append(row)
                if ps<0.05:
                    all_open_y.append(isunk); all_open_s.append(rel)
        # Negative: L1 original and within-domain edited-pair shuffle.
        y0=target_y(emb['mpnet'],'l1'); _,pi0,rel0=nnls_solve(A_full,y0); sig0=float(np.linalg.norm(y0)); ps0=conformal_p(sigcal,sig0); po0=conformal_p(full_open_cal,rel0) if ps0<0.05 else None
        state0='Low-Signal' if ps0>=0.05 else ('Bank-Insufficient' if po0<0.05 else 'Decomposable')
        pred0=[ANCESTORS[i] for i,x in enumerate(pi0) if x>=tau]; f10=f1(pred0,TRUE_L1)
        bbase,ebase=emb['mpnet']['base']; bt,et=emb['mpnet']['l1']; domains=raw['domains']; perm=np.arange(len(domains))
        for d in sorted(set(domains)):
            ids=np.where(np.asarray(domains)==d)[0]
            perm[ids]=np.roll(ids,1)
        ysh=((et[perm]-bt)-(ebase[perm]-bbase)).reshape(-1)
        _,pish,relsh=nnls_solve(A_full,ysh); sigsh=float(np.linalg.norm(ysh)); pssh=conformal_p(sigcal,sigsh); posh=conformal_p(full_open_cal,relsh) if pssh<0.05 else None
        statesh='Low-Signal' if pssh>=0.05 else ('Bank-Insufficient' if posh<0.05 else 'Decomposable')
        predsh=[ANCESTORS[i] for i,x in enumerate(pish) if x>=tau]; f1sh=f1(predsh,TRUE_L1)
        collapsed=(statesh!='Decomposable') or (f1sh<=f10-0.25)
        pair_collapse += int(collapsed)
        # Exact nonidentity label permutation diagnostic.
        label_f1=[]
        for perm_names in itertools.permutations(ANCESTORS):
            if list(perm_names)==ANCESTORS: continue
            lab=[perm_names[i] for i,x in enumerate(pi0) if x>=tau]
            label_f1.append(f1(lab,TRUE_L1))
        # Exact zero-signal sanity.
        pzero=conformal_p(sigcal,0.0); zfalse=int(pzero<0.05); zero_false+=zfalse
        nr={'seed':seed,'original':{'state':state0,'parent_f1':f10,'relative_residual':rel0,'signal_p':ps0,'open_p':po0,'pi':[float(x) for x in pi0]},
            'pair_shuffle':{'state':statesh,'parent_f1':f1sh,'relative_residual':relsh,'signal_p':pssh,'open_p':posh,'pi':[float(x) for x in pish],'collapsed':bool(collapsed)},
            'label_permutation_mean_parent_f1':float(np.mean(label_f1)),'base_zero_signal_p':pzero,'base_zero_false_signal':bool(zfalse)}
        neg_rows.append(nr)
        per_seed[str(seed)]={'encoder':enc_res,'restricted_cal_open_n':20,'restricted_cal_open':cal,'negative':nr}
    # Aggregate C6
    alt=[r for r in enc_rows if r['encoder']!='mpnet']
    mean_j=float(np.mean([r['support_jaccard_vs_primary'] for r in alt])); mean_rho=float(np.mean([r['coordinate_spearman_vs_primary'] for r in alt]))
    alt_f1={e:float(np.mean([r['parent_f1'] for r in enc_rows if r['encoder']==e])) for e in ['minilm','bge']}
    c6=bool(mean_j>=0.55 and mean_rho>=0.50 and all(v>=0.67 for v in alt_f1.values()))
    # Aggregate C8
    eligible_known=[r for r in open_rows if r['is_unknown']==0 and r['signal_p']<0.05]; eligible_unknown=[r for r in open_rows if r['is_unknown']==1 and r['signal_p']<0.05]
    false_bank=float(np.mean([r['state']=='Bank-Insufficient' for r in eligible_known])) if eligible_known else float('nan')
    unk_det=float(np.mean([r['state']=='Bank-Insufficient' for r in eligible_unknown])) if eligible_unknown else float('nan')
    auc=auc_rank(all_open_y,all_open_s)
    base_zero_rate=zero_false/len(SEEDS)
    c8=bool(np.isfinite(auc) and auc>=0.75 and false_bank<=0.15 and unk_det>=0.67 and pair_collapse>=2 and base_zero_rate<=0.0)
    aggregate={
      'protocol':'FAS_STAGE7_ROBUST_AUDIT_V1','seeds':SEEDS,
      'C6_encoder_sensitivity':{'mean_alt_support_jaccard_vs_primary':mean_j,'mean_alt_coordinate_spearman_vs_primary':mean_rho,'alt_encoder_mean_true_parent_f1':alt_f1,'gate_pass':c6,
        'claim_if_pass':'C6 supported for tested sentence encoders: ancestry support/ranking conclusions are qualitatively stable. Finite-sample calibration remains specific to the primary evaluator.',
        'claim_if_fail':'C6 downgraded: evaluator sensitivity is material; paper claims are restricted to the primary MPNet evaluator.'},
      'C8_selective_open_set':{'pooled_signal_eligible_known_n':len(eligible_known),'pooled_signal_eligible_unknown_n':len(eligible_unknown),'pooled_open_auroc':auc,'pooled_known_false_bank_insufficient_rate':false_bank,'pooled_unknown_bank_insufficient_rate':unk_det,'probe_pair_shuffle_collapse_seeds':pair_collapse,'base_zero_false_signal_rate':base_zero_rate,'gate_pass':c8,
        'claim_if_pass':'C8 supported in the frozen restricted-bank G0 setting: selective inference abstains on withheld-ancestor targets while controlling known-target false bank-insufficiency, and negative controls collapse.',
        'claim_if_fail':'C8 downgraded: retain low-signal/bank-insufficiency gates as safeguards, without a broad open-set claim.'},
      'stage7_gate_pass':bool(c6 and c8),
      'guardrails':['No alternative-encoder conformal transfer claim.','Science withheld from restricted bank/cal-open/known-test construction.','Stage5/C3 remains closed regardless of Stage7.']
    }
    jdump(run/'aggregate_stage7.json',aggregate); jdump(run/'per_seed_stage7.json',per_seed); jdump(run/'negative_controls.json',neg_rows)
    # CSV exports
    with open(run/'encoder_sensitivity_rows.csv','w',encoding='utf-8',newline='') as f:
        fields=['seed','encoder','parent_f1','relative_residual','support_jaccard_vs_primary','coordinate_spearman_vs_primary','top2_support','pi']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in enc_rows: w.writerow({**r,'top2_support':json.dumps(r['top2_support']),'pi':json.dumps(r['pi'])})
    with open(run/'open_set_rows.csv','w',encoding='utf-8',newline='') as f:
        fields=['seed','target_id','role','is_unknown','signal_score','signal_p','relative_residual','open_p','state','parents','weights','pi_known'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in open_rows: w.writerow({**r,'parents':json.dumps(r['parents']),'weights':json.dumps(r['weights']),'pi_known':json.dumps(r['pi_known'])})
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__': main()
