#!/usr/bin/env python3
"""Download/normalize the four controlled sibling-expert corpora.

No data are bundled in the repository. The script resolves immutable HF commit
SHAs, deterministically samples the same number of examples per domain, writes
canonical JSONL, and records provenance/token statistics. It does not use any
probe/evaluation examples.
"""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.data_governance import stable_text_id,format_mcq


def transform(row,kind,domain):
    if kind=='openmath2':
        p=str(row.get('problem','')).strip(); r=str(row.get('generated_solution','')).strip()
        sid=str(row.get('problem_source','openmath2'))
    elif kind=='code_feedback':
        msgs=row.get('messages') or []
        users=[m.get('content','') for m in msgs if str(m.get('role','')).lower()=='user']
        assists=[m.get('content','') for m in msgs if str(m.get('role','')).lower()=='assistant']
        p=(users[0] if users else row.get('query','')).strip(); r=(assists[-1] if assists else row.get('answer','')).strip(); sid=str(row.get('id','codefeedback'))
    elif kind=='medmcqa':
        choices=[row.get('opa',''),row.get('opb',''),row.get('opc',''),row.get('opd','')]
        cop=row.get('cop',0)
        try: idx=int(cop)
        except Exception: idx=0
        # MedMCQA uses 0-based correct option in the canonical HF conversion.
        p,ans=format_mcq(row.get('question',''),choices,idx)
        exp=str(row.get('exp') or '').strip(); r=ans+(('\nExplanation: '+exp) if exp else '')
        sid=str(row.get('id') or row.get('question','')[:64])
    elif kind=='stem_science_text':
        if str(row.get('subject','')).lower()!='science' or bool(row.get('pic_prob')) or bool(row.get('pic_choice')): return None
        p,r=format_mcq(row.get('problem',''),row.get('choices') or [],row.get('answer_idx',0)); sid=f"{row.get('grade','')}/{row.get('skill','')}"
    else: raise ValueError(kind)
    if not p or not r:return None
    return {'id':stable_text_id(domain,p,r,prefix=domain.lower()),'domain':domain,'prompt':p,'response':r,'source_record':sid}


def resolve_sha(repo,revision='main'):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo,revision=revision).sha
    except Exception:return revision


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/data/formal_sources.yaml');ap.add_argument('--output-dir',default='data/training/formal_raw');ap.add_argument('--manifest',default='data/governance/formal_training_manifest.json');ap.add_argument('--tokenizer',default=None);args=ap.parse_args()
    from datasets import load_dataset
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8')); n=int(cfg['raw_examples_per_domain']); seed=int(cfg['sampling_seed']); out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    tok=None
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.tokenizer or cfg['base_model'],trust_remote_code=True)
    except Exception: pass
    manifest={'config':args.config,'sampling_seed':seed,'raw_examples_per_domain':n,'final_examples_per_domain':int(cfg['final_examples_per_domain']),'domains':{}}
    for di,(key,s) in enumerate(cfg['domains'].items()):
        revision=resolve_sha(s['dataset']); ds=load_dataset(s['dataset'],s.get('config_name'),split=s['split'],revision=revision,streaming=True)
        ds=ds.shuffle(seed=seed+di,buffer_size=int(cfg.get('shuffle_buffer',20000)))
        rows=[];seen=set()
        for raw in ds:
            rec=transform(raw,s['transform'],s['domain_label'])
            if rec is None:continue
            if len(rec['prompt'])>int(cfg['max_prompt_chars']) or len(rec['response'])>int(cfg['max_response_chars']):continue
            if rec['id'] in seen:continue
            seen.add(rec['id']);rec.update({'source_dataset':s['dataset'],'source_revision':revision,'source_split':s['split'],'license':s['expected_license']});rows.append(rec)
            if len(rows)>=n:break
        if len(rows)<n:raise RuntimeError(f"{key}: only {len(rows)} usable rows, need {n}")
        path=out/f'{key}.jsonl'
        with path.open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        if tok:
            pt=sum(len(tok(r['prompt'],add_special_tokens=False)['input_ids']) for r in rows); rt=sum(len(tok(r['response'],add_special_tokens=False)['input_ids']) for r in rows)
        else: pt=rt=None
        manifest['domains'][key]={'dataset':s['dataset'],'revision':revision,'license':s['expected_license'],'n':len(rows),'prompt_tokens':pt,'response_tokens':rt,'path':str(path.relative_to(ROOT)),'source_url':s['source_url']}
        print(key,len(rows),revision)
    mp=ROOT/args.manifest;mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(mp)
if __name__=='__main__':main()
