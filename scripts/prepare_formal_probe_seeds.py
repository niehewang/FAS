#!/usr/bin/env python3
"""Prepare source-disjoint held-out prompts used only to build the probe pool."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.data_governance import stable_text_id,format_mcq


def trow(x,kind,domain):
    if kind=='hendrycks_math': p=str(x.get('problem','')).strip(); meta={'answer':x.get('solution','')}
    elif kind=='humaneval': p=str(x.get('prompt','')).strip(); meta={'tests':x.get('test',''),'entry_point':x.get('entry_point','')}
    elif kind=='mbppplus': p=str(x.get('prompt') or x.get('text') or '').strip(); meta={'tests':x.get('test') or x.get('tests') or ''}
    elif kind=='pubmedqa':
        ctx=x.get('context',{}); cs=ctx.get('contexts',[]) if isinstance(ctx,dict) else []
        p=(str(x.get('question','')).strip()+'\nContext: '+' '.join(map(str,cs))).strip(); meta={'answer':x.get('final_decision','')}
    elif kind=='arc':
        c=x.get('choices',{}); labels=c.get('label',[]) if isinstance(c,dict) else []; texts=c.get('text',[]) if isinstance(c,dict) else []
        opts='\n'.join(f'{a}. {b}' for a,b in zip(labels,texts));p=(str(x.get('question','')).strip()+'\n'+opts).strip();meta={'answer':x.get('answerKey','')}
    else:raise ValueError(kind)
    if not p:return None
    return {'seed_id':stable_text_id(domain,p,prefix='seed'),'domain':domain,'prompt':p,'source_meta':meta}


def resolve_sha(repo,revision='main'):
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo,revision=revision).sha
    except Exception:return revision

def load_source(s):
    from datasets import load_dataset
    revision=resolve_sha(s['dataset'])
    configs=s.get('configs') or [s.get('config_name')]
    for cfg in configs:
        ds=load_dataset(s['dataset'],cfg,split=s['split'],revision=revision)
        for x in ds:yield x,cfg,revision


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/data/probe_sources.yaml');ap.add_argument('--output',default='data/probes/formal_seed_pool.jsonl');ap.add_argument('--manifest',default='data/governance/formal_probe_seed_manifest.json');args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'));rng=random.Random(int(cfg['sampling_seed']));per=int(cfg['per_domain']); by={}
    for key,s in cfg['sources'].items():
        rows=[]
        for x,sub,revision in load_source(s):
            r=trow(x,s['transform'],s['domain_label'])
            if r:r.update({'source_dataset':s['dataset'],'source_revision':revision,'source_config':sub,'source_split':s['split'],'license':s['expected_license']});rows.append(r)
        rng.shuffle(rows);by.setdefault(s['domain_label'],[]).extend(rows)
    out=[]
    for d,rows in by.items():
        # dedup by seed id and cap after combining multiple sources
        uniq={r['seed_id']:r for r in rows};arr=list(uniq.values());rng.shuffle(arr)
        if len(arr)<per:raise RuntimeError(f'{d}: only {len(arr)} seeds, need {per}')
        out.extend(arr[:per])
    out.sort(key=lambda r:(r['domain'],r['seed_id']))
    p=ROOT/args.output;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    import hashlib
    sources={}
    for r in out:
        k=(r['source_dataset'],str(r.get('source_config')),r['source_split'],r.get('source_revision'),r.get('license'))
        sources.setdefault(k,0);sources[k]+=1
    manifest={'config':args.config,'sampling_seed':int(cfg['sampling_seed']),'per_domain':per,'output':str(p.relative_to(ROOT)),'output_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sources':[{'dataset':k[0],'config':k[1] if k[1]!='None' else None,'split':k[2],'revision':k[3],'license':k[4],'retained_rows':n} for k,n in sorted(sources.items())]}
    mp=ROOT/args.manifest;mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(p.relative_to(ROOT)),'manifest':str(mp.relative_to(ROOT)),'n':len(out),'domains':{d:sum(r['domain']==d for r in out) for d in sorted(by)}},indent=2))
if __name__=='__main__':main()
