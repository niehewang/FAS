#!/usr/bin/env python3
"""Remove exact/high-overlap training examples against frozen probe/eval text."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.data_governance import contamination_candidates,max_overlap

def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-dir',default='data/training/formal_raw');ap.add_argument('--references',default='data/probes/formal_seed_pool.jsonl');ap.add_argument('--output-dir',default='data/training/formal');ap.add_argument('--report',default='data/governance/contamination_report.json');ap.add_argument('--final-per-domain',type=int,default=3000);ap.add_argument('--ngram',type=int,default=8);ap.add_argument('--jaccard',type=float,default=.45);ap.add_argument('--containment',type=float,default=.70);args=ap.parse_args()
    refs=read_jsonl(ROOT/args.references); ref,inv=contamination_candidates(refs,args.ngram);outdir=ROOT/args.output_dir;outdir.mkdir(parents=True,exist_ok=True);report={'thresholds':{'ngram':args.ngram,'jaccard':args.jaccard,'containment':args.containment},'domains':{}}
    for src in sorted((ROOT/args.train_dir).glob('*.jsonl')):
        kept=[];flags=[]
        for r in read_jsonl(src):
            score=max_overlap(f"{r.get('prompt','')}\n{r.get('response','')}",ref,inv,args.ngram)
            bad=score['exact'] or score['jaccard']>=args.jaccard or score['containment']>=args.containment
            if bad:flags.append({'id':r.get('id'),'score':score})
            else:kept.append(r)
        clean_before_trim=len(kept); input_count=clean_before_trim+len(flags)
        if clean_before_trim<args.final_per_domain: raise RuntimeError(f'{src.stem}: only {clean_before_trim} clean rows remain; need {args.final_per_domain}')
        kept=kept[:args.final_per_domain]
        dst=outdir/src.name
        with dst.open('w',encoding='utf-8') as f:
            for r in kept:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        report['domains'][src.stem]={'input':input_count,'clean_before_trim':clean_before_trim,'kept':len(kept),'removed_contamination':len(flags),'trimmed_clean':clean_before_trim-len(kept),'examples':flags[:20]}
        print(src.stem,'kept',len(kept),'removed',len(flags))
    rp=ROOT/args.report;rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
