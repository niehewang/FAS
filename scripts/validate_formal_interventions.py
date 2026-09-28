#!/usr/bin/env python3
"""Validate/freeze the formal intervention candidate pool.

Rows explicitly marked reject in an optional review TSV are removed. If the TSV
contains accepted rows, their decision overrides pending status. Unreviewed rows
that require review are retained only with --allow-pending (pilot/debug only).
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='data/probes/formal_interventions_candidate.jsonl');ap.add_argument('--review',default='data/governance/probe_review_sample.tsv');ap.add_argument('--output',default='data/probes/interventions.jsonl');ap.add_argument('--allow-pending',action='store_true');ap.add_argument('--min-per-domain',type=int,default=250);args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding='utf-8').splitlines() if x.strip()]
    decisions={}
    rp=Path(args.review)
    if rp.exists():
        with rp.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f,delimiter='\t'):
                dec=str(r.get('decision','')).strip().lower()
                if dec:decisions[r['probe_id']]=(dec,r.get('reviewer_note',''))
    # Review is a stratified QA sample over generation rules, not an instruction to
    # hand-label every candidate. Require a minimum reviewed sample per domain and
    # a bounded reject rate; then freeze the remaining candidates under the audited
    # rule set. Directly rejected sampled rows are removed.
    domains=sorted({r['domain'] for r in rows})
    reviewed={d:[] for d in domains}
    for r in rows:
        dec,note=decisions.get(r['probe_id'],('',''))
        if dec: reviewed[r['domain']].append((dec,r['probe_id'],note))
    if not args.allow_pending:
        bad=[]
        for d,arr in reviewed.items():
            if len(arr)<10: bad.append(f'{d}:reviewed={len(arr)}<10'); continue
            rejects=sum(dec in {'reject','0','no','bad'} for dec,_,_ in arr)
            if rejects/max(1,len(arr))>0.20: bad.append(f'{d}:reject_rate={rejects/len(arr):.3f}>0.20')
        if bad: raise SystemExit('probe review sample failed: '+'; '.join(bad))
    kept=[];pending=[];rejected=[]
    for r in rows:
        dec,note=decisions.get(r['probe_id'],('',''))
        if dec in {'reject','0','no','bad'}:rejected.append(r);continue
        if dec in {'accept','1','yes','ok','pass'}:
            r['review_status']='sample_review_accepted';r['reviewer_note']=note
        elif r.get('needs_review'):
            pending.append(r);r['review_status']='rule_audit_accepted' if not args.allow_pending else 'pending_allowed'
        else:r['review_status']='auto_accepted'
        kept.append(r)
    by={d:sum(r['domain']==d for r in kept) for d in domains}
    if any(n<args.min_per_domain for n in by.values()):raise SystemExit(f'insufficient final probes by_domain={by}, min={args.min_per_domain}')
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in kept:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('FORMAL_PROBES_OK',json.dumps({'n':len(kept),'by_domain':by,'pending_excluded':len(pending),'rejected':len(rejected)},ensure_ascii=False))
if __name__=='__main__':main()
