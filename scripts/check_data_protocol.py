#!/usr/bin/env python3
"""Static pre-flight checks for formal training/probe data governance."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-config',default='configs/data/formal_sources.yaml')
    ap.add_argument('--probe-config',default='configs/data/probe_sources.yaml')
    args=ap.parse_args()
    tr=yaml.safe_load((ROOT/args.train_config).read_text(encoding='utf-8'))
    pr=yaml.safe_load((ROOT/args.probe_config).read_text(encoding='utf-8'))
    errors=[]
    if not tr.get('base_model_expected_license'): errors.append('missing base_model_expected_license')
    train_repos={v['dataset'] for v in tr['domains'].values()}
    probe_repos={v['dataset'] for v in pr['sources'].values()}
    overlap=train_repos & probe_repos
    if overlap: errors.append('training/probe repository overlap: '+', '.join(sorted(overlap)))
    raw=int(tr.get('raw_examples_per_domain',0)); final=int(tr.get('final_examples_per_domain',0))
    if raw <= final: errors.append(f'raw_examples_per_domain={raw} must exceed final_examples_per_domain={final}')
    if final < 1000: errors.append('final_examples_per_domain is unexpectedly small')
    train_domains={v['domain_label'] for v in tr['domains'].values()}
    probe_domains={v['domain_label'] for v in pr['sources'].values()}
    if train_domains != probe_domains:
        errors.append(f'domain mismatch training={sorted(train_domains)} probe={sorted(probe_domains)}')
    for role,items in [('training',tr['domains']),('probe',pr['sources'])]:
        for key,v in items.items():
            for req in ['dataset','split','expected_license','source_url','domain_label']:
                if not v.get(req): errors.append(f'{role}:{key} missing {req}')
    if int(pr.get('per_domain',0)) < 200: errors.append('probe seed pool should contain >=200 examples/domain')
    if errors:
        print('DATA_PROTOCOL_FAIL')
        for e in errors: print(' -',e)
        raise SystemExit(1)
    print('DATA_PROTOCOL_OK',f'train_domains={len(train_domains)} raw={raw} final={final} probe_per_domain={pr["per_domain"]}')
if __name__=='__main__': main()
