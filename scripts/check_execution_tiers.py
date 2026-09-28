#!/usr/bin/env python3
"""Preflight checks for the staged evidence policy."""
from __future__ import annotations
import argparse,csv
from pathlib import Path

def fail(msg): raise SystemExit('EXECUTION_TIER_FAIL: '+msg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest_tiered.csv');ap.add_argument('--max-core',type=int,default=24);args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')))
    core=[r for r in rows if r.get('evidence_tier')=='core']
    if not core: fail('no core rows')
    if len(core)>args.max_core: fail(f'core rows {len(core)} exceed budget {args.max_core}')
    sc={r['scenario'] for r in core}
    for need in ['L1-linear-2p','L2-ties-3p','L5-mixture-kd','L7-deep-ancestry','L8-open-set']:
        if need not in sc: fail(f'missing required core scenario {need}')
    if any(r['scenario'].startswith('L6-') for r in core): fail('Router-KD must remain conditional')
    if any(r.get('student')=='0p6b' for r in core): fail('0.6B cross-scale must remain conditional')
    if any(r['scenario'].startswith(('L3-','L4-')) for r in core): fail('intermediate postprocessing stress belongs to conditional tier')
    print(f'EXECUTION_TIER_OK core={len(core)} full={len(rows)}')
if __name__=='__main__': main()
