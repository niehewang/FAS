#!/usr/bin/env python3
"""Check planned or realized conformal calibration resolution.

Without --actual-* inputs, validates the protocol's planned sample counts.  When
actual score files are supplied (as at the end of the G0 calibration job), the
realized number of finite scores becomes authoritative.  With
--require-planned-count, realized counts must also meet the protocol's planned_n
so a paper cannot silently cite a larger calibration set than was actually
retained after selective filtering.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]

def _levels(obj,key):
    x=obj['alphas'][key]
    if isinstance(x,dict):
        return [float(v) for v in x.get('main',[])],[float(v) for v in x.get('sensitivity',[])],[float(v) for v in x.get('optional',[])]
    vals=[float(v) for v in x];return vals,[],[]

def count_scores(path,key='score'):
    p=Path(path); suf=p.suffix.lower()
    if suf in {'.csv','.tsv'}:
        delim='\t' if suf=='.tsv' else ','
        rows=list(csv.DictReader(p.open(encoding='utf-8',newline=''),delimiter=delim))
        vals=[r.get(key,'') for r in rows]
    elif suf=='.json':
        obj=json.loads(p.read_text(encoding='utf-8')); vals=obj.get(key,[]) if isinstance(obj,dict) else obj
    elif suf=='.npy': vals=np.load(p).reshape(-1)
    elif suf=='.npz':
        z=np.load(p)
        if key not in z: raise KeyError(f'{p}: missing {key!r}')
        vals=z[key].reshape(-1)
    else: raise ValueError(f'unsupported actual calibration format: {p}')
    good=[]
    for v in vals:
        try:
            x=float(v)
            if np.isfinite(x): good.append(x)
        except Exception: pass
    return len(good)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--actual-signal')
    ap.add_argument('--actual-open')
    ap.add_argument('--score-key',default='score')
    ap.add_argument('--require-planned-count',action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/calibration_protocol.yaml').read_text())
    actual={'cal_signal':args.actual_signal,'cal_open':args.actual_open}
    pairs=[('signal','cal_signal'),('open','cal_open')];bad=[]
    for alpha_key,split in pairs:
        planned=int(cfg['splits'][split]['planned_n'])
        path=actual[split]
        n=count_scores(path,args.score_key) if path else planned
        source=f'actual:{path}' if path else 'planned'
        minp=1/(n+1)
        main_l,sens_l,opt_l=_levels(cfg,alpha_key); required=main_l+sens_l
        print(f'{split}: n={n} ({source}), planned_n={planned}, min attainable p={minp:.6f}, required={required}, optional={opt_l}')
        if args.require_planned_count and path and n<planned:
            bad.append((split,'realized_count_below_planned',n,planned))
        for a in required:
            if not minp < a: bad.append((split,'unattainable_alpha',n,a,minp))
        for a in opt_l:
            status='available' if minp < a else 'not enabled at current n'
            print(f'  optional alpha={a:g}: {status}')
    if bad:
        for x in bad: print('ERROR:',x)
        raise SystemExit(2)
    print('OK: all required conformal levels are attainable at the authoritative calibration sizes')
if __name__=='__main__':main()
