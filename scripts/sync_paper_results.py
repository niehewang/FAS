#!/usr/bin/env python3
"""Populate paper-facing result CSVs from derived metric files when available.

This script is intentionally conservative: it only overwrites a cell when it
can match a derived metric unambiguously. It never fabricates or interpolates
numbers. Run it before `build_paper_assets.py`.
"""
from __future__ import annotations
import argparse,csv,shutil
from pathlib import Path


def read(path):
    p=Path(path)
    return list(csv.DictReader(p.open(encoding='utf-8',newline=''))) if p.exists() else []


def write(path,rows,fields):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def fmt(v):
    if v in ('',None): return ''
    try: return f"{float(v):.4f}"
    except Exception: return str(v)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--templates',default='results/templates'); ap.add_argument('--derived',default='results/derived'); args=ap.parse_args()
    t=Path(args.templates); d=Path(args.derived); changed=[]

    # Main results: merge all decomposition summaries that use the same schema.
    # FAS/random/absolute live in decomposition_summary.csv; strong released
    # baselines are evaluated into their own summaries so N/A settings are not
    # silently turned into zeros.  Concatenate only the rows that actually
    # exist; method/setting keys remain the paper-facing join contract.
    summ=[]
    for name in ['decomposition_summary.csv','dna_decomposition_summary.csv','modeldna_decomposition_summary.csv']:
        summ.extend(read(d/name))
    mp=t/'main_results.csv'
    if summ and mp.exists():
        rows=read(mp); idx={(r.get('method',''),r.get('setting','')):r for r in summ}
        setting_cols={'clean':'clean_f1','sft':'sft_f1','kd':'kd_f1','deep':'deep_f1'}
        for r in rows:
            for st,cf in setting_cols.items():
                z=idx.get((r['method'],st))
                if z and z.get('end_to_end_parent_f1_mean','')!='': r[cf]=fmt(z['end_to_end_parent_f1_mean'])
        write(mp,rows,list(rows[0].keys())); changed.append(str(mp))


    # Selective coverage: FAS only, derived from decomposition summary.
    sc=t/'selective_coverage.csv'
    if summ and sc.exists():
        rows=read(sc); m={(r.get('method',''),r.get('setting','')):r for r in summ}
        for r in rows:
            z=m.get(('FAS',r['setting'])) or m.get(('fas',r['setting']))
            if z:
                for src,dst in [('decomposable_rate','decomposable_rate'),('low_signal_rate','low_signal_rate'),('bank_insufficient_rate','bank_insufficient_rate'),('parent_f1_mean','conditional_parent_f1')]:
                    if z.get(src,'')!='': r[dst]=fmt(z[src])
        write(sc,rows,list(rows[0].keys())); changed.append(str(sc))

    # Functional validity.
    fv=read(d/'functional_validity.csv'); fp=t/'functional_validity.csv'
    if fv and fp.exists():
        rows=read(fp); m={r['quantity']:r for r in fv}
        for r in rows:
            if r['quantity'] in m: r['spearman']=fmt(m[r['quantity']].get('spearman',''))
        write(fp,rows,list(rows[0].keys())); changed.append(str(fp))

    # Open-set: if derived file contains matching setting rows.
    om=read(d/'open_set_metrics.csv'); op=t/'open_set.csv'
    if om and op.exists():
        rows=read(op); m={r['setting']:r for r in om}
        for r in rows:
            z=m.get(r['setting'])
            if z:
                for src,dst in [('auroc','auroc'),('fpr95','fpr95'),('false_abstain_delta0.05','false_abstain_delta05')]:
                    if z.get(src,'')!='': r[dst]=fmt(z[src])
        write(op,rows,list(rows[0].keys())); changed.append(str(op))


    # Active-probe curve from the first death test is already paper-shaped.
    ar=Path('results/raw/active_probe_curve.csv'); at=t/'active_probe_curve.csv'
    if ar.exists() and at.exists():
        src=read(ar); dst=read(at)
        if src and dst and set(src[0].keys())==set(dst[0].keys()):
            shutil.copyfile(ar,at); changed.append(str(at))

    # Geometry mechanism is already paper-shaped; copy only when schema matches.
    gm=d/'geometry_mechanism.csv'; gp=t/'geometry_mechanism.csv'
    if gm.exists() and gp.exists():
        src=read(gm); dst=read(gp)
        # Current paper template may evolve; only copy when identical headers.
        if src and dst and set(src[0].keys())==set(dst[0].keys()):
            shutil.copyfile(gm,gp); changed.append(str(gp))

    print('Updated:' if changed else 'No unambiguous derived results found.')
    for x in changed: print(' -',x)

if __name__=='__main__': main()
