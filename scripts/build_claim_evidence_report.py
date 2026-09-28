#!/usr/bin/env python3
"""Build a non-interpretive claim evidence snapshot from paper-facing CSVs.

The report never decides that a scientific claim is 'proven'. It records which
planned evidence is populated and mechanically reports observable differences,
so the final wording can be checked against CLAIMS_AND_EVIDENCE.md.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'/'templates'

def read(name):
    p=RES/name
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def num(x):
    try:return float(x)
    except:return None
def row(rows,key,val):
    for r in rows:
        if str(r.get(key,'')).strip().lower()==val.lower():return r
    return None
def fmt(x):return 'NA' if x is None else f'{x:.4f}'

def main():
    main=read('main_results.csv'); act=read('active_probe_curve.csv'); valid=read('functional_validity.csv'); opens=read('open_set.csv'); cross=read('crossscale.csv'); cov=read('selective_coverage.csv')
    L=['# Claim Evidence Snapshot','', '> Auto-generated. This is an evidence inventory, not a proof/claim decision.','']
    fas=row(main,'method','FAS'); rnd=row(main,'method','Random-IFRF + NNLS'); dna=row(main,'method','DNA-Decomp')
    L += ['## Main support recovery']
    for st,col in [('Clean','clean_f1'),('SFT','sft_f1'),('KD','kd_f1'),('Deep','deep_f1')]:
        a=num(fas[col]) if fas else None;b=num(rnd[col]) if rnd else None;c=num(dna[col]) if dna else None
        L.append(f'- {st}: FAS={fmt(a)}, Random-IFRF={fmt(b)}, DNA-Decomp={fmt(c)}, FAS-Random={fmt(None if a is None or b is None else a-b)}')
    L += ['','## Active probing mechanism']
    budgets=sorted({r.get('budget','') for r in act if r.get('budget','')}, key=lambda x: float(x))
    for b in budgets:
        aa=next((r for r in act if r.get('budget')==b and r.get('method','').lower()=='active'),None); rr=next((r for r in act if r.get('budget')==b and r.get('method','').lower()=='random'),None)
        if aa or rr: L.append(f"- B={b}: active L1={fmt(num(aa.get('l1_error')) if aa else None)}, random L1={fmt(num(rr.get('l1_error')) if rr else None)}, active sigma_min={fmt(num(aa.get('sigma_min_raw')) if aa else None)}, random sigma_min={fmt(num(rr.get('sigma_min_raw')) if rr else None)}")
    L += ['','## Functional validity']
    for r in valid:L.append(f"- {r.get('quantity')}: Spearman={r.get('spearman') or 'NA'}")
    L += ['','## Selective/open-set evidence']
    for r in cov:L.append(f"- {r.get('setting')}: decomp={r.get('decomposable_rate') or 'NA'}, low-signal={r.get('low_signal_rate') or 'NA'}, bank-insufficient={r.get('bank_insufficient_rate') or 'NA'}, conditional F1={r.get('conditional_parent_f1') or 'NA'}")
    for r in opens:L.append(f"- open/{r.get('setting')}: AUROC={r.get('auroc') or 'NA'}, FPR95={r.get('fpr95') or 'NA'}, false-abstain@.05={r.get('false_abstain_delta05') or 'NA'}")
    L += ['','## Cross-scale paired-anchor evidence']
    for r in cross:L.append(f"- {r.get('setting')} | {r.get('anchor')}: F1={r.get('parent_f1') or 'NA'}, decomp={r.get('decomposable_rate') or 'NA'}, queries={r.get('queries') or 'NA'}")
    out=ROOT/'results'/'CLAIM_EVIDENCE_REPORT.md';out.write_text('\n'.join(L)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
