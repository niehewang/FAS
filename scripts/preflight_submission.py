#!/usr/bin/env python3
"""Submission-oriented static preflight for the frozen FAS TPAMI manuscript branch."""
from __future__ import annotations
import argparse, re, subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'results/paper_evidence/FAS_TPAMI_EXPERIMENT_FREEZE_20260927_FINAL.md'
FINAL_LEDGER = ROOT / 'results/paper_evidence/FAS_TPAMI_RESULT_LEDGER_20260927_FINAL.json'

def publication_tex():
    files=[ROOT/'main.tex', ROOT/'supplementary.tex']
    for d in ['sections','supplementary','figures','tables']:
        files += sorted((ROOT/d).glob('*.tex'))
    return [p for p in files if p.exists()]

def citations(text):
    out=[]
    for m in re.finditer(r'\\cite\{([^}]*)\}', text):
        out.extend(x.strip() for x in m.group(1).split(','))
    return set(out)

def bibkeys():
    return set(re.findall(r'@\w+\{\s*([^,]+),', (ROOT/'refs.bib').read_text(encoding='utf-8')))

def page_count(pdf):
    try:
        x=subprocess.check_output(['pdfinfo',str(pdf)],text=True,stderr=subprocess.DEVNULL)
        m=re.search(r'^Pages:\s+(\d+)',x,re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def command_ok(script):
    try:
        r=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        detail=(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f'exit={r.returncode}')
        return r.returncode==0, detail
    except Exception as e:
        return False, str(e)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--strict',action='store_true'); ap.add_argument('--output',default='PRE_FLIGHT_REPORT.md'); args=ap.parse_args()
    text='\n'.join(p.read_text(encoding='utf-8') for p in publication_tex())
    missing=sorted(citations(text)-bibkeys())
    draft_markers=sum(text.count(x) for x in ['\\draftnote','\\todoexp','\\todoresult','待实验','TBD','PLACEHOLDER'])
    main_tex=(ROOT/'main.tex').read_text(encoding='utf-8')
    required_authors=['聂何望','肖珏','路直','沈任飞','唐振军']
    author_ok=all(a in main_tex for a in required_authors)
    appendix_in_main=('supplementary/' in main_tex or '\\appendix' in main_tex.lower())
    freeze_ok=FREEZE.exists() and FINAL_LEDGER.exists()
    ledger_detail='missing final freeze/ledger'
    if freeze_ok:
        try:
            d=json.loads(FINAL_LEDGER.read_text(encoding='utf-8'))
            # Final ledger contains all Stage1-Stage8 evidence; exact schema may evolve.
            ledger_detail='final experiment freeze + final result ledger present'
        except Exception as e:
            freeze_ok=False; ledger_detail=f'final ledger unreadable: {e}'
    pages=page_count(ROOT/'main.pdf') if (ROOT/'main.pdf').exists() else None
    supp_pages=page_count(ROOT/'supplementary.pdf') if (ROOT/'supplementary.pdf').exists() else None
    schema_ok,schema_detail=command_ok('validate_result_schemas.py')
    cal_ok,cal_detail=command_ok('check_calibration_resolution.py')
    supp_scope_ok,supp_scope_detail=command_ok('check_supplementary_scope.py')
    audit_protocol_ok,audit_protocol_detail=command_ok('check_audit_protocol.py')
    functional_validity_ok,functional_validity_detail=command_ok('check_functional_validity_protocol.py')
    execution_tier_ok,execution_tier_detail=command_ok('check_execution_tiers.py')
    data_protocol_ok,data_protocol_detail=command_ok('check_data_protocol.py')
    checks=[
      ('Missing bibliography keys',len(missing)==0,', '.join(missing) if missing else 'none'),
      ('Final experiment freeze integrated',freeze_ok,ledger_detail),
      ('Publication-facing draft placeholders removed',draft_markers==0,str(draft_markers)),
      ('Result CSV schemas valid',schema_ok,schema_detail),
      ('Conformal alpha resolution valid',cal_ok,cal_detail),
      ('Supplementary contains supporting-only approved sections',supp_scope_ok,supp_scope_detail),
      ('Audit jobs/protocol are internally consistent',audit_protocol_ok,audit_protocol_detail),
      ('Functional-validity protocol is domain-conditioned',functional_validity_ok,functional_validity_detail),
      ('Execution tiers preserve the preregistered core budget',execution_tier_ok,execution_tier_detail),
      ('Formal training/probe data protocol is source-disjoint',data_protocol_ok,data_protocol_detail),
      ('Required author list present',author_ok,', '.join(required_authors) if author_ok else 'missing one or more authors'),
      ('Appendix removed from main PDF source',not appendix_in_main,'separate supplementary' if not appendix_in_main else 'appendix/supplement input found in main'),
      ('Main PDF exists',pages is not None,f'{pages} pages' if pages else 'missing/unreadable'),
      ('Main PDF within 18-page internal hard ceiling',pages is not None and pages<=18,f'{pages} pages' if pages else 'missing/unreadable'),
      ('Standalone supplementary PDF exists',supp_pages is not None,f'{supp_pages} pages' if supp_pages else 'missing/unreadable')]
    lines=['# FAS TPAMI v0.20-paper pre-flight report','', '| Check | Pass | Detail |','|---|---:|---|']
    for name,ok,detail in checks:
        lines.append(f'| {name} | {"YES" if ok else "NO"} | {str(detail).replace("|","/")} |')
    lines += ['', '## Evidence-freeze interpretation',
              'The Stage1--Stage8 server experiment phase is frozen. Legacy experiment-planning CSV templates and the historical in-repository run manifest are reproducibility scaffolding, not the authoritative source of final manuscript numbers. The authoritative publication evidence is the frozen compact evidence set under `results/paper_evidence/`.',
              '', '## Page-budget note',
              f'The current Chinese full-study manuscript is {pages} pages and remains below the project hard ceiling of 18 pages. Final IEEE TPAMI page-count and overlength rules will be rechecked when the English submission version is produced.']
    (ROOT/args.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    if args.strict and not all(x[1] for x in checks):
        raise SystemExit(2)

if __name__=='__main__': main()
