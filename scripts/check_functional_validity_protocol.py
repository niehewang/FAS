#!/usr/bin/env python3
"""Static guardrails for domain-conditioned functional-validity experiments.

The paper defines pi^F(D), so counterfactual Shapley values computed for a
functional domain D must be compared with FAS coordinates estimated from probes
selected within the *same* D.  This checker prevents a future refactor from
silently reverting to one global ancestry vector.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def need(text, token, errs, where):
    if token not in text: errs.append(f'{where}: missing {token!r}')

def forbid(text, token, errs, where):
    if token in text: errs.append(f'{where}: forbidden legacy pattern {token!r}')

def main():
    errs=[]
    gen=(ROOT/'scripts/generate_shapley_commands.py').read_text(encoding='utf-8')
    asm=(ROOT/'scripts/assemble_functional_validity.py').read_text(encoding='utf-8')
    exp=(ROOT/'sections/07_experiments.tex').read_text(encoding='utf-8')
    bench=(ROOT/'sections/06_benchmark.tex').read_text(encoding='utf-8')
    prep=(ROOT/'scripts/prepare_pilot_datasets.py').read_text(encoding='utf-8')
    need(gen,'select_domain_probes.py',errs,'generate_shapley_commands.py')
    need(gen,"--fas {q(dom+\"=\"+coord)}",errs,'generate_shapley_commands.py')
    need(asm,'DOMAIN=coordinate.json',errs,'assemble_functional_validity.py')
    need(asm,"if dom=='ALL' or dom not in fas: continue",errs,'assemble_functional_validity.py')
    # Main text must explicitly preserve domain conditioning.
    if not (('同一功能域' in exp or '同域' in exp) and ('\\pi_i^F(\\cD)' in exp or '\\boldsymbol\\pi^F(\\cD)' in exp)):
        errs.append('sections/07_experiments.tex: functional-validity paragraph must state same-domain pi^F(D) comparison')
    if '\\varphi_i(\\cD)' not in bench:
        errs.append('sections/06_benchmark.tex: domain-conditioned Shapley definition missing')
    # Pilot utility must not execute generated code; it is a semantic/capability utility benchmark.
    need(prep,"'metric'",errs,'prepare_pilot_datasets.py')
    if 'exec(' in prep or 'subprocess.run' in prep:
        errs.append('prepare_pilot_datasets.py: code execution must not be introduced into pilot utility construction')
    if errs:
        print('FUNCTIONAL_VALIDITY_PROTOCOL_FAIL')
        for e in errs: print(' -',e)
        raise SystemExit(2)
    print('FUNCTIONAL_VALIDITY_PROTOCOL_OK domain-conditioned FAS/Shapley comparison enforced')

if __name__=='__main__': main()
