#!/usr/bin/env python3
"""Guard against Supplementary drifting into a duplicate of the main paper."""
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
tex=(ROOT/'supplementary.tex').read_text(encoding='utf-8')
inputs=re.findall(r'\\input\{([^}]+)\}',tex)
allowed={
 'supplementary/app_work_comparison',
 'supplementary/app_proofs',
 'supplementary/app_benchmark_additional',
 'supplementary/app_experiments_additional',
 'supplementary/app_additional_analysis',
 'supplementary/app_additional_threats',
}
legacy={
 'supplementary/app_related_full','supplementary/app_theory_full',
 'supplementary/app_benchmark_full','supplementary/app_experiments_full',
 'supplementary/app_discussion_full','supplementary/app_limitations_full',
 'supplementary/app_details',
}
errors=[]
extra=set(inputs)-allowed
missing=allowed-set(inputs)
if extra: errors.append('unexpected inputs: '+', '.join(sorted(extra)))
if missing: errors.append('missing approved inputs: '+', '.join(sorted(missing)))
if set(inputs)&legacy: errors.append('legacy duplicated sections are included')
for f in legacy:
    p=ROOT/(f+'.tex')
    if p.exists(): errors.append(f'legacy duplicate file still present: {p.relative_to(ROOT)}')
# Guard against importing main sections wholesale.
if any(x.startswith('sections/') for x in inputs): errors.append('main-paper section imported into supplementary')
if errors:
    print('SUPPLEMENTARY_SCOPE_FAIL')
    for e in errors: print('-',e)
    sys.exit(2)
print('SUPPLEMENTARY_SCOPE_OK', len(inputs), 'approved sections')
