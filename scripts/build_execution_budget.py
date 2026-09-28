#!/usr/bin/env python3
"""Summarize staged experiment cost and artifact reuse from the tiered manifest."""
from __future__ import annotations
import argparse,csv
from collections import Counter
from pathlib import Path

RESOURCE={
 'L1-linear-2p':'merge_only',
 'L2-ties-3p':'merge_only',
 'L3-dare-sft':'merge_plus_sft',
 'L4-merge-sft-int4':'merge_plus_sft',
 'L5-mixture-kd':'student_train',
 'L6-router-kd-sft':'student_train_plus_sft',
 'L7-deep-ancestry':'deep_teacher_plus_student',
 'L8-open-set':'audit_stress',
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest_tiered.csv');ap.add_argument('--output',default='EXECUTION_BUDGET.md');args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')))
    lines=['# Staged execution budget (v0.9)','',
      'This file separates the complete design space from the default expensive execution set.  The paper may describe the full matrix, but the default GPU plan executes only `core` until claim-specific gates pass.','']
    for tier in ['core','conditional','optional']:
        rr=[r for r in rows if r.get('evidence_tier')==tier]
        lines += [f'## {tier.title()} tier',f'- Planned rows: **{len(rr)}**']
        sc=Counter(r['scenario'] for r in rr)
        for k,v in sorted(sc.items()): lines.append(f'- `{k}`: {v}')
        rc=Counter(RESOURCE.get(r['scenario'],'other') for r in rr)
        lines.append('- Resource classes: '+', '.join(f'`{k}`={v}' for k,v in sorted(rc.items())))
        lines.append('')
    core=[r for r in rows if r.get('evidence_tier')=='core']
    # Expensive trained students are L5/L7 only; L7 teacher stage is shared across student scale and each seed.
    l5={(r['variant'],r['student'],r['seed']) for r in core if r['scenario'].startswith('L5-')}
    l7_students={(r['student'],r['seed']) for r in core if r['scenario'].startswith('L7-')}
    l7_teachers={r['seed'] for r in core if r['scenario'].startswith('L7-')}
    lines += ['## Default expensive training count (core)',
      f'- Formal sibling LoRA experts: **12** (4 domains × 3 independent genealogy seeds). Pilot still uses one convenience quartet.',
      f'- L5 distilled students: **{len(l5)}**.',
      f'- L7 shared deep-chain teacher stages: **{len(l7_teachers)}**.',
      f'- L7 final distilled students: **{len(l7_students)}**.',
      '- L1/L2 core rows are merge-only; each seed reuses its own independently trained sibling quartet.',
      '- G0 conformal and Known/Unknown-A open-set descendants are weighted-adapter compositions, not separately trained full models.',
      '- Functional-validity exact Shapley defaults to adapter compositions; nonlinear KD-Shapley is conditional.',
      '',
      '## Stop/go policy',
      '1. Run the synthetic functional-mixture death test and negative controls before any expensive student training.',
      '2. Run core L1/L2. Stop if FAS does not beat Random-IFRF at useful coverage.',
      '3. Run core L5 (1.7B). Only if C3 passes, run Router-KD, 0.6B cross-scale, or second-family breadth.',
      '4. Run core L7 only after L5 gives evidence of functional inheritance through distillation.',
      '5. Run diffusion only after the same mechanism is established on language models; it is a breadth validation, not a prerequisite for the central theory.',
    ]
    Path(args.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(args.output)
if __name__=='__main__': main()
