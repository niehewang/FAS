#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
R=Path(a.run_root); rows=[]
for s in [23,47,71]:
    p=R/f's{s}/diagnostic.json'
    if p.exists(): rows.extend(json.loads(p.read_text())['rows'])

def subset(st): return [r for r in rows if r['strategy']==st]
def meanpath(rr,*ks):
    vals=[]
    for r in rr:
        x=r
        for k in ks: x=x[k]
        vals.append(float(x))
    return float(np.mean(vals)) if vals else None
summary={}
for st in ['active','random']:
    rr=subset(st)
    summary[st]={
      'n':len(rr),
      'legacy_residual':meanpath(rr,'legacy_stage5','reported_residual'),
      'protocol_normed_paired_residual':meanpath(rr,'protocol_normed_paired','relative_residual'),
      'protocol_normed_paired_parent_f1_if_scored':meanpath(rr,'protocol_normed_paired','parent_f1_if_scored'),
      'unit_direction_residual':meanpath(rr,'unit_direction_diagnostic','relative_residual'),
      'diag_transfer_residual':meanpath(rr,'anchor_diagonal_transfer','relative_residual'),
      'ridge_transfer_residual':meanpath(rr,'anchor_ridge_transfer','relative_residual'),
      'absolute_normed_residual':meanpath(rr,'absolute_protocol_normed_context','relative_residual'),
      'anchor_raw_cosine':meanpath(rr,'anchor_eval','raw_mean_cosine'),
      'anchor_ridge_cosine':meanpath(rr,'anchor_eval','ridge_mean_cosine'),
      'anchor_raw_relerr':meanpath(rr,'anchor_eval','raw_relative_error'),
      'anchor_ridge_relerr':meanpath(rr,'anchor_eval','ridge_relative_error'),
      'no_anchor_residual':meanpath(rr,'anchor_controls','no_anchor','relative_residual'),
      'wrong_4b_anchor_residual':meanpath(rr,'anchor_controls','wrong_4b_anchor','relative_residual'),
      'shuffled_1p7b_anchor_residual':meanpath(rr,'anchor_controls','shuffled_1p7b_anchor','relative_residual')
    }
act=summary.get('active',{})
raw=act.get('protocol_normed_paired_residual'); ridge=act.get('ridge_transfer_residual'); unit=act.get('unit_direction_residual')
anchor_raw=act.get('anchor_raw_relerr'); anchor_ridge=act.get('anchor_ridge_relerr')
ridge_promising=bool(None not in (raw,ridge,anchor_raw,anchor_ridge) and ridge <= raw-0.05 and anchor_ridge <= 0.9*anchor_raw)
unit_promising=bool(None not in (raw,unit) and unit <= raw-0.05)
controls=[act.get('no_anchor_residual'),act.get('wrong_4b_anchor_residual'),act.get('shuffled_1p7b_anchor_residual')]
controls=[x for x in controls if x is not None]
paired_helpful=bool(raw is not None and controls and raw < min(controls)-0.01)
if ridge_promising:
    rec='Anchor-only cross-scale mapping materially improves held-out anchor alignment and L5 residual. Next: freeze a scale-aware transfer rule on anchor-only data, build matched G1 calibration for that rule, then re-evaluate existing KD students before any L7 training.'
elif unit_promising:
    rec='Direction normalization materially improves L5 residual. Next: investigate scale/magnitude normalization using anchor-only calibration, then re-evaluate existing KD students before L7.'
else:
    rec='Neither anchor-only ridge transfer nor direction normalization yields a material residual improvement. Treat current 4B-to-1.7B IFRF dictionary transfer as structurally weak; do not run L7 yet. Next diagnostic should test scale-matched 1.7B ancestor fields or revise the cross-scale representation before more lineage depth.'
out={
 'stage':'stage5d_crossscale_anchor_diagnostic_v1','seeds':[23,47,71],'summary':summary,
 'stage5_C3_status':'not_supported_by_current_L5_measurement',
 'implementation_consistency':{'stage5_legacy_missing_global_ancestor_norms':True,'effect':'changes pi/support scaling but not NNLS cone residual; does not overturn C3 failure'},
 'diagnostic_flags':{'anchor_ridge_promising':ridge_promising,'unit_direction_promising':unit_promising,'paired_anchor_helpful_vs_controls':paired_helpful},
 'recommended_branch':rec,
 'guardrail':'Post-hoc diagnostics only; no transformed variant has a matched-G0 finite-sample guarantee and construction exposure is not functional ground truth.'
}
Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
