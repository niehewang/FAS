#!/usr/bin/env python3
"""Validate paper-facing and raw result CSV schemas before asset generation.

The validator catches accidental column edits, duplicate IDs, non-numeric values
in known numeric columns, and invalid selective states. Blank cells are allowed
because the manuscript intentionally starts with experiment placeholders.
"""
from __future__ import annotations
import argparse,csv,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

EXPECTED={
  'templates/main_results.csv':['method','clean_f1','sft_f1','kd_f1','deep_f1','audit_access'],
  'templates/scaling.csv':['setting','true_parents','candidate_bank','parent_f1','exact_support','l1_error'],
  'templates/functional_validity.csv':['quantity','spearman'],
  'templates/open_set.csv':['setting','auroc','fpr95','false_abstain_delta05'],
  'templates/crossscale.csv':['setting','anchor','parent_f1','decomposable_rate','queries'],
  'templates/crossmodality.csv':['modality','setting','parent_f1','decomposable_rate','queries'],
  'templates/ablation.csv':['variant','parent_f1','l1_error','open_auroc'],
  'templates/active_probe_curve.csv':['method','budget','sigma_min_raw','sigma_min_unit','coherence','l1_error'],
  'templates/ancestry_trajectory.csv':['stage','ancestor','coordinate'],
  'templates/ancestry_profile.csv':['domain','ancestor','coordinate'],
  'templates/sampling_budget.csv':['m','strategy','sigma_min_raw','l1_error','online_calls'],
  'templates/geometry_mechanism.csv':['quantity','spearman','ci_low','ci_high'],
  'templates/encoder_sensitivity.csv':['encoder','parent_f1','coordinate_spearman','support_jaccard'],
  'templates/probe_type_stats.csv':['intervention_type','selected_fraction','mean_marginal_gain'],
  'templates/negative_controls.csv':['control','false_positive_or_f1','signal_pass_or_expected','interpretation'],
  'templates/selective_coverage.csv':['setting','decomposable_rate','low_signal_rate','bank_insufficient_rate','conditional_parent_f1'],
  'templates/coordinate_uncertainty.csv':['setting','ancestor','coordinate','ci_low','ci_high','support_frequency'],
  'raw_templates/decomposition_predictions.csv':['target_id','method','setting','seed','parent_names','true_support','true_pi','pred_pi','pred_support','rho','p_signal','p_open','state','queries'],
  'raw_templates/functional_vectors.csv':['case_id','domain','quantity','parent','value','shapley'],
  'raw_templates/geometry_runs.csv':['run_id','strategy','budget','seed','sigma_min_raw','sigma_min_unit','coherence','logdet','coordinate_l1'],
  'raw_templates/open_set_scores.csv':['target_id','setting','is_unknown','score','p_signal','p_open','state'],
  'raw_templates/signal_gate_scores.csv':['split','label','score','run_id'],
  'raw_templates/subset_utilities.csv':['domain','subset','utility'],
}
NUMERIC_HINTS={'seed','rho','p_signal','p_open','queries','score','is_unknown','value','shapley','utility','budget','sigma_min_raw','sigma_min_unit','coherence','logdet','coordinate_l1','parent_f1','exact_support','l1_error','auroc','fpr95','false_abstain_delta05','spearman','ci_low','ci_high','coordinate','support_frequency','selected_fraction','mean_marginal_gain','online_calls','m','true_parents','candidate_bank','clean_f1','clean_l1','sft_f1','sft_l1','kd_f1','kd_l1','deep_f1','deep_l1','open_auroc','coordinate_spearman','support_jaccard','false_positive_or_f1','signal_pass_or_expected','decomposable_rate','low_signal_rate','bank_insufficient_rate','conditional_parent_f1','conditional_l1'}
VALID_STATES={'','decomposable','low_signal','bank_insufficient','uncalibrated'}
ID_KEYS={'raw_templates/decomposition_predictions.csv':['target_id','method','setting','seed'],
         'raw_templates/open_set_scores.csv':['target_id','setting'],
         'raw_templates/geometry_runs.csv':['run_id'],
         'raw_templates/signal_gate_scores.csv':['run_id','split','label']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='results'); args=ap.parse_args()
    root=ROOT/args.root; errors=[]; warnings=[]
    for rel,headers in EXPECTED.items():
        p=root/rel
        if not p.exists(): errors.append(f'missing {p.relative_to(ROOT)}'); continue
        with p.open(encoding='utf-8',newline='') as f:
            r=csv.DictReader(f); actual=r.fieldnames or []; rows=list(r)
        if actual!=headers: errors.append(f'{p.relative_to(ROOT)} header mismatch: expected {headers}, got {actual}')
        seen=set(); idkeys=ID_KEYS.get(rel)
        for i,row in enumerate(rows,2):
            if 'state' in row and (row.get('state') or '').strip() not in VALID_STATES:
                errors.append(f'{p.relative_to(ROOT)}:{i} invalid state={row.get("state")!r}')
            for k,v in row.items():
                if k in NUMERIC_HINTS and str(v).strip():
                    token=str(v).strip().lower()
                    if token in {'n/a','na','not_applicable','weight reads','weight_reads','--','—'}:
                        continue
                    try:
                        x=float(v)
                        if not math.isfinite(x): raise ValueError
                    except Exception: errors.append(f'{p.relative_to(ROOT)}:{i} {k} is not finite numeric or approved N/A token: {v!r}')
            if idkeys:
                key=tuple((row.get(k) or '').strip() for k in idkeys)
                if all(key):
                    if key in seen: errors.append(f'{p.relative_to(ROOT)}:{i} duplicate key {idkeys}={key}')
                    seen.add(key)
        if not rows: warnings.append(f'{p.relative_to(ROOT)} has no data rows')
    print(f'validated {len(EXPECTED)} schemas')
    for x in warnings: print('WARNING:',x)
    if errors:
        for x in errors: print('ERROR:',x)
        raise SystemExit(1)
    print('OK: result schemas are structurally valid')
if __name__=='__main__': main()
