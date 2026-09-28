#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed23-eval',required=True)
    ap.add_argument('--seed47-eval',required=True)
    ap.add_argument('--seed71-eval',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    paths={23:Path(a.seed23_eval),47:Path(a.seed47_eval),71:Path(a.seed71_eval)}
    ev={s:json.loads(p.read_text()) for s,p in paths.items()}
    seeds={}
    pooled_new=[]; pooled_old=[]; pooled_imp=[]
    all_strategy_improvements=[]
    for s in [23,47,71]:
        d=ev[s]
        rows=[]
        for r in d['rows']:
            n=r['same_scale_1p7b_surrogate_fas']
            row={
                'strategy':r['strategy'],
                'cross_scale_residual':float(r['cross_scale_4b_dictionary_reference_residual']),
                'same_scale_residual':float(n['relative_residual']),
                'improvement':float(n['improvement_vs_cross_scale']),
                'pi':[float(x) for x in n['pi']],
                'exposure_l1_diagnostic':float(n['exposure_l1_diagnostic']),
                'support_size_diagnostic':int(n['support_size']),
                'parent_f1_if_scored_diagnostic':float(n['parent_f1_if_scored'])
            }
            rows.append(row)
            pooled_new.append(row['same_scale_residual']); pooled_old.append(row['cross_scale_residual']); pooled_imp.append(row['improvement'])
            if s in [47,71]: all_strategy_improvements.append(row['improvement'])
        seed_new=float(np.mean([x['same_scale_residual'] for x in rows]))
        seed_old=float(np.mean([x['cross_scale_residual'] for x in rows]))
        seed_imp=seed_old-seed_new
        seed_gate=bool(seed_new<=0.80 and seed_imp>=0.15 and min(x['improvement'] for x in rows)>=0.10)
        align=d.get('cross_scale_field_alignment',{})
        seeds[str(s)]={
            'rows':rows,
            'same_scale_mean_residual':seed_new,
            'cross_scale_reference_mean_residual':seed_old,
            'mean_improvement':seed_imp,
            'seed_gate_same_rule_as_stage5e':seed_gate,
            'mean_field_cosine_4b_vs_1p7b':float(np.mean([float(v['field_cosine_flat']) for v in align.values()])) if align else None,
            'field_alignment':align
        }
    pooled_new_mean=float(np.mean(pooled_new)); pooled_old_mean=float(np.mean(pooled_old)); pooled_imp_mean=float(np.mean(pooled_imp))
    gate=bool(
        seeds['47']['seed_gate_same_rule_as_stage5e'] and
        seeds['71']['seed_gate_same_rule_as_stage5e'] and
        min(all_strategy_improvements)>=0.10 and
        pooled_new_mean<=0.80
    )
    out={
        'protocol':'FAS_STAGE5F_SCALEMATCHED17_MULTISEED_V1',
        'scientific_role':'diagnostic_only_scale_matched_surrogate_dictionary_not_genealogical_ancestor_evidence',
        'seeds':seeds,
        'pooled':{
            'same_scale_mean_residual':pooled_new_mean,
            'cross_scale_reference_mean_residual':pooled_old_mean,
            'mean_absolute_improvement':pooled_imp_mean
        },
        'multiseed_gate':{
            'frozen_rule':'seeds 47 and 71 each: mean same-scale residual<=0.80, mean improvement>=0.15, each Active/Random improvement>=0.10; pooled all-three-seed same-scale mean residual<=0.80',
            'pass':gate,
            'recommended_next':('design_and_test_frozen_scale_aware_bridge_on_actual_4b_to_1p7b_lineage_before_L7' if gate else 'stop_surrogate_expansion_and_revise_functional_representation_before_L7')
        },
        'interpretation_guardrails':[
            'Passing this diagnostic does not retroactively make Stage5 cross-scale C3 pass.',
            '1.7B surrogate experts are not genealogical parents of the KD students.',
            'Support/F1 here is diagnostic only because all four teachers were exposed and no G1 calibrated support threshold exists.',
            'Original cross-scale Stage5 and strong-baseline results must remain reported.'
        ]
    }
    Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
