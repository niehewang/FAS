#!/usr/bin/env python3
"""Generate deterministic staged manifests and formal core execution jobs."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(*args): subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--tiered',default='data/metadata/experiment_run_manifest_tiered.csv');args=ap.parse_args()
    run('scripts/expand_experiment_matrix.py','--output',args.full_manifest)
    run('scripts/assign_evidence_tiers.py','--manifest',args.full_manifest,'--output',args.tiered)
    run('scripts/build_execution_budget.py','--manifest',args.tiered)
    run('scripts/generate_seeded_expert_configs.py')
    run('scripts/generate_seeded_bank_commands.py')
    core='data/metadata/evidence_tiers/core.csv'
    run('scripts/generate_core_merge_commands.py','--manifest',core)
    run('scripts/generate_full_experiment_commands.py','--manifest',core,'--out-dir','configs/generated_core_runs','--shared-out-dir','configs/generated_core_shared','--job','jobs/run_core_nonlinear.sh')
    # One role-disjoint G0 plan is reused across independent ancestor families; the actual descendants are seed-specific.
    run('scripts/generate_calibration_plan.py','--output-dir','runs/calibration/plan_v09')
    for seed in [11,23,47]:
        run('scripts/generate_calibration_commands.py','--plan-dir','runs/calibration/plan_v09','--bank',f'runs/banks/s{seed}/response_bank.estimate.npz','--selected',f'runs/banks/s{seed}/selected_probes.json','--probes','data/probes/interventions.jsonl','--run-root',f'runs/calibration/s{seed}','--ancestor-root',f'runs/experts/s{seed}','--output',f'jobs/run_g0_calibration_s{seed}.sh')
    run('scripts/generate_audit_commands.py','--manifest',core,'--output','jobs/run_core_formal_audit.sh')
    run('scripts/generate_open_set_commands.py','--seed','11','--output','jobs/run_core_open_set.sh')
    run('scripts/check_execution_tiers.py','--manifest',args.tiered)
    print('Formal staged plan ready:')
    for x in ['jobs/train_formal_experts.sh','jobs/build_formal_banks.sh','jobs/run_core_merges.sh','jobs/run_g0_calibration_s11.sh','jobs/run_g0_calibration_s23.sh','jobs/run_g0_calibration_s47.sh','jobs/run_core_nonlinear.sh','jobs/run_core_formal_audit.sh','jobs/run_core_open_set.sh']:
        print(' -',x)
if __name__=='__main__': main()
