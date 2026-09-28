#!/usr/bin/env python3
"""Static checks for paper-facing audit jobs.

Catches protocol regressions that unit tests cannot see: deterministic duplicate
replicates, full-pool suspect queries, broken shell working directories, and
accidental finite-sample claims on G1 calibration transfer runs.
"""
from __future__ import annotations
import re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[]

def need(cond,msg):
    if not cond: errors.append(msg)

for rel in ['jobs/run_pilot.sh','jobs/run_nonlinear_experiments.sh','jobs/run_nonlinear_audit.sh','jobs/slurm_probe_bank.sh','jobs/slurm_train_expert.sh']:
    p=ROOT/rel
    need(p.exists(),f'missing {rel}')
    if p.exists():
        cp=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
        need(cp.returncode==0,f'{rel}: bash -n failed: {cp.stderr.strip()}')

pilot=(ROOT/'jobs/run_pilot.sh').read_text(encoding='utf-8')
need('cd "$(dirname "$0")/.."' in pilot,'pilot must cd to repository root')
need('AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}' in pilot,'pilot must use stochastic audit decoding by default')
need('subset_probe_pool.py' in pilot,'pilot real-merge phase must materialize selected probes')
need('target_math_code_B64.npz' in pilot,'pilot real merge should query selected B probes rather than full pool')

audit=(ROOT/'jobs/run_nonlinear_audit.sh').read_text(encoding='utf-8')
need('--target-selected-only' in audit,'nonlinear audit must use target-selected-only decomposition')
need('runs/audit/selected_probes.jsonl' in audit,'nonlinear audit must query selected probes only')
need('CALIBRATION_REGIME=${CALIBRATION_REGIME:-g1_transfer}' in audit,'nonlinear audit must default to G1 transfer semantics')
need('SIGNAL_CAL_1P7B' in audit and 'SIGNAL_CAL_0P6B' in audit,'cross-scale signal calibration must be anchor-specific')
need('--temperature "$AUDIT_TEMPERATURE"' in audit,'nonlinear audit must use stochastic decoding when m>1')

# No generated job may invoke an unsupported --config on the probe runner.
for p in (ROOT/'jobs').glob('*.sh'):
    t=p.read_text(encoding='utf-8')
    need('run_text_probe_bank.py --config' not in t,f'{p.name}: unsupported probe --config invocation')
    # Heuristic: a command with --samples >1 must expose a positive-temperature argument.
    for line in t.splitlines():
        if 'run_text_probe_bank.py' in line and re.search(r'--samples\s+(?:[2-9]|\d{2,})',line):
            need('--temperature' in line,f'{p.name}: samples>1 without explicit stochastic temperature')


# v0.9 formal staged jobs must preserve independent genealogy seeds.
formal_jobs=['jobs/train_formal_experts.sh','jobs/build_formal_banks.sh','jobs/run_core_merges.sh','jobs/run_core_nonlinear.sh','jobs/run_core_formal_audit.sh','jobs/run_core_open_set.sh']
for rel in formal_jobs:
    p=ROOT/rel
    need(p.exists(),f'missing {rel}; run generate_staged_execution.py')
    if p.exists():
        cp=subprocess.run(['bash','-n',str(p)],capture_output=True,text=True)
        need(cp.returncode==0,f'{rel}: bash -n failed: {cp.stderr.strip()}')

if (ROOT/'jobs/train_formal_experts.sh').exists():
    t=(ROOT/'jobs/train_formal_experts.sh').read_text(encoding='utf-8')
    for seed in [11,23,47]: need(f'runs/experts/s{seed}/math/adapter' in t,f'formal experts missing seed s{seed}')
    need('runs/experts/math/adapter' not in t,'formal experts must not use legacy unseeded path')
if (ROOT/'jobs/build_formal_banks.sh').exists():
    t=(ROOT/'jobs/build_formal_banks.sh').read_text(encoding='utf-8')
    for seed in [11,23,47]:
        need(f'runs/banks/s{seed}/response_bank.estimate.npz' in t,f'formal bank missing s{seed}')
        need(f'runs/banks/s{seed}/selected_probes.json' in t,f'formal selected probes missing s{seed}')
    need(t.count('--balanced') >= 3,'formal banks must use Balanced D-optimal selection')
    need(t.count('--group-key domain') >= 3,'formal banks must balance by frozen domain groups')
if (ROOT/'jobs/run_core_formal_audit.sh').exists():
    t=(ROOT/'jobs/run_core_formal_audit.sh').read_text(encoding='utf-8')
    need('--target-selected-only' in t,'formal audit must be selected-only')
    for seed in [11,23,47]:
        need(f'runs/banks/s{seed}/response_bank.estimate.npz' in t,f'formal audit missing seed-matched bank s{seed}')
        need(f'runs/calibration/s{seed}/scores/signal_scores.csv' in t,f'formal audit missing seed-matched calibration s{seed}')
if (ROOT/'jobs/run_core_open_set.sh').exists():
    t=(ROOT/'jobs/run_core_open_set.sh').read_text(encoding='utf-8')
    need('--names Math,Code,Medical' in t,'open-set verifier bank must exclude Science')
    need('unknown_A' in t and 'Science' in t,'Unknown-A must use withheld Science ancestor')
    need('--balanced' in t and '--group-key domain' in t,'open-set probe selection must use Balanced D-optimal selection')

teacher=(ROOT/'scripts/generate_teacher_data.py').read_text(encoding='utf-8')
need('unload_model' in teacher and 'by_teacher' in teacher,'teacher generation must stream one teacher at a time')

if errors:
    print('AUDIT_PROTOCOL_FAIL')
    for e in errors: print(' -',e)
    sys.exit(1)
print('AUDIT_PROTOCOL_OK')
