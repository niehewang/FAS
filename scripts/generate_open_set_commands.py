#!/usr/bin/env python3
"""Generate the core s11 restricted-bank open-set experiment without a new unseen expert."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def q(x):return "'"+str(x).replace("'","'\\''")+"'"

def ref(p):
    p=Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=11);ap.add_argument('--probes',default='data/probes/interventions.jsonl');ap.add_argument('--output',default='jobs/run_core_open_set.sh');args=ap.parse_args();seed=args.seed
    root=f'runs/open_set/s{seed}';bankroot=f'runs/banks/s{seed}'
    # Restricted bank is materialized at runtime; generators must not depend on future artifacts.
    plan_dir=f'{root}/cal_plan'
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."',f'mkdir -p {root}/{{adapters,responses,inputs,outputs}}','',
       f'python scripts/subset_ancestor_bank.py {bankroot}/response_bank.select.npz --names Math,Code,Medical --output {root}/bank.select.npz',
       f'python scripts/subset_ancestor_bank.py {bankroot}/response_bank.estimate.npz --names Math,Code,Medical --output {root}/bank.estimate.npz',
       f'python scripts/select_probes.py {root}/bank.select.npz --budget 64 --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output {root}/selected.json',
       f'python scripts/subset_probe_pool.py --probes {q(args.probes)} --selected {root}/selected.json --output {root}/selected_probes.jsonl','',
       f'python scripts/generate_calibration_plan.py --config configs/calibration_open_set.yaml --parents Math,Code,Medical --seed 20260921 --output-dir {plan_dir} --g1-shift-n 0',
       f'python scripts/generate_calibration_commands.py --plan-dir {plan_dir} --bank {root}/bank.estimate.npz --selected {root}/selected.json --probes {q(args.probes)} --run-root {root}/calibration --ancestor-root runs/experts/s{seed} --output {root}/run_calibration.sh',
       f'bash {root}/run_calibration.sh','',
       '# Build Known-only and Unknown-A targets. Science is deliberately withheld from the verifier bank.',
       f'cat > {root}/targets.csv <<\'CSV\'','calibration_id,parents,weights',
       'known_only,"[""Math"",""Code"",""Medical""]","{""Math"":0.34,""Code"":0.33,""Medical"":0.33}"',
       'unknown_A,"[""Math"",""Code"",""Science""]","{""Math"":0.34,""Code"":0.33,""Science"":0.33}"','CSV',
       f'python scripts/build_weighted_lora_adapters.py --base Qwen/Qwen3-4B-Base --plan {root}/targets.csv --adapter Math=runs/experts/s{seed}/math/adapter --adapter Code=runs/experts/s{seed}/code/adapter --adapter Medical=runs/experts/s{seed}/medical/adapter --adapter Science=runs/experts/s{seed}/science/adapter --output-dir {root}/adapters','',
       'AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}','AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}',
       f'python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes {root}/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed {50000+seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {root}/responses/base4b.npz']
    for name in ['known_only','unknown_A']:
        resp=f'{root}/responses/{name}.npz';inp=f'{root}/inputs/{name}.npz';dec=f'{root}/outputs/L8-open-set__{name}__same__s{seed}.decomposition.json'
        L += [f'python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter {root}/adapters/{name} --probes {root}/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed {51000+seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {resp}',f'python scripts/prepare_target_decomposition.py --bank {root}/bank.estimate.npz --base {root}/responses/base4b.npz --target {resp} --selected {root}/selected.json --target-selected-only --output {inp}',f'TAU=$(python -c "import json; print(json.load(open(\'{root}/calibration/scores/support_threshold.json\'))[\'selected\'][\'threshold\'])")',f'python scripts/decompose_target.py {inp} --target-id L8-open-set__{name}__same__s{seed} --threshold "$TAU" --signal-calibration {root}/calibration/scores/signal_scores.csv --open-calibration {root}/calibration/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime matched --output {dec}','']
    # Unknown-B reuses the seed-matched core Deep-Chain descendant; it should not be called unknown.
    rid=f'L7-deep-ancestry__v1__1p7b__s{seed}';resp=f'{root}/responses/unknown_B.npz';inp=f'{root}/inputs/unknown_B.npz';dec=f'{root}/outputs/L8-open-set__unknown_B_heavy_transform__same__s{seed}.decomposition.json'
    L += [f'python scripts/run_text_probe_bank.py --model Qwen/Qwen3-1.7B-Base --adapter runs/checkpoints/{rid}/deep_student/adapter --probes {root}/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed {52000+seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {resp}',f'python scripts/run_text_probe_bank.py --model Qwen/Qwen3-1.7B-Base --probes {root}/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed {53000+seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {root}/responses/base1p7b.npz',f'python scripts/prepare_target_decomposition.py --bank {root}/bank.estimate.npz --base {root}/responses/base1p7b.npz --target {resp} --selected {root}/selected.json --target-selected-only --output {inp}',f'TAU=$(python -c "import json; print(json.load(open(\'{root}/calibration/scores/support_threshold.json\'))[\'selected\'][\'threshold\'])")',f'python scripts/decompose_target.py {inp} --target-id L8-open-set__unknown_B_heavy_transform__same__s{seed} --threshold "$TAU" --signal-calibration {root}/calibration/scores/signal_scores.csv --open-calibration {root}/calibration/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output {dec}','',f'python scripts/collect_open_set_outputs.py "{root}/outputs/*.decomposition.json" --output results/raw/open_set_scores.csv','python scripts/compute_open_set_metrics.py results/raw/open_set_scores.csv','python scripts/sync_paper_results.py','python scripts/build_paper_assets.py']
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(L)+'\n',encoding='utf-8');out.chmod(0o755);print(ref(out))
if __name__=='__main__':main()
