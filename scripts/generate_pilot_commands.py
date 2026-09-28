#!/usr/bin/env python3
"""Generate a resumable shell plan for the first FAS 'death test'.

The generated script does not run automatically. It gives the author a single,
ordered command file for: pilot data -> four sibling LoRAs -> materialized
checkpoints -> response bank -> active probes -> controlled pairwise merges.
Expensive commands are guarded by file/directory existence checks so the script
can be resumed after interruption.
"""
from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='jobs/run_pilot.sh'); ap.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2'); ap.add_argument('--samples',type=int,default=4); args=ap.parse_args()
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    domains=['math','code','medical','science']; base='Qwen/Qwen3-4B-Base'
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','', '# Auto-generated pilot plan. Review GPU/model/data licenses before running.','mkdir -p runs/pilot/{experts,full,response,merges,selected}','AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}','AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}','']
    L += ['if [ ! -s data/training/pilot/math.jsonl ]; then','  python scripts/prepare_pilot_datasets.py --output-dir data/training/pilot','fi','']
    for d in domains:
        L += [f'if [ ! -d runs/experts/{d}/adapter ]; then',f'  python scripts/train_lora_expert.py --config configs/experts/{d}.yaml','fi','']
    L += ['PROBES=${PROBES:-data/probes/interventions_pilot.jsonl}',
          'if [ ! -s "$PROBES" ]; then',
          '  python scripts/build_pilot_interventions.py --input data/distillation/pilot_prompts_with_domains.jsonl --output "$PROBES" --per-domain 250',
          'fi',
          'python scripts/validate_probe_pool.py "$PROBES"','']
    models=[('base',base,None)]+[(d,base,f'runs/experts/{d}/adapter') for d in domains]
    for name,model,adapter in models:
        adapter_arg=f' --adapter {adapter}' if adapter else ''
        L += [f'if [ ! -f runs/pilot/response/{name}.npz ]; then',f'  python scripts/run_text_probe_bank.py --model {model}{adapter_arg} --probes "$PROBES" --encoder {args.encoder} --samples {args.samples} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/{name}.npz','fi']
    for name,_,_ in models:
        L += [f'python scripts/split_embedding_replicates.py runs/pilot/response/{name}.npz --select-output runs/pilot/response/{name}.select.npz --estimate-output runs/pilot/response/{name}.estimate.npz']
    L += ['', 'python scripts/build_response_bank.py --base runs/pilot/response/base.select.npz --ancestors ' + ' '.join(f'runs/pilot/response/{d}.select.npz' for d in domains) + ' --names Math Code Medical Science --output runs/pilot/response/response_bank.select.npz',
          'python scripts/build_response_bank.py --base runs/pilot/response/base.estimate.npz --ancestors ' + ' '.join(f'runs/pilot/response/{d}.estimate.npz' for d in domains) + ' --names Math Code Medical Science --output runs/pilot/response/response_bank.estimate.npz',
          'for B in 8 16 32 64 128; do python scripts/select_probes.py runs/pilot/response/response_bank.select.npz --budget "$B" --balanced --probe-jsonl "$PROBES" --group-key domain --output "runs/pilot/selected/probes_B${B}.json"; done',
          'python scripts/subset_probe_pool.py --probes "$PROBES" --selected runs/pilot/selected/probes_B64.json --output runs/pilot/selected/probes_B64.jsonl',
          'python scripts/subset_response_npz.py --input runs/pilot/response/base.estimate.npz --selected runs/pilot/selected/probes_B64.json --output runs/pilot/response/base_B64.estimate.npz','']
    L += ['',
          '# Phase 1: exact functional-mixture death test (no full 4B checkpoint materialization required).',
          'python scripts/generate_synthetic_response_targets.py --bank runs/pilot/response/response_bank.estimate.npz --output-dir runs/pilot/synthetic_targets --manifest runs/pilot/synthetic_targets.csv',
          'python scripts/benchmark_probe_selection.py --selection-bank runs/pilot/response/response_bank.select.npz --bank runs/pilot/response/response_bank.estimate.npz --base runs/pilot/response/base.estimate.npz --targets runs/pilot/synthetic_targets.csv --budgets 8,16,32,64,128 --random-repeats 20 --output-curve results/raw/active_probe_curve.csv --output-geometry results/raw/geometry_runs.csv',
          'python scripts/compute_geometry_mechanism.py results/raw/geometry_runs.csv --output results/derived/geometry_mechanism.csv',
          'python scripts/sync_paper_results.py',
          'python scripts/build_paper_assets.py',
          '',
          'echo "Phase-1 functional-mixture death test complete. Inspect geometry/error mechanism before real merges."',
          '',
          '# Phase 2 is optional: RUN_REAL_MERGES=1 materializes experts and queries real weight merges.',
          'if [ "${RUN_REAL_MERGES:-0}" = "1" ]; then']
    for d in domains:
        L += [f'  if [ ! -f runs/pilot/full/{d}/config.json ]; then',f'    python scripts/materialize_lora.py --base {base} --adapter runs/experts/{d}/adapter --output runs/pilot/full/{d}','  fi']
    pairs=[('math_code',['math','code'],[.5,.5]),('math_medical',['math','medical'],[.3,.7]),('code_science',['code','science'],[.7,.3])]
    for tag,ps,ws in pairs:
        L += [f'  if [ ! -f runs/pilot/merges/{tag}/config.json ]; then',f'    python scripts/merge_linear_models.py --models ' + ' '.join(f'runs/pilot/full/{p}' for p in ps) + ' --weights ' + ' '.join(map(str,ws)) + f' --output runs/pilot/merges/{tag}','  fi']
        L += [f'  if [ ! -f runs/pilot/response/target_{tag}_B64.npz ]; then',f'    python scripts/run_text_probe_bank.py --model runs/pilot/merges/{tag} --probes runs/pilot/selected/probes_B64.jsonl --encoder {args.encoder} --samples {args.samples} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/target_{tag}_B64.npz','  fi']
    L += ['  echo "Real merge checkpoints/responses prepared. Their construction weights are not treated as FAS coordinate ground truth; use support and counterfactual functional analyses."','fi']
    out.write_text('\n'.join(L)+'\n',encoding='utf-8'); out.chmod(0o755); print(out)
if __name__=='__main__': main()
