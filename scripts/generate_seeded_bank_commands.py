#!/usr/bin/env python3
"""Generate formal seed-specific ancestor response-bank jobs.

Each genealogy seed gets an independent sibling family, cross-fitted ancestor
responses, a select-bank, an estimate-bank, and its own active probe subset.
"""
from __future__ import annotations
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=['math','code','medical','science']
LABELS=['Math','Code','Medical','Science']

def ref(p):
    p=Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)

def q(x): return "'"+str(x).replace("'","'\\''")+"'"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='11,23,47');ap.add_argument('--probes',default='data/probes/interventions.jsonl');ap.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2');ap.add_argument('--samples',type=int,default=4);ap.add_argument('--budget',type=int,default=64);ap.add_argument('--output',default='jobs/build_formal_banks.sh');args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if args.samples<2: raise SystemExit('formal bank cross-fitting requires samples>=2')
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','',f'test -s {q(args.probes)} || {{ echo "missing formal probe pool"; exit 2; }}','AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}','AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}','']
    for seed in seeds:
        root=f'runs/banks/s{seed}';raw=f'{root}/raw';split=f'{root}/split';L += [f'echo "=== FORMAL BANK s{seed} ==="',f'mkdir -p {raw} {split}']
        common=f'--probes {q(args.probes)} --encoder {q(args.encoder)} --samples {args.samples} --seed {10000+seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds'
        L += [f'if [ ! -s {raw}/base.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base {common} --output {raw}/base.npz; fi']
        for d in DOMAINS:
            L += [f'test -d runs/experts/s{seed}/{d}/adapter || {{ echo "missing runs/experts/s{seed}/{d}/adapter"; exit 2; }}',f'if [ ! -s {raw}/{d}.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s{seed}/{d}/adapter {common} --output {raw}/{d}.npz; fi']
        for name in ['base']+DOMAINS:
            L += [f'python scripts/split_embedding_replicates.py {raw}/{name}.npz --seed {20000+seed} --select-output {split}/{name}.select.npz --estimate-output {split}/{name}.estimate.npz']
        ancestors_sel=' '.join(f'{split}/{d}.select.npz' for d in DOMAINS);ancestors_est=' '.join(f'{split}/{d}.estimate.npz' for d in DOMAINS);names=' '.join(LABELS)
        L += [f'python scripts/build_response_bank.py --base {split}/base.select.npz --ancestors {ancestors_sel} --names {names} --output {root}/response_bank.select.npz',f'python scripts/build_response_bank.py --base {split}/base.estimate.npz --ancestors {ancestors_est} --names {names} --output {root}/response_bank.estimate.npz',f'python scripts/select_probes.py {root}/response_bank.select.npz --budget {args.budget} --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output {root}/selected_probes.json','']
    L += ['echo "All formal seed-specific ancestry banks ready."']
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(L)+'\n',encoding='utf-8');out.chmod(0o755);print(ref(out))
if __name__=='__main__':main()
