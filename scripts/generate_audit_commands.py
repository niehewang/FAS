#!/usr/bin/env python3
"""Generate seed-matched selected-probe audit commands for formal runs.

Formal targets are decomposed against the ancestry dictionary from the same
independently trained genealogy seed.  Main nonlinear/cross-scale runs are
`g1_transfer` by default; the only finite-sample matched claim remains the G0
weighted-adapter calibration experiment.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STUDENT={'1p7b':'Qwen/Qwen3-1.7B-Base','0p6b':'Qwen/Qwen3-0.6B-Base'}

def q(s): return "'"+str(s).replace("'","'\\''")+"'"

def target_spec(r):
    rid=r['run_id']; sc=r['scenario']; student=r.get('student','')
    if sc.startswith(('L1-','L2-')): return f'runs/checkpoints/{rid}/model', None, 'Qwen/Qwen3-4B-Base', 'g0_transfer'
    if sc.startswith('L3-'): return f'runs/checkpoints/{rid}/model', None, 'Qwen/Qwen3-4B-Base', 'g1_transfer'
    if sc.startswith('L4-'): return f'runs/checkpoints/{rid}/model', None, 'Qwen/Qwen3-4B-Base', 'g1_transfer'
    if sc.startswith('L5-'): return STUDENT[student], f'runs/checkpoints/{rid}/student/adapter', STUDENT[student], 'g1_transfer'
    if sc.startswith('L6-'): return f'runs/checkpoints/{rid}/student_router_full', f'runs/checkpoints/{rid}/student_router_sft/adapter', STUDENT[student], 'g1_transfer'
    if sc.startswith('L7-'): return STUDENT[student], f'runs/checkpoints/{rid}/deep_student/adapter', STUDENT[student], 'g1_transfer'
    raise ValueError(sc)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--probes',default='data/probes/interventions.jsonl');ap.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2');ap.add_argument('--samples',type=int,default=4);ap.add_argument('--output',default='jobs/run_formal_audit.sh');args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')));rows=[r for r in rows if r['scenario'].startswith(('L1-','L2-','L3-','L4-','L5-','L6-','L7-'))]
    seeds=sorted({int(r['seed']) for r in rows})
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','',
       '# Formal audit: seed-matched ancestor dictionary and active probe subset.',
       '# L1/L2 use G0-transfer diagnostics; postprocessed/KD/deep runs use G1-transfer by default.',
       'AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}','AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}','mkdir -p runs/formal_audit','']
    # One selected probe JSONL per genealogy seed.
    for seed in seeds:
        L += [f'test -s runs/banks/s{seed}/selected_probes.json || {{ echo "missing seed-matched selected probes s{seed}"; exit 2; }}',f'test -s runs/banks/s{seed}/response_bank.estimate.npz || {{ echo "missing seed-matched estimate bank s{seed}"; exit 2; }}',f'test -s runs/calibration/s{seed}/scores/signal_scores.csv || {{ echo "missing G0 signal calibration s{seed}"; exit 2; }}',f'test -s runs/calibration/s{seed}/scores/open_scores.csv || {{ echo "missing G0 open calibration s{seed}"; exit 2; }}',f'mkdir -p runs/formal_audit/s{seed}/{{responses,inputs,outputs}}',f'python scripts/subset_probe_pool.py --probes {q(args.probes)} --selected runs/banks/s{seed}/selected_probes.json --output runs/formal_audit/s{seed}/selected_probes.jsonl','']
    common=f'--encoder {q(args.encoder)} --samples {args.samples} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds'
    # Generate only the anchor sizes actually used in this manifest/seed.
    anchor_done=set()
    for r in rows:
        seed=int(r['seed']); model,adapter,anchor,regime=target_spec(r); tag='4b' if anchor.endswith('4B-Base') else ('1p7b' if anchor.endswith('1.7B-Base') else '0p6b')
        key=(seed,tag)
        if key not in anchor_done:
            ar=f'runs/formal_audit/s{seed}/responses/base_{tag}.npz'; L += [f'if [ ! -s {ar} ]; then python scripts/run_text_probe_bank.py --model {q(anchor)} --probes runs/formal_audit/s{seed}/selected_probes.jsonl {common} --seed {30000+seed} --output {ar}; fi',''];anchor_done.add(key)
    for r in rows:
        rid=r['run_id'];seed=int(r['seed']);model,adapter,anchor,regime=target_spec(r);tag='4b' if anchor.endswith('4B-Base') else ('1p7b' if anchor.endswith('1.7B-Base') else '0p6b')
        root=f'runs/formal_audit/s{seed}';resp=f'{root}/responses/{rid}.npz';inp=f'{root}/inputs/{rid}.npz';dec=f'{root}/outputs/{rid}.decomposition.json'; bank=f'runs/banks/s{seed}/response_bank.estimate.npz';sel=f'runs/banks/s{seed}/selected_probes.json';sig=f'runs/calibration/s{seed}/scores/signal_scores.csv';opn=f'runs/calibration/s{seed}/scores/open_scores.csv';tau_json=f'runs/calibration/s{seed}/scores/support_threshold.json'
        cmd=f'python scripts/run_text_probe_bank.py --model {q(model)}'
        if adapter: cmd += f' --adapter {q(adapter)}'
        cmd += f' --probes {root}/selected_probes.jsonl {common} --seed {40000+seed} --output {q(resp)}'
        L += [f'echo "AUDIT {rid}"',f'test -s {tau_json} || {{ echo "missing support threshold s{seed}"; exit 2; }}',f'TAU=$(python -c "import json; print(json.load(open(\'{tau_json}\'))[\'selected\'][\'threshold\'])")',f'if [ ! -s {q(resp)} ]; then {cmd}; fi',f'python scripts/prepare_target_decomposition.py --bank {bank} --base {root}/responses/base_{tag}.npz --target {q(resp)} --selected {sel} --target-selected-only --output {q(inp)}',f'python scripts/decompose_target.py {q(inp)} --target-id {q(rid)} --threshold "$TAU" --signal-calibration {sig} --open-calibration {opn} --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime {regime} --output {q(dec)}',f'python scripts/manifest_status.py set {q(rid)} done --checkpoint-or-api {q(model + ((" + "+adapter) if adapter else ""))}','']
    L += ['echo "Formal seed-matched audit complete. Collect JSONs with collect_decomposition_jsons.py."']
    out.write_text('\n'.join(L)+'\n',encoding='utf-8');out.chmod(0o755)
    try: print(out.relative_to(ROOT))
    except ValueError: print(out)
if __name__=='__main__': main()
