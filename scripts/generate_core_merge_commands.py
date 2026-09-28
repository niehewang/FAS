#!/usr/bin/env python3
"""Generate seed-aware L1/L2 core merge construction commands."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def q(x): return "'"+str(x).replace("'","'\\''")+"'"
def ref(p):
    p=Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)
def parse(s): return json.loads(s) if str(s).strip() else []

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/evidence_tiers/core.csv');ap.add_argument('--out-dir',default='configs/generated_core_merges');ap.add_argument('--output',default='jobs/run_core_merges.sh');args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')));rows=[r for r in rows if r['scenario'].startswith(('L1-','L2-'))]
    od=ROOT/args.out_dir;od.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','mkdir -p runs/{full,checkpoints,metadata}','']
    for r in rows:
        rid=r['run_id'];seed=int(r['seed']);parents=[x for x in r['parents'].split(';') if x];weights=parse(r['weights']);construction=r['construction'];out=f'runs/checkpoints/{rid}/model'
        L += [f'echo "=== {rid} ==="']
        for p in parents:
            adapter=f'runs/experts/s{seed}/{p}/adapter';full=f'runs/full/s{seed}/{p}'
            L += [f'test -d {adapter} || {{ echo "missing {adapter}"; exit 2; }}',f'if [ ! -f {full}/config.json ]; then mkdir -p runs/full/s{seed}; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter {adapter} --output {full}; fi']
        if construction=='linear':
            models=' '.join(f'runs/full/s{seed}/{p}' for p in parents);ws=' '.join(map(str,weights))
            L += [f'if [ ! -f {out}/config.json ]; then python scripts/merge_linear_models.py --models {models} --weights {ws} --normalize --output {out}; fi']
            cfg_ref='configs/experiment_matrix.yaml'
        elif construction=='ties':
            cfg={'merge_method':'ties','base_model':'Qwen/Qwen3-4B-Base','models':[{'model':f'runs/full/s{seed}/{p}','parameters':{'weight':float(w),'density':0.5}} for p,w in zip(parents,weights)],'parameters':{'normalize':True},'tokenizer_source':'base','dtype':'bfloat16'}
            cp=od/f'{rid}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8');cfg_ref=ref(cp)
            L += [f'if [ ! -f {out}/config.json ]; then mergekit-yaml {q(cfg_ref)} {q(out)} --cuda --lazy-unpickle; fi']
        else: raise SystemExit(f'unsupported core merge construction {construction}')
        L += [f'python scripts/record_run_metadata.py --run-id {q(rid)} --config {q(cfg_ref)} --checkpoint {q(out)} --parents {q(";".join(parents))} --output {q(f"runs/metadata/{rid}.json")}',f'python scripts/manifest_status.py set {q(rid)} ready --checkpoint-or-api {q(out)}','']
    L += ['echo "Core seed-aware merge descendants ready."']
    outp=ROOT/args.output;outp.parent.mkdir(parents=True,exist_ok=True);outp.write_text('\n'.join(L)+'\n',encoding='utf-8');outp.chmod(0o755);print(ref(outp))
if __name__=='__main__':main()
