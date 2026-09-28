#!/usr/bin/env python3
"""Generate formal sibling-expert configs for independent genealogy seeds.

Pilot experts remain under runs/experts/<domain>. Formal TPAMI experiments use
runs/experts/s<seed>/<domain> so seeds correspond to independently trained
ancestor families rather than repeated auditing of one fixed genealogy.
"""
from __future__ import annotations
import argparse,copy
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=['math','code','medical','science']

def ref(p):
    p=Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='11,23,47');ap.add_argument('--data-dir',default='data/training/formal');ap.add_argument('--out-dir',default='configs/formal_experts');ap.add_argument('--job',default='jobs/train_formal_experts.sh');args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    out=ROOT/args.out_dir;out.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','', '# Formal sibling experts: independent genealogy seeds.']
    for seed in seeds:
        for d in DOMAINS:
            src=ROOT/f'configs/experts/{d}.yaml';cfg=yaml.safe_load(src.read_text(encoding='utf-8'))
            cfg=copy.deepcopy(cfg);cfg['seed']=seed;cfg['output_dir']=f'runs/experts/s{seed}/{d}';cfg['data']['path']=f'{args.data_dir}/{d}.jsonl'
            p=out/f's{seed}/{d}.yaml';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
            L += [f'test -s {cfg["data"]["path"]} || {{ echo "missing formal corpus: {cfg["data"]["path"]}"; exit 2; }}',f'if [ ! -d {cfg["output_dir"]}/adapter ]; then python scripts/train_lora_expert.py --config {ref(p)}; fi']
    L += ['echo "Formal seeded sibling experts ready."']
    job=ROOT/args.job;job.parent.mkdir(parents=True,exist_ok=True);job.write_text('\n'.join(L)+'\n',encoding='utf-8');job.chmod(0o755)
    print(f'{len(seeds)*len(DOMAINS)} configs -> {ref(out)}');print(ref(job))
if __name__=='__main__':main()
