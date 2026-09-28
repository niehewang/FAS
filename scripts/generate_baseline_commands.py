#!/usr/bin/env python3
"""Generate reproducible LLM-DNA and modelDNA baseline commands.

Requires a machine-local checkpoint map resolving the four sibling experts.
LLM-DNA is run for every target in the baseline target manifest. modelDNA is run
only where its open-weight/same-shape assumptions apply; other rows remain N/A.
"""
from __future__ import annotations
import argparse,csv,os,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def q(x):return "'"+str(x).replace("'","'\\''")+"'"

def expand(x):return os.path.expandvars(str(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',default='data/metadata/baseline_target_manifest.csv');ap.add_argument('--checkpoint-map',required=True);ap.add_argument('--truth',default='data/metadata/target_truth.csv');ap.add_argument('--output',default='jobs/run_baselines.sh');args=ap.parse_args()
    rows=list(csv.DictReader(open(args.targets,encoding='utf-8',newline='')))
    mp=yaml.safe_load(Path(args.checkpoint_map).read_text(encoding='utf-8')); base=expand(mp['base_model']); raw=mp['checkpoints']
    parents={'Math':expand(raw['math']),'Code':expand(raw['code']),'Medical':expand(raw['medical']),'Science':expand(raw['science'])}
    job=ROOT/args.output;job.parent.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','mkdir -p runs/baselines/{dna_targets,dna_decomp,modeldna} results/raw','',
       '# Strong baselines use their released upstream implementations.','python - <<\'PY\'\nimport importlib.metadata as m\nprint("llm-dna",m.version("llm-dna")); print("modeldna",m.version("modeldna"))\nPY','',
       'if [ ! -s runs/baselines/dna_ancestors.npz ]; then',
       '  python scripts/run_llm_dna_baseline.py --models '+ ' '.join(q(parents[k]) for k in ['Math','Code','Medical','Science']) +' --names Math Code Medical Science --output runs/baselines/dna_ancestors.npz','fi','']
    for r in rows:
        tid=r['target_id'];ck=r['checkpoint']
        if not ck: continue
        dna=f'runs/baselines/dna_targets/{tid}.npz'; dec=f'runs/baselines/dna_decomp/{tid}.json'
        L += [f'if [ ! -s {q(dna)} ]; then python scripts/run_llm_dna_baseline.py --models {q(ck)} --names Target --output {q(dna)}; fi',f'python scripts/decompose_vectors.py --ancestors runs/baselines/dna_ancestors.npz --target {q(dna)} --output {q(dec)}']
        if str(r.get('modeldna_applicable','0')) in {'1','true','True'}:
            md=f'runs/baselines/modeldna/{tid}.json'
            L += [f'python scripts/run_modeldna_baseline.py --suspect {q(ck)} --parents '+ ' '.join(q(parents[k]) for k in ['Math','Code','Medical','Science']) + f' --base {q(base)} --mergekit --allow-na --output {q(md)}']
    L += ['',
       f'python scripts/collect_baseline_jsons.py --truth {q(args.truth)} --input-dir runs/baselines/dna_decomp --method DNA-Decomp --output results/raw/dna_decomposition_predictions.csv',
       f'python scripts/collect_baseline_jsons.py --truth {q(args.truth)} --input-dir runs/baselines/modeldna --method modelDNA --output results/raw/modeldna_decomposition_predictions.csv',
       'python scripts/evaluate_decompositions.py results/raw/dna_decomposition_predictions.csv --output-prefix results/derived/dna_decomposition',
       'if [ -s results/raw/modeldna_decomposition_predictions.csv ] && [ "$(wc -l < results/raw/modeldna_decomposition_predictions.csv)" -gt 1 ]; then python scripts/evaluate_decompositions.py results/raw/modeldna_decomposition_predictions.csv --output-prefix results/derived/modeldna_decomposition; fi',
       'python scripts/sync_paper_results.py',
       'python scripts/build_paper_assets.py',
       'echo "Baseline extraction, evaluation, and paper synchronization complete."']
    job.write_text('\n'.join(L)+'\n',encoding='utf-8');job.chmod(0o755);print(job)
if __name__=='__main__':main()
