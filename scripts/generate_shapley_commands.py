#!/usr/bin/env python3
"""Generate exact three-parent counterfactual utility/Shapley jobs.

Default low-cost mode uses weighted LoRA adapter composition and creates all
2^3 subset descendants under a fixed-coefficient counterfactual game: removing
a parent sets only that parent's coefficient to zero while the remaining parent
coefficients stay unchanged, so marginal contribution is not confounded by
reweighting. A renormalized-subset variant is available only as sensitivity.
It then evaluates
all subset models, computes exact Shapley values, and performs a domain-wise FAS
analysis of the full triplet using probes selected *within each domain*.

The latter detail is essential: pi^F(D) is domain-conditioned, so a global FAS
vector must not be correlated against domain-specific Shapley contributions.
"""
from __future__ import annotations
import argparse,csv,itertools,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def q(x): return "'"+str(x).replace("'","'\\''")+"'"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parents',default='Math,Medical,Science')
    ap.add_argument('--weights',default='0.2,0.5,0.3')
    ap.add_argument('--base',default='Qwen/Qwen3-4B-Base')
    ap.add_argument('--eval-data',default='data/evaluation/pilot_utility.jsonl')
    ap.add_argument('--probes',default='data/probes/interventions_pilot.jsonl')
    ap.add_argument('--selection-bank',default='runs/pilot/response/response_bank.select.npz')
    ap.add_argument('--estimate-bank',default='runs/pilot/response/response_bank.estimate.npz')
    ap.add_argument('--encoder',default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--budget',type=int,default=32)
    ap.add_argument('--samples',type=int,default=4)
    ap.add_argument('--subset-weight-rule',choices=['fixed','renormalized'],default='fixed',help='fixed keeps each parent coefficient from the full model; renormalized is a supplementary sensitivity')
    ap.add_argument('--plan',default='runs/shapley/adapter_merge_plan.csv')
    ap.add_argument('--job',default='jobs/run_shapley_adapter_merge.sh')
    args=ap.parse_args()
    parents=[x.strip() for x in args.parents.split(',') if x.strip()]
    base_weights=[float(x) for x in args.weights.split(',') if x.strip()]
    if len(parents)!=3 or len(base_weights)!=3: raise SystemExit('exact default protocol requires exactly 3 parents and 3 weights')
    den=sum(base_weights); base_w={p:w/den for p,w in zip(parents,base_weights)}
    plan=ROOT/args.plan;plan.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for k in range(1,4):
        for sub in itertools.combinations(parents,k):
            cid='subset_'+'_'.join(sub)
            if args.subset_weight_rule=='fixed':
                wsub={p:base_w[p] for p in sub}
            else:
                d=sum(base_w[p] for p in sub)
                wsub={p:base_w[p]/d for p in sub}
            rows.append({'calibration_id':cid,'subset':';'.join(sub),'parents':json.dumps(list(sub)),'weights':json.dumps(wsub,sort_keys=True),'weight_rule':args.subset_weight_rule})
    with plan.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['calibration_id','subset','parents','weights','weight_rule']);w.writeheader();w.writerows(rows)
    weights_json=ROOT/'runs/shapley/construction_weights.json';weights_json.parent.mkdir(parents=True,exist_ok=True);weights_json.write_text(json.dumps(base_w,indent=2),encoding='utf-8')
    full_id='subset_'+'_'.join(parents)
    job=ROOT/args.job;job.parent.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','',
       '# Exact 3-parent adapter-merge Shapley + domain-conditioned FAS pipeline.','mkdir -p runs/shapley/{adapters,eval,selected,responses,prepared,coordinates} results/raw results/derived',
       'AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}','AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}','',
       'python scripts/build_weighted_lora_adapters.py \\',f'  --base {q(args.base)} \\',f'  --plan {q(args.plan)} \\',
       '  --adapter Math=runs/experts/math/adapter \\', '  --adapter Code=runs/experts/code/adapter \\', '  --adapter Medical=runs/experts/medical/adapter \\', '  --adapter Science=runs/experts/science/adapter \\', '  --output-dir runs/shapley/adapters','',
       f'python scripts/evaluate_text_utility.py --model {q(args.base)} --data {q(args.eval_data)} --output runs/shapley/eval/empty.json']
    for r in rows:
        cid=r['calibration_id']
        L.append(f'python scripts/evaluate_text_utility.py --model {q(args.base)} --adapter {q("runs/shapley/adapters/"+cid)} --data {q(args.eval_data)} --output {q("runs/shapley/eval/"+cid+".json")}')
    L += [f'python scripts/collect_subset_utilities.py --plan {q(args.plan)} --eval-dir runs/shapley/eval --output results/raw/subset_utilities.csv','python scripts/compute_shapley.py results/raw/subset_utilities.csv --output results/derived/shapley.csv','']
    fas_args=[]
    for j,dom in enumerate(parents):
        tag=dom.lower(); sel=f'runs/shapley/selected/{tag}.json'; seljsonl=f'runs/shapley/selected/{tag}.jsonl'; targ=f'runs/shapley/responses/{tag}.target.npz'; anc=f'runs/shapley/responses/{tag}.anchor.npz'; prep=f'runs/shapley/prepared/{tag}.npz'; coord=f'runs/shapley/coordinates/{tag}.json'; seed=8300+j*100
        L += [f'python scripts/select_domain_probes.py --bank {q(args.selection_bank)} --probes {q(args.probes)} --domain {q(dom)} --budget {args.budget} --output {q(sel)}',f'python scripts/subset_probe_pool.py --probes {q(args.probes)} --selected {q(sel)} --output {q(seljsonl)}',f'python scripts/run_text_probe_bank.py --model {q(args.base)} --adapter {q("runs/shapley/adapters/"+full_id)} --probes {q(seljsonl)} --encoder {q(args.encoder)} --samples {args.samples} --seed {seed} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {q(targ)}',f'python scripts/run_text_probe_bank.py --model {q(args.base)} --probes {q(seljsonl)} --encoder {q(args.encoder)} --samples {args.samples} --seed {seed+10000019} --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output {q(anc)}',f'python scripts/prepare_target_decomposition.py --bank {q(args.estimate_bank)} --base {q(anc)} --target {q(targ)} --selected {q(sel)} --target-selected-only --output {q(prep)}',f'python scripts/analyze_fas_coordinates.py {q(prep)} --domain {q(dom)} --output {q(coord)}']
        fas_args.append(f'--fas {q(dom+"="+coord)}')
    L += ['python scripts/assemble_functional_validity.py --shapley results/derived/shapley.csv --case-id adapter_merge_triplet '+ ' '.join(fas_args) + f' --construction-weights {q(str(weights_json.relative_to(ROOT)))} --output results/raw/functional_vectors.csv','python scripts/compute_functional_validity.py results/raw/functional_vectors.csv --output results/derived/functional_validity.csv','python scripts/sync_paper_results.py','python scripts/build_paper_assets.py','echo "Domain-conditioned exact Shapley/FAS functional-validity pipeline complete."']
    job.write_text('\n'.join(L)+'\n',encoding='utf-8');job.chmod(0o755)
    print(plan);print(job)
if __name__=='__main__':main()
