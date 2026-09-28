#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Exact 3-parent adapter-merge Shapley + domain-conditioned FAS pipeline.
mkdir -p runs/shapley/{adapters,eval,selected,responses,prepared,coordinates} results/raw results/derived
AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}

python scripts/build_weighted_lora_adapters.py \
  --base 'Qwen/Qwen3-4B-Base' \
  --plan 'runs/shapley/adapter_merge_plan.csv' \
  --adapter Math=runs/experts/math/adapter \
  --adapter Code=runs/experts/code/adapter \
  --adapter Medical=runs/experts/medical/adapter \
  --adapter Science=runs/experts/science/adapter \
  --output-dir runs/shapley/adapters

python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --data 'data/evaluation/pilot_utility.jsonl' --output runs/shapley/eval/empty.json
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Math.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Medical' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Medical.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Science' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Science.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Medical' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Math_Medical.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Science' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Math_Science.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Medical_Science' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Medical_Science.json'
python scripts/evaluate_text_utility.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Medical_Science' --data 'data/evaluation/pilot_utility.jsonl' --output 'runs/shapley/eval/subset_Math_Medical_Science.json'
python scripts/collect_subset_utilities.py --plan 'runs/shapley/adapter_merge_plan.csv' --eval-dir runs/shapley/eval --output results/raw/subset_utilities.csv
python scripts/compute_shapley.py results/raw/subset_utilities.csv --output results/derived/shapley.csv

python scripts/select_domain_probes.py --bank 'runs/pilot/response/response_bank.select.npz' --probes 'data/probes/interventions_pilot.jsonl' --domain 'Math' --budget 32 --output 'runs/shapley/selected/math.json'
python scripts/subset_probe_pool.py --probes 'data/probes/interventions_pilot.jsonl' --selected 'runs/shapley/selected/math.json' --output 'runs/shapley/selected/math.jsonl'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Medical_Science' --probes 'runs/shapley/selected/math.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 8300 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/math.target.npz'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes 'runs/shapley/selected/math.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10008319 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/math.anchor.npz'
python scripts/prepare_target_decomposition.py --bank 'runs/pilot/response/response_bank.estimate.npz' --base 'runs/shapley/responses/math.anchor.npz' --target 'runs/shapley/responses/math.target.npz' --selected 'runs/shapley/selected/math.json' --target-selected-only --output 'runs/shapley/prepared/math.npz'
python scripts/analyze_fas_coordinates.py 'runs/shapley/prepared/math.npz' --domain 'Math' --output 'runs/shapley/coordinates/math.json'
python scripts/select_domain_probes.py --bank 'runs/pilot/response/response_bank.select.npz' --probes 'data/probes/interventions_pilot.jsonl' --domain 'Medical' --budget 32 --output 'runs/shapley/selected/medical.json'
python scripts/subset_probe_pool.py --probes 'data/probes/interventions_pilot.jsonl' --selected 'runs/shapley/selected/medical.json' --output 'runs/shapley/selected/medical.jsonl'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Medical_Science' --probes 'runs/shapley/selected/medical.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 8400 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/medical.target.npz'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes 'runs/shapley/selected/medical.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10008419 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/medical.anchor.npz'
python scripts/prepare_target_decomposition.py --bank 'runs/pilot/response/response_bank.estimate.npz' --base 'runs/shapley/responses/medical.anchor.npz' --target 'runs/shapley/responses/medical.target.npz' --selected 'runs/shapley/selected/medical.json' --target-selected-only --output 'runs/shapley/prepared/medical.npz'
python scripts/analyze_fas_coordinates.py 'runs/shapley/prepared/medical.npz' --domain 'Medical' --output 'runs/shapley/coordinates/medical.json'
python scripts/select_domain_probes.py --bank 'runs/pilot/response/response_bank.select.npz' --probes 'data/probes/interventions_pilot.jsonl' --domain 'Science' --budget 32 --output 'runs/shapley/selected/science.json'
python scripts/subset_probe_pool.py --probes 'data/probes/interventions_pilot.jsonl' --selected 'runs/shapley/selected/science.json' --output 'runs/shapley/selected/science.jsonl'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --adapter 'runs/shapley/adapters/subset_Math_Medical_Science' --probes 'runs/shapley/selected/science.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 8500 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/science.target.npz'
python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes 'runs/shapley/selected/science.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10008519 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/shapley/responses/science.anchor.npz'
python scripts/prepare_target_decomposition.py --bank 'runs/pilot/response/response_bank.estimate.npz' --base 'runs/shapley/responses/science.anchor.npz' --target 'runs/shapley/responses/science.target.npz' --selected 'runs/shapley/selected/science.json' --target-selected-only --output 'runs/shapley/prepared/science.npz'
python scripts/analyze_fas_coordinates.py 'runs/shapley/prepared/science.npz' --domain 'Science' --output 'runs/shapley/coordinates/science.json'
python scripts/assemble_functional_validity.py --shapley results/derived/shapley.csv --case-id adapter_merge_triplet --fas 'Math=runs/shapley/coordinates/math.json' --fas 'Medical=runs/shapley/coordinates/medical.json' --fas 'Science=runs/shapley/coordinates/science.json' --construction-weights 'runs/shapley/construction_weights.json' --output results/raw/functional_vectors.csv
python scripts/compute_functional_validity.py results/raw/functional_vectors.csv --output results/derived/functional_validity.csv
python scripts/sync_paper_results.py
python scripts/build_paper_assets.py
echo "Domain-conditioned exact Shapley/FAS functional-validity pipeline complete."
