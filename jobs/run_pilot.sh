#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-generated pilot plan. Review GPU/model/data licenses before running.
mkdir -p runs/pilot/{experts,full,response,merges,selected}
AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}

if [ ! -s data/training/pilot/math.jsonl ]; then
  python scripts/prepare_pilot_datasets.py --output-dir data/training/pilot
fi

if [ ! -d runs/experts/math/adapter ]; then
  python scripts/train_lora_expert.py --config configs/experts/math.yaml
fi

if [ ! -d runs/experts/code/adapter ]; then
  python scripts/train_lora_expert.py --config configs/experts/code.yaml
fi

if [ ! -d runs/experts/medical/adapter ]; then
  python scripts/train_lora_expert.py --config configs/experts/medical.yaml
fi

if [ ! -d runs/experts/science/adapter ]; then
  python scripts/train_lora_expert.py --config configs/experts/science.yaml
fi

PROBES=${PROBES:-data/probes/interventions_pilot.jsonl}
if [ ! -s "$PROBES" ]; then
  python scripts/build_pilot_interventions.py --input data/distillation/pilot_prompts_with_domains.jsonl --output "$PROBES" --per-domain 250
fi
python scripts/validate_probe_pool.py "$PROBES"

if [ ! -f runs/pilot/response/base.npz ]; then
  python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes "$PROBES" --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/base.npz
fi
if [ ! -f runs/pilot/response/math.npz ]; then
  python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/math/adapter --probes "$PROBES" --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/math.npz
fi
if [ ! -f runs/pilot/response/code.npz ]; then
  python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/code/adapter --probes "$PROBES" --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/code.npz
fi
if [ ! -f runs/pilot/response/medical.npz ]; then
  python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/medical/adapter --probes "$PROBES" --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/medical.npz
fi
if [ ! -f runs/pilot/response/science.npz ]; then
  python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/science/adapter --probes "$PROBES" --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/science.npz
fi
python scripts/split_embedding_replicates.py runs/pilot/response/base.npz --select-output runs/pilot/response/base.select.npz --estimate-output runs/pilot/response/base.estimate.npz
python scripts/split_embedding_replicates.py runs/pilot/response/math.npz --select-output runs/pilot/response/math.select.npz --estimate-output runs/pilot/response/math.estimate.npz
python scripts/split_embedding_replicates.py runs/pilot/response/code.npz --select-output runs/pilot/response/code.select.npz --estimate-output runs/pilot/response/code.estimate.npz
python scripts/split_embedding_replicates.py runs/pilot/response/medical.npz --select-output runs/pilot/response/medical.select.npz --estimate-output runs/pilot/response/medical.estimate.npz
python scripts/split_embedding_replicates.py runs/pilot/response/science.npz --select-output runs/pilot/response/science.select.npz --estimate-output runs/pilot/response/science.estimate.npz

python scripts/build_response_bank.py --base runs/pilot/response/base.select.npz --ancestors runs/pilot/response/math.select.npz runs/pilot/response/code.select.npz runs/pilot/response/medical.select.npz runs/pilot/response/science.select.npz --names Math Code Medical Science --output runs/pilot/response/response_bank.select.npz
python scripts/build_response_bank.py --base runs/pilot/response/base.estimate.npz --ancestors runs/pilot/response/math.estimate.npz runs/pilot/response/code.estimate.npz runs/pilot/response/medical.estimate.npz runs/pilot/response/science.estimate.npz --names Math Code Medical Science --output runs/pilot/response/response_bank.estimate.npz
for B in 8 16 32 64 128; do python scripts/select_probes.py runs/pilot/response/response_bank.select.npz --budget "$B" --balanced --probe-jsonl "$PROBES" --group-key domain --output "runs/pilot/selected/probes_B${B}.json"; done
python scripts/subset_probe_pool.py --probes "$PROBES" --selected runs/pilot/selected/probes_B64.json --output runs/pilot/selected/probes_B64.jsonl
python scripts/subset_response_npz.py --input runs/pilot/response/base.estimate.npz --selected runs/pilot/selected/probes_B64.json --output runs/pilot/response/base_B64.estimate.npz


# Phase 1: exact functional-mixture death test (no full 4B checkpoint materialization required).
python scripts/generate_synthetic_response_targets.py --bank runs/pilot/response/response_bank.estimate.npz --output-dir runs/pilot/synthetic_targets --manifest runs/pilot/synthetic_targets.csv
python scripts/benchmark_probe_selection.py --selection-bank runs/pilot/response/response_bank.select.npz --bank runs/pilot/response/response_bank.estimate.npz --base runs/pilot/response/base.estimate.npz --targets runs/pilot/synthetic_targets.csv --budgets 8,16,32,64,128 --random-repeats 20 --output-curve results/raw/active_probe_curve.csv --output-geometry results/raw/geometry_runs.csv
python scripts/compute_geometry_mechanism.py results/raw/geometry_runs.csv --output results/derived/geometry_mechanism.csv
python scripts/sync_paper_results.py
python scripts/build_paper_assets.py

echo "Phase-1 functional-mixture death test complete. Inspect geometry/error mechanism before real merges."

# Phase 2 is optional: RUN_REAL_MERGES=1 materializes experts and queries real weight merges.
if [ "${RUN_REAL_MERGES:-0}" = "1" ]; then
  if [ ! -f runs/pilot/full/math/config.json ]; then
    python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/math/adapter --output runs/pilot/full/math
  fi
  if [ ! -f runs/pilot/full/code/config.json ]; then
    python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/code/adapter --output runs/pilot/full/code
  fi
  if [ ! -f runs/pilot/full/medical/config.json ]; then
    python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/medical/adapter --output runs/pilot/full/medical
  fi
  if [ ! -f runs/pilot/full/science/config.json ]; then
    python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/science/adapter --output runs/pilot/full/science
  fi
  if [ ! -f runs/pilot/merges/math_code/config.json ]; then
    python scripts/merge_linear_models.py --models runs/pilot/full/math runs/pilot/full/code --weights 0.5 0.5 --output runs/pilot/merges/math_code
  fi
  if [ ! -f runs/pilot/response/target_math_code_B64.npz ]; then
    python scripts/run_text_probe_bank.py --model runs/pilot/merges/math_code --probes runs/pilot/selected/probes_B64.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/target_math_code_B64.npz
  fi
  if [ ! -f runs/pilot/merges/math_medical/config.json ]; then
    python scripts/merge_linear_models.py --models runs/pilot/full/math runs/pilot/full/medical --weights 0.3 0.7 --output runs/pilot/merges/math_medical
  fi
  if [ ! -f runs/pilot/response/target_math_medical_B64.npz ]; then
    python scripts/run_text_probe_bank.py --model runs/pilot/merges/math_medical --probes runs/pilot/selected/probes_B64.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/target_math_medical_B64.npz
  fi
  if [ ! -f runs/pilot/merges/code_science/config.json ]; then
    python scripts/merge_linear_models.py --models runs/pilot/full/code runs/pilot/full/science --weights 0.7 0.3 --output runs/pilot/merges/code_science
  fi
  if [ ! -f runs/pilot/response/target_code_science_B64.npz ]; then
    python scripts/run_text_probe_bank.py --model runs/pilot/merges/code_science --probes runs/pilot/selected/probes_B64.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/pilot/response/target_code_science_B64.npz
  fi
  echo "Real merge checkpoints/responses prepared. Their construction weights are not treated as FAS coordinate ground truth; use support and counterfactual functional analyses."
fi
