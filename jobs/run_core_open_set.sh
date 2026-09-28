#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/open_set/s11/{adapters,responses,inputs,outputs}

python scripts/subset_ancestor_bank.py runs/banks/s11/response_bank.select.npz --names Math,Code,Medical --output runs/open_set/s11/bank.select.npz
python scripts/subset_ancestor_bank.py runs/banks/s11/response_bank.estimate.npz --names Math,Code,Medical --output runs/open_set/s11/bank.estimate.npz
python scripts/select_probes.py runs/open_set/s11/bank.select.npz --budget 64 --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output runs/open_set/s11/selected.json
python scripts/subset_probe_pool.py --probes 'data/probes/interventions.jsonl' --selected runs/open_set/s11/selected.json --output runs/open_set/s11/selected_probes.jsonl

python scripts/generate_calibration_plan.py --config configs/calibration_open_set.yaml --parents Math,Code,Medical --seed 20260921 --output-dir runs/open_set/s11/cal_plan --g1-shift-n 0
python scripts/generate_calibration_commands.py --plan-dir runs/open_set/s11/cal_plan --bank runs/open_set/s11/bank.estimate.npz --selected runs/open_set/s11/selected.json --probes 'data/probes/interventions.jsonl' --run-root runs/open_set/s11/calibration --ancestor-root runs/experts/s11 --output runs/open_set/s11/run_calibration.sh
bash runs/open_set/s11/run_calibration.sh

# Build Known-only and Unknown-A targets. Science is deliberately withheld from the verifier bank.
cat > runs/open_set/s11/targets.csv <<'CSV'
calibration_id,parents,weights
known_only,"[""Math"",""Code"",""Medical""]","{""Math"":0.34,""Code"":0.33,""Medical"":0.33}"
unknown_A,"[""Math"",""Code"",""Science""]","{""Math"":0.34,""Code"":0.33,""Science"":0.33}"
CSV
python scripts/build_weighted_lora_adapters.py --base Qwen/Qwen3-4B-Base --plan runs/open_set/s11/targets.csv --adapter Math=runs/experts/s11/math/adapter --adapter Code=runs/experts/s11/code/adapter --adapter Medical=runs/experts/s11/medical/adapter --adapter Science=runs/experts/s11/science/adapter --output-dir runs/open_set/s11/adapters

AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}
python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes runs/open_set/s11/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed 50011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/open_set/s11/responses/base4b.npz
python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/open_set/s11/adapters/known_only --probes runs/open_set/s11/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed 51011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/open_set/s11/responses/known_only.npz
python scripts/prepare_target_decomposition.py --bank runs/open_set/s11/bank.estimate.npz --base runs/open_set/s11/responses/base4b.npz --target runs/open_set/s11/responses/known_only.npz --selected runs/open_set/s11/selected.json --target-selected-only --output runs/open_set/s11/inputs/known_only.npz
TAU=$(python -c "import json; print(json.load(open('runs/open_set/s11/calibration/scores/support_threshold.json'))['selected']['threshold'])")
python scripts/decompose_target.py runs/open_set/s11/inputs/known_only.npz --target-id L8-open-set__known_only__same__s11 --threshold "$TAU" --signal-calibration runs/open_set/s11/calibration/scores/signal_scores.csv --open-calibration runs/open_set/s11/calibration/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime matched --output runs/open_set/s11/outputs/L8-open-set__known_only__same__s11.decomposition.json

python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/open_set/s11/adapters/unknown_A --probes runs/open_set/s11/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed 51011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/open_set/s11/responses/unknown_A.npz
python scripts/prepare_target_decomposition.py --bank runs/open_set/s11/bank.estimate.npz --base runs/open_set/s11/responses/base4b.npz --target runs/open_set/s11/responses/unknown_A.npz --selected runs/open_set/s11/selected.json --target-selected-only --output runs/open_set/s11/inputs/unknown_A.npz
TAU=$(python -c "import json; print(json.load(open('runs/open_set/s11/calibration/scores/support_threshold.json'))['selected']['threshold'])")
python scripts/decompose_target.py runs/open_set/s11/inputs/unknown_A.npz --target-id L8-open-set__unknown_A__same__s11 --threshold "$TAU" --signal-calibration runs/open_set/s11/calibration/scores/signal_scores.csv --open-calibration runs/open_set/s11/calibration/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime matched --output runs/open_set/s11/outputs/L8-open-set__unknown_A__same__s11.decomposition.json

python scripts/run_text_probe_bank.py --model Qwen/Qwen3-1.7B-Base --adapter runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter --probes runs/open_set/s11/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed 52011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/open_set/s11/responses/unknown_B.npz
python scripts/run_text_probe_bank.py --model Qwen/Qwen3-1.7B-Base --probes runs/open_set/s11/selected_probes.jsonl --encoder sentence-transformers/all-mpnet-base-v2 --samples 4 --seed 53011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/open_set/s11/responses/base1p7b.npz
python scripts/prepare_target_decomposition.py --bank runs/open_set/s11/bank.estimate.npz --base runs/open_set/s11/responses/base1p7b.npz --target runs/open_set/s11/responses/unknown_B.npz --selected runs/open_set/s11/selected.json --target-selected-only --output runs/open_set/s11/inputs/unknown_B.npz
TAU=$(python -c "import json; print(json.load(open('runs/open_set/s11/calibration/scores/support_threshold.json'))['selected']['threshold'])")
python scripts/decompose_target.py runs/open_set/s11/inputs/unknown_B.npz --target-id L8-open-set__unknown_B_heavy_transform__same__s11 --threshold "$TAU" --signal-calibration runs/open_set/s11/calibration/scores/signal_scores.csv --open-calibration runs/open_set/s11/calibration/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output runs/open_set/s11/outputs/L8-open-set__unknown_B_heavy_transform__same__s11.decomposition.json

python scripts/collect_open_set_outputs.py "runs/open_set/s11/outputs/*.decomposition.json" --output results/raw/open_set_scores.csv
python scripts/compute_open_set_metrics.py results/raw/open_set_scores.csv
python scripts/sync_paper_results.py
python scripts/build_paper_assets.py
