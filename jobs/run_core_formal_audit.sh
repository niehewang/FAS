#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Formal audit: seed-matched ancestor dictionary and active probe subset.
# L1/L2 use G0-transfer diagnostics; postprocessed/KD/deep runs use G1-transfer by default.
AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}
mkdir -p runs/formal_audit

test -s runs/banks/s11/selected_probes.json || { echo "missing seed-matched selected probes s11"; exit 2; }
test -s runs/banks/s11/response_bank.estimate.npz || { echo "missing seed-matched estimate bank s11"; exit 2; }
test -s runs/calibration/s11/scores/signal_scores.csv || { echo "missing G0 signal calibration s11"; exit 2; }
test -s runs/calibration/s11/scores/open_scores.csv || { echo "missing G0 open calibration s11"; exit 2; }
mkdir -p runs/formal_audit/s11/{responses,inputs,outputs}
python scripts/subset_probe_pool.py --probes 'data/probes/interventions.jsonl' --selected runs/banks/s11/selected_probes.json --output runs/formal_audit/s11/selected_probes.jsonl

test -s runs/banks/s23/selected_probes.json || { echo "missing seed-matched selected probes s23"; exit 2; }
test -s runs/banks/s23/response_bank.estimate.npz || { echo "missing seed-matched estimate bank s23"; exit 2; }
test -s runs/calibration/s23/scores/signal_scores.csv || { echo "missing G0 signal calibration s23"; exit 2; }
test -s runs/calibration/s23/scores/open_scores.csv || { echo "missing G0 open calibration s23"; exit 2; }
mkdir -p runs/formal_audit/s23/{responses,inputs,outputs}
python scripts/subset_probe_pool.py --probes 'data/probes/interventions.jsonl' --selected runs/banks/s23/selected_probes.json --output runs/formal_audit/s23/selected_probes.jsonl

test -s runs/banks/s47/selected_probes.json || { echo "missing seed-matched selected probes s47"; exit 2; }
test -s runs/banks/s47/response_bank.estimate.npz || { echo "missing seed-matched estimate bank s47"; exit 2; }
test -s runs/calibration/s47/scores/signal_scores.csv || { echo "missing G0 signal calibration s47"; exit 2; }
test -s runs/calibration/s47/scores/open_scores.csv || { echo "missing G0 open calibration s47"; exit 2; }
mkdir -p runs/formal_audit/s47/{responses,inputs,outputs}
python scripts/subset_probe_pool.py --probes 'data/probes/interventions.jsonl' --selected runs/banks/s47/selected_probes.json --output runs/formal_audit/s47/selected_probes.jsonl

if [ ! -s runs/formal_audit/s11/responses/base_4b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30011 --output runs/formal_audit/s11/responses/base_4b.npz; fi

if [ ! -s runs/formal_audit/s23/responses/base_4b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30023 --output runs/formal_audit/s23/responses/base_4b.npz; fi

if [ ! -s runs/formal_audit/s47/responses/base_4b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-4B-Base' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30047 --output runs/formal_audit/s47/responses/base_4b.npz; fi

if [ ! -s runs/formal_audit/s11/responses/base_1p7b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30011 --output runs/formal_audit/s11/responses/base_1p7b.npz; fi

if [ ! -s runs/formal_audit/s23/responses/base_1p7b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30023 --output runs/formal_audit/s23/responses/base_1p7b.npz; fi

if [ ! -s runs/formal_audit/s47/responses/base_1p7b.npz ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 30047 --output runs/formal_audit/s47/responses/base_1p7b.npz; fi

echo "AUDIT L1-linear-2p__v1__same__s11"
test -s runs/calibration/s11/scores/support_threshold.json || { echo "missing support threshold s11"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s11/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s11/responses/L1-linear-2p__v1__same__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L1-linear-2p__v1__same__s11/model' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40011 --output 'runs/formal_audit/s11/responses/L1-linear-2p__v1__same__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s11/response_bank.estimate.npz --base runs/formal_audit/s11/responses/base_4b.npz --target 'runs/formal_audit/s11/responses/L1-linear-2p__v1__same__s11.npz' --selected runs/banks/s11/selected_probes.json --target-selected-only --output 'runs/formal_audit/s11/inputs/L1-linear-2p__v1__same__s11.npz'
python scripts/decompose_target.py 'runs/formal_audit/s11/inputs/L1-linear-2p__v1__same__s11.npz' --target-id 'L1-linear-2p__v1__same__s11' --threshold "$TAU" --signal-calibration runs/calibration/s11/scores/signal_scores.csv --open-calibration runs/calibration/s11/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s11/outputs/L1-linear-2p__v1__same__s11.decomposition.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s11' done --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s11/model'

echo "AUDIT L1-linear-2p__v1__same__s23"
test -s runs/calibration/s23/scores/support_threshold.json || { echo "missing support threshold s23"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s23/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s23/responses/L1-linear-2p__v1__same__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L1-linear-2p__v1__same__s23/model' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40023 --output 'runs/formal_audit/s23/responses/L1-linear-2p__v1__same__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s23/response_bank.estimate.npz --base runs/formal_audit/s23/responses/base_4b.npz --target 'runs/formal_audit/s23/responses/L1-linear-2p__v1__same__s23.npz' --selected runs/banks/s23/selected_probes.json --target-selected-only --output 'runs/formal_audit/s23/inputs/L1-linear-2p__v1__same__s23.npz'
python scripts/decompose_target.py 'runs/formal_audit/s23/inputs/L1-linear-2p__v1__same__s23.npz' --target-id 'L1-linear-2p__v1__same__s23' --threshold "$TAU" --signal-calibration runs/calibration/s23/scores/signal_scores.csv --open-calibration runs/calibration/s23/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s23/outputs/L1-linear-2p__v1__same__s23.decomposition.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s23' done --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s23/model'

echo "AUDIT L1-linear-2p__v1__same__s47"
test -s runs/calibration/s47/scores/support_threshold.json || { echo "missing support threshold s47"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s47/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s47/responses/L1-linear-2p__v1__same__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L1-linear-2p__v1__same__s47/model' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40047 --output 'runs/formal_audit/s47/responses/L1-linear-2p__v1__same__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s47/response_bank.estimate.npz --base runs/formal_audit/s47/responses/base_4b.npz --target 'runs/formal_audit/s47/responses/L1-linear-2p__v1__same__s47.npz' --selected runs/banks/s47/selected_probes.json --target-selected-only --output 'runs/formal_audit/s47/inputs/L1-linear-2p__v1__same__s47.npz'
python scripts/decompose_target.py 'runs/formal_audit/s47/inputs/L1-linear-2p__v1__same__s47.npz' --target-id 'L1-linear-2p__v1__same__s47' --threshold "$TAU" --signal-calibration runs/calibration/s47/scores/signal_scores.csv --open-calibration runs/calibration/s47/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s47/outputs/L1-linear-2p__v1__same__s47.decomposition.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s47' done --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s47/model'

echo "AUDIT L2-ties-3p__v1__same__s11"
test -s runs/calibration/s11/scores/support_threshold.json || { echo "missing support threshold s11"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s11/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s11/responses/L2-ties-3p__v1__same__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L2-ties-3p__v1__same__s11/model' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40011 --output 'runs/formal_audit/s11/responses/L2-ties-3p__v1__same__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s11/response_bank.estimate.npz --base runs/formal_audit/s11/responses/base_4b.npz --target 'runs/formal_audit/s11/responses/L2-ties-3p__v1__same__s11.npz' --selected runs/banks/s11/selected_probes.json --target-selected-only --output 'runs/formal_audit/s11/inputs/L2-ties-3p__v1__same__s11.npz'
python scripts/decompose_target.py 'runs/formal_audit/s11/inputs/L2-ties-3p__v1__same__s11.npz' --target-id 'L2-ties-3p__v1__same__s11' --threshold "$TAU" --signal-calibration runs/calibration/s11/scores/signal_scores.csv --open-calibration runs/calibration/s11/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s11/outputs/L2-ties-3p__v1__same__s11.decomposition.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s11' done --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s11/model'

echo "AUDIT L2-ties-3p__v1__same__s23"
test -s runs/calibration/s23/scores/support_threshold.json || { echo "missing support threshold s23"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s23/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s23/responses/L2-ties-3p__v1__same__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L2-ties-3p__v1__same__s23/model' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40023 --output 'runs/formal_audit/s23/responses/L2-ties-3p__v1__same__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s23/response_bank.estimate.npz --base runs/formal_audit/s23/responses/base_4b.npz --target 'runs/formal_audit/s23/responses/L2-ties-3p__v1__same__s23.npz' --selected runs/banks/s23/selected_probes.json --target-selected-only --output 'runs/formal_audit/s23/inputs/L2-ties-3p__v1__same__s23.npz'
python scripts/decompose_target.py 'runs/formal_audit/s23/inputs/L2-ties-3p__v1__same__s23.npz' --target-id 'L2-ties-3p__v1__same__s23' --threshold "$TAU" --signal-calibration runs/calibration/s23/scores/signal_scores.csv --open-calibration runs/calibration/s23/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s23/outputs/L2-ties-3p__v1__same__s23.decomposition.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s23' done --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s23/model'

echo "AUDIT L2-ties-3p__v1__same__s47"
test -s runs/calibration/s47/scores/support_threshold.json || { echo "missing support threshold s47"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s47/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s47/responses/L2-ties-3p__v1__same__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L2-ties-3p__v1__same__s47/model' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40047 --output 'runs/formal_audit/s47/responses/L2-ties-3p__v1__same__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s47/response_bank.estimate.npz --base runs/formal_audit/s47/responses/base_4b.npz --target 'runs/formal_audit/s47/responses/L2-ties-3p__v1__same__s47.npz' --selected runs/banks/s47/selected_probes.json --target-selected-only --output 'runs/formal_audit/s47/inputs/L2-ties-3p__v1__same__s47.npz'
python scripts/decompose_target.py 'runs/formal_audit/s47/inputs/L2-ties-3p__v1__same__s47.npz' --target-id 'L2-ties-3p__v1__same__s47' --threshold "$TAU" --signal-calibration runs/calibration/s47/scores/signal_scores.csv --open-calibration runs/calibration/s47/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g0_transfer --output 'runs/formal_audit/s47/outputs/L2-ties-3p__v1__same__s47.decomposition.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s47' done --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s47/model'

echo "AUDIT L5-mixture-kd__v2__1p7b__s11"
test -s runs/calibration/s11/scores/support_threshold.json || { echo "missing support threshold s11"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s11/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s11/responses/L5-mixture-kd__v2__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40011 --output 'runs/formal_audit/s11/responses/L5-mixture-kd__v2__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s11/response_bank.estimate.npz --base runs/formal_audit/s11/responses/base_1p7b.npz --target 'runs/formal_audit/s11/responses/L5-mixture-kd__v2__1p7b__s11.npz' --selected runs/banks/s11/selected_probes.json --target-selected-only --output 'runs/formal_audit/s11/inputs/L5-mixture-kd__v2__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/formal_audit/s11/inputs/L5-mixture-kd__v2__1p7b__s11.npz' --target-id 'L5-mixture-kd__v2__1p7b__s11' --threshold "$TAU" --signal-calibration runs/calibration/s11/scores/signal_scores.csv --open-calibration runs/calibration/s11/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s11/outputs/L5-mixture-kd__v2__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s11' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter'

echo "AUDIT L5-mixture-kd__v2__1p7b__s23"
test -s runs/calibration/s23/scores/support_threshold.json || { echo "missing support threshold s23"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s23/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s23/responses/L5-mixture-kd__v2__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40023 --output 'runs/formal_audit/s23/responses/L5-mixture-kd__v2__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s23/response_bank.estimate.npz --base runs/formal_audit/s23/responses/base_1p7b.npz --target 'runs/formal_audit/s23/responses/L5-mixture-kd__v2__1p7b__s23.npz' --selected runs/banks/s23/selected_probes.json --target-selected-only --output 'runs/formal_audit/s23/inputs/L5-mixture-kd__v2__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/formal_audit/s23/inputs/L5-mixture-kd__v2__1p7b__s23.npz' --target-id 'L5-mixture-kd__v2__1p7b__s23' --threshold "$TAU" --signal-calibration runs/calibration/s23/scores/signal_scores.csv --open-calibration runs/calibration/s23/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s23/outputs/L5-mixture-kd__v2__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s23' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter'

echo "AUDIT L5-mixture-kd__v2__1p7b__s47"
test -s runs/calibration/s47/scores/support_threshold.json || { echo "missing support threshold s47"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s47/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s47/responses/L5-mixture-kd__v2__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40047 --output 'runs/formal_audit/s47/responses/L5-mixture-kd__v2__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s47/response_bank.estimate.npz --base runs/formal_audit/s47/responses/base_1p7b.npz --target 'runs/formal_audit/s47/responses/L5-mixture-kd__v2__1p7b__s47.npz' --selected runs/banks/s47/selected_probes.json --target-selected-only --output 'runs/formal_audit/s47/inputs/L5-mixture-kd__v2__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/formal_audit/s47/inputs/L5-mixture-kd__v2__1p7b__s47.npz' --target-id 'L5-mixture-kd__v2__1p7b__s47' --threshold "$TAU" --signal-calibration runs/calibration/s47/scores/signal_scores.csv --open-calibration runs/calibration/s47/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s47/outputs/L5-mixture-kd__v2__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s47' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s11"
test -s runs/calibration/s11/scores/support_threshold.json || { echo "missing support threshold s11"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s11/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s11/responses/L7-deep-ancestry__v1__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' --probes runs/formal_audit/s11/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40011 --output 'runs/formal_audit/s11/responses/L7-deep-ancestry__v1__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s11/response_bank.estimate.npz --base runs/formal_audit/s11/responses/base_1p7b.npz --target 'runs/formal_audit/s11/responses/L7-deep-ancestry__v1__1p7b__s11.npz' --selected runs/banks/s11/selected_probes.json --target-selected-only --output 'runs/formal_audit/s11/inputs/L7-deep-ancestry__v1__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/formal_audit/s11/inputs/L7-deep-ancestry__v1__1p7b__s11.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s11' --threshold "$TAU" --signal-calibration runs/calibration/s11/scores/signal_scores.csv --open-calibration runs/calibration/s11/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s11/outputs/L7-deep-ancestry__v1__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s11' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s23"
test -s runs/calibration/s23/scores/support_threshold.json || { echo "missing support threshold s23"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s23/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s23/responses/L7-deep-ancestry__v1__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' --probes runs/formal_audit/s23/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40023 --output 'runs/formal_audit/s23/responses/L7-deep-ancestry__v1__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s23/response_bank.estimate.npz --base runs/formal_audit/s23/responses/base_1p7b.npz --target 'runs/formal_audit/s23/responses/L7-deep-ancestry__v1__1p7b__s23.npz' --selected runs/banks/s23/selected_probes.json --target-selected-only --output 'runs/formal_audit/s23/inputs/L7-deep-ancestry__v1__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/formal_audit/s23/inputs/L7-deep-ancestry__v1__1p7b__s23.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s23' --threshold "$TAU" --signal-calibration runs/calibration/s23/scores/signal_scores.csv --open-calibration runs/calibration/s23/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s23/outputs/L7-deep-ancestry__v1__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s23' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s47"
test -s runs/calibration/s47/scores/support_threshold.json || { echo "missing support threshold s47"; exit 2; }
TAU=$(python -c "import json; print(json.load(open('runs/calibration/s47/scores/support_threshold.json'))['selected']['threshold'])")
if [ ! -s 'runs/formal_audit/s47/responses/L7-deep-ancestry__v1__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' --probes runs/formal_audit/s47/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --seed 40047 --output 'runs/formal_audit/s47/responses/L7-deep-ancestry__v1__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank runs/banks/s47/response_bank.estimate.npz --base runs/formal_audit/s47/responses/base_1p7b.npz --target 'runs/formal_audit/s47/responses/L7-deep-ancestry__v1__1p7b__s47.npz' --selected runs/banks/s47/selected_probes.json --target-selected-only --output 'runs/formal_audit/s47/inputs/L7-deep-ancestry__v1__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/formal_audit/s47/inputs/L7-deep-ancestry__v1__1p7b__s47.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s47' --threshold "$TAU" --signal-calibration runs/calibration/s47/scores/signal_scores.csv --open-calibration runs/calibration/s47/scores/open_scores.csv --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime g1_transfer --output 'runs/formal_audit/s47/outputs/L7-deep-ancestry__v1__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s47' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter'

echo "Formal seed-matched audit complete. Collect JSONs with collect_decomposition_jsons.py."
