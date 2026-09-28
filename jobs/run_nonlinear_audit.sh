#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Nonlinear/cross-scale runs are G1 calibration-shift tests by default.
# Set CALIBRATION_REGIME=matched only when calibration descendants are truly exchangeable with the test genealogy.
CAL_ROOT=${CAL_ROOT:-runs/calibration}
SIGNAL_CAL_1P7B=${SIGNAL_CAL_1P7B:-$CAL_ROOT/1p7b/signal_scores.csv}
OPEN_CAL_1P7B=${OPEN_CAL_1P7B:-$CAL_ROOT/1p7b/open_scores.csv}
SIGNAL_CAL_0P6B=${SIGNAL_CAL_0P6B:-$CAL_ROOT/0p6b/signal_scores.csv}
OPEN_CAL_0P6B=${OPEN_CAL_0P6B:-$CAL_ROOT/0p6b/open_scores.csv}
SUPPORT_TAU=${SUPPORT_TAU:-0.01}
CALIBRATION_REGIME=${CALIBRATION_REGIME:-g1_transfer}
AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}
PROBES='data/probes/interventions.jsonl'
SELECTED='runs/selected_probes.json'
BANK='runs/response_bank.estimate.npz'
mkdir -p runs/audit/{responses,inputs,outputs}

for f in "$SIGNAL_CAL_1P7B" "$OPEN_CAL_1P7B" "$SIGNAL_CAL_0P6B" "$OPEN_CAL_0P6B"; do test -s "$f" || { echo "missing $f"; exit 2; }; done
test -s "$SELECTED" || { echo "missing $SELECTED"; exit 2; }
test -s "$BANK" || { echo "missing $BANK"; exit 2; }
python scripts/subset_probe_pool.py --probes "$PROBES" --selected "$SELECTED" --output runs/audit/selected_probes.jsonl

if [ ! -f runs/audit/responses/base_1p7b.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-1.7B-Base --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/audit/responses/base_1p7b.npz; fi
if [ ! -f runs/audit/responses/base_0p6b.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-0.6B-Base --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/audit/responses/base_0p6b.npz; fi

echo "AUDIT L5-mixture-kd__v1__1p7b__s11"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s11/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s11.npz' --target-id 'L5-mixture-kd__v1__1p7b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s11' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v1__1p7b__s11/student/adapter'

echo "AUDIT L5-mixture-kd__v1__1p7b__s23"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s23/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s23.npz' --target-id 'L5-mixture-kd__v1__1p7b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s23' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v1__1p7b__s23/student/adapter'

echo "AUDIT L5-mixture-kd__v1__1p7b__s47"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s47/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__1p7b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__1p7b__s47.npz' --target-id 'L5-mixture-kd__v1__1p7b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s47' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v1__1p7b__s47/student/adapter'

echo "AUDIT L5-mixture-kd__v1__0p6b__s11"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s11/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s11.npz' --target-id 'L5-mixture-kd__v1__0p6b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__0p6b__s11.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s11' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v1__0p6b__s11/student/adapter'

echo "AUDIT L5-mixture-kd__v1__0p6b__s23"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s23/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s23.npz' --target-id 'L5-mixture-kd__v1__0p6b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__0p6b__s23.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s23' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v1__0p6b__s23/student/adapter'

echo "AUDIT L5-mixture-kd__v1__0p6b__s47"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s47/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v1__0p6b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v1__0p6b__s47.npz' --target-id 'L5-mixture-kd__v1__0p6b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v1__0p6b__s47.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s47' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v1__0p6b__s47/student/adapter'

echo "AUDIT L5-mixture-kd__v2__1p7b__s11"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s11.npz' --target-id 'L5-mixture-kd__v2__1p7b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s11' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter'

echo "AUDIT L5-mixture-kd__v2__1p7b__s23"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s23.npz' --target-id 'L5-mixture-kd__v2__1p7b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s23' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter'

echo "AUDIT L5-mixture-kd__v2__1p7b__s47"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__1p7b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__1p7b__s47.npz' --target-id 'L5-mixture-kd__v2__1p7b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s47' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter'

echo "AUDIT L5-mixture-kd__v2__0p6b__s11"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s11/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s11.npz' --target-id 'L5-mixture-kd__v2__0p6b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__0p6b__s11.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s11' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v2__0p6b__s11/student/adapter'

echo "AUDIT L5-mixture-kd__v2__0p6b__s23"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s23/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s23.npz' --target-id 'L5-mixture-kd__v2__0p6b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__0p6b__s23.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s23' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v2__0p6b__s23/student/adapter'

echo "AUDIT L5-mixture-kd__v2__0p6b__s47"
if [ ! -f 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s47/student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L5-mixture-kd__v2__0p6b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L5-mixture-kd__v2__0p6b__s47.npz' --target-id 'L5-mixture-kd__v2__0p6b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L5-mixture-kd__v2__0p6b__s47.decomposition.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s47' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L5-mixture-kd__v2__0p6b__s47/student/adapter'

echo "AUDIT L6-router-kd-sft__v1__1p7b__s11"
if [ ! -f 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_full' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_sft/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s11.npz' --target-id 'L6-router-kd-sft__v1__1p7b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L6-router-kd-sft__v1__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s11' done --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_full + runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_sft/adapter'

echo "AUDIT L6-router-kd-sft__v1__1p7b__s23"
if [ ! -f 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_full' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_sft/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s23.npz' --target-id 'L6-router-kd-sft__v1__1p7b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L6-router-kd-sft__v1__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s23' done --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_full + runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_sft/adapter'

echo "AUDIT L6-router-kd-sft__v1__1p7b__s47"
if [ ! -f 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_full' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_sft/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L6-router-kd-sft__v1__1p7b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L6-router-kd-sft__v1__1p7b__s47.npz' --target-id 'L6-router-kd-sft__v1__1p7b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L6-router-kd-sft__v1__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s47' done --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_full + runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_sft/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s11"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s11.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__1p7b__s11.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s11' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s23"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s23.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__1p7b__s23.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s23' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__1p7b__s47"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_1p7b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__1p7b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__1p7b__s47.npz' --target-id 'L7-deep-ancestry__v1__1p7b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_1P7B" --open-calibration "$OPEN_CAL_1P7B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__1p7b__s47.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s47' done --checkpoint-or-api 'Qwen/Qwen3-1.7B-Base + runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__0p6b__s11"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s11.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s11.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s11.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s11.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s11.npz' --target-id 'L7-deep-ancestry__v1__0p6b__s11' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__0p6b__s11.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s11' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__0p6b__s23"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s23.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s23.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s23.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s23.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s23.npz' --target-id 'L7-deep-ancestry__v1__0p6b__s23' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__0p6b__s23.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s23' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23/deep_student/adapter'

echo "AUDIT L7-deep-ancestry__v1__0p6b__s47"
if [ ! -f 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s47.npz' ]; then python scripts/run_text_probe_bank.py --model 'Qwen/Qwen3-0.6B-Base' --adapter 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47/deep_student/adapter' --probes runs/audit/selected_probes.jsonl --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s47.npz'; fi
python scripts/prepare_target_decomposition.py --bank "$BANK" --base 'runs/audit/responses/base_0p6b.npz' --target 'runs/audit/responses/L7-deep-ancestry__v1__0p6b__s47.npz' --selected "$SELECTED" --target-selected-only --output 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s47.npz'
python scripts/decompose_target.py 'runs/audit/inputs/L7-deep-ancestry__v1__0p6b__s47.npz' --target-id 'L7-deep-ancestry__v1__0p6b__s47' --threshold "$SUPPORT_TAU" --signal-calibration "$SIGNAL_CAL_0P6B" --open-calibration "$OPEN_CAL_0P6B" --alpha-signal 0.05 --alpha-open 0.05 --calibration-regime "$CALIBRATION_REGIME" --output 'runs/audit/outputs/L7-deep-ancestry__v1__0p6b__s47.decomposition.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s47' done --checkpoint-or-api 'Qwen/Qwen3-0.6B-Base + runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47/deep_student/adapter'

python scripts/build_target_truth.py --selected "$SELECTED" --samples 4
python scripts/collect_decomposition_jsons.py --truth data/metadata/target_truth.csv --input-dir runs/audit/outputs --output results/raw/decomposition_predictions.csv
python scripts/evaluate_decompositions.py results/raw/decomposition_predictions.csv
python scripts/sync_paper_results.py
python scripts/build_paper_assets.py
echo "Audit/results synchronization complete."
