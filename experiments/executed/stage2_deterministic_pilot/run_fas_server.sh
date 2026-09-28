#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${HOME}/FAS_TPAMI_SERVER_SHARED"
ENV_DIR="${SHARED_ROOT}/envs/fas-py311"
export PATH="${ENV_DIR}/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${SHARED_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_DIR="${SHARED_ROOT}/stage1_v3/project"
RUN_ROOT="${PROJECT_DIR}/runs/stage2_core/s11"
DET_ROOT="${RUN_ROOT}/deterministic_pilot_v1"
RETURN_ROOT="${SHARED_ROOT}/stage2_deterministic_pilot_v1_return"
RETURN_DIR="${RETURN_ROOT}/packet"
LOG_FILE="${RETURN_ROOT}/deterministic_pilot_v1.log"
STATUS="${RETURN_DIR}/status.json"
FINAL_TAR="${SCRIPT_DIR}/FAS_SERVER_RETURN_STAGE2_DETERMINISTIC_PILOT_V1.tar.gz"
SUCCESS=0
STAGE="bootstrap"

rm -rf "$RETURN_ROOT"
mkdir -p "$RETURN_DIR" "$DET_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

PY="${ENV_DIR}/bin/python"

write_status() {
  local code="$1"; local msg="$2"
  "$PY" - "$STATUS" "$code" "$STAGE" "$msg" <<'PY'
import json,sys,datetime,os
p,code,stage,msg=sys.argv[1:]
os.makedirs(os.path.dirname(p),exist_ok=True)
json.dump({
 "workflow":"FAS_STAGE2_DETERMINISTIC_PILOT_V1",
 "exit_code":int(code),"stage":stage,"message":msg,
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()
},open(p,"w"),indent=2)
PY
}

collect() {
  set +e
  cp -f "$LOG_FILE" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$DET_ROOT/results.json" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$RUN_ROOT/results/core_gate_results.json" "$RETURN_DIR/stochastic_core_gate_results.json" 2>/dev/null || true
  "$PY" - "$DET_ROOT" "$RETURN_DIR/progress.json" <<'PY'
import sys,json
from pathlib import Path
r=Path(sys.argv[1]); out=Path(sys.argv[2])
names=["base","math","code","medical","science","l1","l2"]
json.dump({n:(r/f"{n}.npz").exists() for n in names},open(out,"w"),indent=2)
PY
  tar -czf "$FINAL_TAR" -C "$RETURN_DIR" .
}

on_exit() {
  rc=$?
  set +e
  if [ "$SUCCESS" -eq 1 ] && [ -s "$DET_ROOT/results.json" ]; then
    rc=0; write_status 0 "Deterministic seed-11 measurement pilot completed."
  else
    [ "$rc" -ne 0 ] || rc=90
    write_status "$rc" "Pilot incomplete; rerun the same script to resume completed model-response chunks."
  fi
  collect
  echo
  echo "RETURN THIS SINGLE FILE:"
  echo "  $FINAL_TAR"
  exit "$rc"
}
trap on_exit EXIT

test -x "$PY" || { echo "Missing env"; exit 11; }
test -s "$RUN_ROOT/selections/union_probes.jsonl" || { echo "Missing union probes"; exit 12; }
for d in math code medical science; do
  test -s "$PROJECT_DIR/runs/experts/s11/$d/adapter/adapter_config.json" || { echo "Missing expert $d"; exit 13; }
done
test -s "$RUN_ROOT/l1_adapter/core_l1_s11/adapter_config.json" || { echo "Missing L1 target"; exit 14; }
test -s "$RUN_ROOT/l2_ties/model/config.json" || { echo "Missing L2 target"; exit 15; }

cp -f "$SCRIPT_DIR/deterministic_probe_runner.py" "$PROJECT_DIR/scripts/stage2_deterministic_probe_runner.py"
cp -f "$SCRIPT_DIR/evaluate_deterministic_pilot.py" "$PROJECT_DIR/scripts/stage2_evaluate_deterministic_pilot.py"
"$PY" -m py_compile "$PROJECT_DIR/scripts/stage2_deterministic_probe_runner.py" "$PROJECT_DIR/scripts/stage2_evaluate_deterministic_pilot.py"

STAGE="gpu_check"
"$PY" - <<'PY'
import torch
print("gpu",torch.cuda.get_device_name(0))
print("bf16",torch.cuda.is_bf16_supported())
assert torch.cuda.is_available()
PY

PROBES="$RUN_ROOT/selections/union_probes.jsonl"
ENC="sentence-transformers/all-mpnet-base-v2"

run_det () {
  local name="$1"; local model="$2"; local adapter="${3:-}"
  local out="$DET_ROOT/${name}.npz"; local work="$DET_ROOT/${name}_parts"
  if [ -s "$out" ]; then echo "SKIP COMPLETE $name"; return; fi
  cmd=("$PY" "$PROJECT_DIR/scripts/stage2_deterministic_probe_runner.py"
       --model "$model" --probes "$PROBES" --encoder "$ENC"
       --max-new-tokens 96 --batch-size 8 --chunk-size 100
       --work-dir "$work" --output "$out")
  if [ -n "$adapter" ]; then cmd+=(--adapter "$adapter"); fi
  "${cmd[@]}"
}

STAGE="deterministic_measurements"
run_det base Qwen/Qwen3-4B-Base
run_det math Qwen/Qwen3-4B-Base "$PROJECT_DIR/runs/experts/s11/math/adapter"
run_det code Qwen/Qwen3-4B-Base "$PROJECT_DIR/runs/experts/s11/code/adapter"
run_det medical Qwen/Qwen3-4B-Base "$PROJECT_DIR/runs/experts/s11/medical/adapter"
run_det science Qwen/Qwen3-4B-Base "$PROJECT_DIR/runs/experts/s11/science/adapter"
run_det l1 Qwen/Qwen3-4B-Base "$RUN_ROOT/l1_adapter/core_l1_s11"
run_det l2 "$RUN_ROOT/l2_ties/model"

STAGE="evaluate"
"$PY" "$PROJECT_DIR/scripts/stage2_evaluate_deterministic_pilot.py" \
  --run-root "$RUN_ROOT" --det-root "$DET_ROOT" --output "$DET_ROOT/results.json"

test -s "$DET_ROOT/results.json" || exit 20
SUCCESS=1
STAGE="complete"
