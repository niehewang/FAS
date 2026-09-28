#!/usr/bin/env bash
FAS_WORK_ROOT="${FAS_WORK_ROOT:-$PWD}"
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${HOME}/FAS_TPAMI_SERVER_SHARED"
ENV_DIR="$SHARED_ROOT/envs/fas-py311"
PROJECT_DIR="$SHARED_ROOT/stage1_v3/project"
PY="$ENV_DIR/bin/python"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"
STAGE5="$PROJECT_DIR/runs/fas_stage5_l5_mixture_kd_v1"
STAGE5E="$PROJECT_DIR/runs/fas_stage5e_scalematched17_pilot_v1"
RUN_ROOT="$PROJECT_DIR/runs/fas_stage5f_scalematched17_multiseed_v1"
RETURN_DIR="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5F_SCALEMATCHED17_MULTISEED_V1"
FINAL="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5F_SCALEMATCHED17_MULTISEED_V1.tar.gz"
LOG="$RUN_ROOT/stage5f.log"
export PATH="$ENV_DIR/bin:$PATH" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 HF_HOME="$SHARED_ROOT/hf_cache" HUGGINGFACE_HUB_CACHE="$SHARED_ROOT/hf_cache/hub" DATASETS_CACHE="$SHARED_ROOT/hf_cache/datasets" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SUCCESS=0; STAGE=bootstrap
mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_DIR"; mkdir -p "$RETURN_DIR"
exec > >(tee -a "$LOG") 2>&1
status(){ "$PY" - "$RETURN_DIR/status.json" "$1" "$STAGE" "$2" <<'PY'
import sys,json,datetime
json.dump({'workflow':'FAS_STAGE5F_SCALEMATCHED17_MULTISEED_V1','exit_code':int(sys.argv[2]),'stage':sys.argv[3],'message':sys.argv[4],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
}
collect(){ set +e
  cp -f "$SCRIPT_DIR/DIAGNOSTIC_FREEZE_STAGE5F_V1.json" "$RETURN_DIR/"
  cp -f "$STAGE5E/evaluation.json" "$RETURN_DIR/evaluation_s23_reference.json" 2>/dev/null || true
  for s in 47 71; do
    cp -f "$RUN_ROOT/s${s}/evaluation.json" "$RETURN_DIR/evaluation_s${s}.json" 2>/dev/null || true
    mkdir -p "$RETURN_DIR/s${s}/experts"
    for d in math code medical science; do
      mkdir -p "$RETURN_DIR/s${s}/experts/$d"
      cp -f "$RUN_ROOT/s${s}/experts/$d/training_metadata.json" "$RETURN_DIR/s${s}/experts/$d/" 2>/dev/null || true
      cp -f "$RUN_ROOT/s${s}/experts/$d/config.yaml" "$RETURN_DIR/s${s}/experts/$d/" 2>/dev/null || true
      cp -f "$RUN_ROOT/s${s}/experts/$d/response.npz" "$RETURN_DIR/s${s}/experts/$d/" 2>/dev/null || true
    done
  done
  cp -f "$RUN_ROOT/aggregate_stage5f.json" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$LOG" "$RETURN_DIR/" 2>/dev/null || true
  df -hT / /data > "$RETURN_DIR/disk_usage.txt" 2>/dev/null || true
  tar -czf "$FINAL" -C "$RETURN_DIR" .
  echo "RETURN_THIS_SINGLE_FILE=$FINAL"
}
trap 'rc=$?; set +e; if [ "$SUCCESS" = 1 ]; then rc=0; status 0 "Stage5F scale-matched 1.7B multiseed diagnostic completed."; else [ "$rc" != 0 ] || rc=90; status "$rc" "Stage5F incomplete; rerun the same package to resume completed expert/checkpoint/response artifacts."; fi; collect; exit $rc' EXIT

STAGE=preflight
for p in "$PY" "$STAGE3/shared/base_full.npz" "$STAGE5/shared/base17.npz" "$STAGE5E/evaluation.json"; do [ -s "$p" ] || { echo "MISSING:$p"; exit 21; }; done
for s in 47 71; do
  for p in "$STAGE5/s${s}/student_response.npz" "$STAGE5/s${s}/evaluation.json" "$STAGE5/s${s}/audit_union.jsonl"; do [ -s "$p" ] || { echo "MISSING:$p"; exit 22; }; done
  for strat in active random; do [ -s "$STAGE3/s${s}/selections/${strat}_B32.json" ] || { echo "MISSING selection:$s:$strat"; exit 23; }; done
done
for d in math code medical science; do [ -s "$PROJECT_DIR/data/training/formal/$d.jsonl" ] || { echo "MISSING formal data:$d"; exit 24; }; done
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(); print('GPU',torch.cuda.get_device_name(0),'torch',torch.__version__)
PY
"$PY" - "$PROJECT_DIR" <<'PY'
import json,sys
from pathlib import Path
P=Path(sys.argv[1])
for d in ['math','code','medical','science']:
 p=P/'data/training/formal'/f'{d}.jsonl'; n=0
 with p.open(encoding='utf-8') as f:
  for line in f:
   if not line.strip(): continue
   r=json.loads(line); assert 'prompt' in r and 'response' in r, (d,r.keys()); n+=1
 assert n==3000,(d,n)
 print(d,n)
PY

for s in 47 71; do
  for d in math code medical science; do
    DOUT="$RUN_ROOT/s${s}/experts/$d"; mkdir -p "$DOUT"
    STAGE="train_${d}_s${s}"
    if [ ! -s "$DOUT/adapter/adapter_config.json" ]; then
      resume=""
      if compgen -G "$DOUT/checkpoint-*" >/dev/null; then resume=$(ls -d "$DOUT"/checkpoint-* | sort -V | tail -1); fi
      args=("$PY" "$SCRIPT_DIR/make_expert17_config_v1.py" --seed "$s" --domain "${d^}" --data "$PROJECT_DIR/data/training/formal/$d.jsonl" --output-dir "$DOUT" --config "$DOUT/config.yaml")
      [ -n "$resume" ] && args+=(--resume "$resume")
      "${args[@]}"
      "$PY" "$SCRIPT_DIR/train_lora_expert_v1.py" --config "$DOUT/config.yaml"
    fi
    STAGE="response_${d}_s${s}"
    if [ ! -s "$DOUT/response.npz" ]; then
      "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v2.py" --model Qwen/Qwen3-1.7B-Base --adapter "$DOUT/adapter" --probes "$STAGE5/s${s}/audit_union.jsonl" --encoder sentence-transformers/all-mpnet-base-v2 --max-new-tokens 96 --batch-size 8 --chunk-size 100 --work-dir "$DOUT/response_parts" --output "$DOUT/response.npz"
      rm -rf "$DOUT/response_parts"
    fi
  done
  STAGE="evaluate_s${s}"
  "$PY" "$SCRIPT_DIR/evaluate_scalematched17_v1.py" --stage3 "$STAGE3" --stage5 "$STAGE5" --run-root "$RUN_ROOT" --seed "$s" --output "$RUN_ROOT/s${s}/evaluation.json"
done
STAGE=aggregate
"$PY" "$SCRIPT_DIR/aggregate_stage5f_v1.py" --seed23-eval "$STAGE5E/evaluation.json" --seed47-eval "$RUN_ROOT/s47/evaluation.json" --seed71-eval "$RUN_ROOT/s71/evaluation.json" --output "$RUN_ROOT/aggregate_stage5f.json"
SUCCESS=1; STAGE=complete
