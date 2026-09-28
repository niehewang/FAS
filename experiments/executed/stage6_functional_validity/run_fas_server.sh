#!/usr/bin/env bash
FAS_WORK_ROOT="${FAS_WORK_ROOT:-$PWD}"
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${HOME}/FAS_TPAMI_SERVER_SHARED"
ENV_DIR="$SHARED_ROOT/envs/fas-py311"
PROJECT_DIR="$SHARED_ROOT/stage1_v3/project"
PY="$ENV_DIR/bin/python"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"
RUN_ROOT="$PROJECT_DIR/runs/fas_stage6_functional_validity_v1_1"
RETURN_DIR="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE6_FUNCTIONAL_VALIDITY_V1_1"
FINAL="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE6_FUNCTIONAL_VALIDITY_V1_1.tar.gz"
LOG="$RUN_ROOT/stage6_v1_1.log"
BASE_MODEL="Qwen/Qwen3-4B-Base"
PROBES="$PROJECT_DIR/data/probes/interventions.jsonl"
SEED_POOL="$PROJECT_DIR/data/probes/formal_seed_pool.jsonl"
PROBE_MANIFEST="$PROJECT_DIR/data/governance/formal_probe_seed_manifest.json"
DATA_LOCK="$PROJECT_DIR/data/governance/formal_data_lock.json"
TRAIN_DIR="$PROJECT_DIR/data/training/formal"
ENCODER="sentence-transformers/all-mpnet-base-v2"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
SUCCESS=0; STAGE=bootstrap
mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_DIR"; mkdir -p "$RETURN_DIR"
exec > >(tee -a "$LOG") 2>&1
status(){ "$PY" - "$RETURN_DIR/status.json" "$1" "$STAGE" "$2" <<'PY'
import sys,json,datetime
json.dump({'workflow':'FAS_STAGE6_FUNCTIONAL_VALIDITY_EXACT_SHAPLEY_V1_1','exit_code':int(sys.argv[2]),'stage':sys.argv[3],'message':sys.argv[4],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
}
collect(){ set +e
  cp -f "$SCRIPT_DIR/FUNCTIONAL_VALIDITY_FREEZE_STAGE6_V1_1.json" "$RETURN_DIR/"
  cp -f "$RUN_ROOT/heldout_utility_manifest.json" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$RUN_ROOT/heldout_utility.jsonl" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$RUN_ROOT/aggregate_stage6.json" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$RUN_ROOT/functional_validity_rows.csv" "$RETURN_DIR/" 2>/dev/null || true
  for s in 23 47 71; do mkdir -p "$RETURN_DIR/s$s"; cp -f "$RUN_ROOT/s$s/utility/subset_utilities.csv" "$RETURN_DIR/s$s/" 2>/dev/null || true; cp -f "$RUN_ROOT/s$s/coordinates.json" "$RETURN_DIR/s$s/" 2>/dev/null || true; cp -f "$RUN_ROOT/s$s/plan.csv" "$RETURN_DIR/s$s/" 2>/dev/null || true; done
  cp -f "$LOG" "$RETURN_DIR/" 2>/dev/null || true
  df -hT / /data > "$RETURN_DIR/disk_usage.txt" 2>/dev/null || true
  tar -czf "$FINAL" -C "$RETURN_DIR" .
  echo "RETURN_THIS_SINGLE_FILE=$FINAL"
}
trap 'rc=$?; set +e; if [ "$SUCCESS" = 1 ]; then rc=0; status 0 "Stage6 v1.1 functional-validity exact-Shapley completed."; else [ "$rc" != 0 ] || rc=90; status "$rc" "Stage6 v1.1 incomplete; prior stages and expert checkpoints were not modified."; fi; collect; exit $rc' EXIT

STAGE=preflight
for p in "$PY" "$PROBES" "$SEED_POOL" "$PROBE_MANIFEST" "$DATA_LOCK" "$TRAIN_DIR/math.jsonl" "$TRAIN_DIR/medical.jsonl" "$TRAIN_DIR/science.jsonl" "$STAGE3/shared/base_full.npz" "$PROJECT_DIR/scripts/build_weighted_lora_adapters.py" "$PROJECT_DIR/scripts/select_domain_probes.py" "$PROJECT_DIR/scripts/subset_probe_pool.py"; do [ -s "$p" ] || { echo "MISSING:$p"; exit 21; }; done
for s in 23 47 71; do for d in math code medical science; do [ -s "$PROJECT_DIR/runs/experts/s$s/$d/adapter/adapter_config.json" ] || { echo "MISSING expert s$s $d"; exit 22; }; [ -s "$STAGE3/s$s/banks/$d.npz" ] || { echo "MISSING bank s$s $d"; exit 23; }; done; done
for f in prepare_formal_utility_holdout_v1.py make_subset_plan.py build_combined_bank.py evaluate_subset_utilities_multi.py run_pair_probe_bank.py compute_stage6_coordinates.py aggregate_stage6.py; do "$PY" -m py_compile "$SCRIPT_DIR/$f"; done

STAGE=prepare_formal_heldout_utility
if [ ! -s "$RUN_ROOT/heldout_utility.jsonl" ]; then
  "$PY" "$SCRIPT_DIR/prepare_formal_utility_holdout_v1.py" --seed-pool "$SEED_POOL" --probe-manifest "$PROBE_MANIFEST" --data-lock "$DATA_LOCK" --training-dir "$TRAIN_DIR" --output "$RUN_ROOT/heldout_utility.jsonl" --manifest "$RUN_ROOT/heldout_utility_manifest.json" --n-per-domain 96 --selection-seed 20260926_stage6_v1_1
fi

for s in 23 47 71; do
  SD="$RUN_ROOT/s$s"; mkdir -p "$SD"/{adapters,utility,selected,responses}
  STAGE="seed_${s}_plan"; [ -s "$SD/plan.csv" ] || "$PY" "$SCRIPT_DIR/make_subset_plan.py" --output "$SD/plan.csv"
  STAGE="seed_${s}_build_subsets"
  if [ ! -s "$SD/adapters/subset_Math_Medical_Science/adapter_config.json" ]; then
    "$PY" "$PROJECT_DIR/scripts/build_weighted_lora_adapters.py" --base "$BASE_MODEL" --plan "$SD/plan.csv" --adapter Math="$PROJECT_DIR/runs/experts/s$s/math/adapter" --adapter Code="$PROJECT_DIR/runs/experts/s$s/code/adapter" --adapter Medical="$PROJECT_DIR/runs/experts/s$s/medical/adapter" --adapter Science="$PROJECT_DIR/runs/experts/s$s/science/adapter" --output-dir "$SD/adapters"
  fi
  STAGE="seed_${s}_counterfactual_utilities"
  if [ ! -s "$SD/utility/subset_utilities.csv" ]; then
    "$PY" "$SCRIPT_DIR/evaluate_subset_utilities_multi.py" --base "$BASE_MODEL" --plan "$SD/plan.csv" --adapter-root "$SD/adapters" --data "$RUN_ROOT/heldout_utility.jsonl" --output "$SD/utility/subset_utilities.csv" --details-dir "$SD/utility/predictions" --max-new-tokens 96 --batch-size 8
  fi
  STAGE="seed_${s}_domain_probe_selection"
  [ -s "$SD/combined_bank.npz" ] || "$PY" "$SCRIPT_DIR/build_combined_bank.py" --stage3 "$STAGE3" --seed "$s" --output "$SD/combined_bank.npz"
  for D in Math Medical Science; do d=$(echo "$D" | tr '[:upper:]' '[:lower:]'); [ -s "$SD/selected/$d.json" ] || "$PY" "$PROJECT_DIR/scripts/select_domain_probes.py" --bank "$SD/combined_bank.npz" --probes "$PROBES" --domain "$D" --budget 32 --output "$SD/selected/$d.json"; done
  "$PY" - "$SD" <<'PY'
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);out=[]
for d in ['math','medical','science']: out.extend(int(x) for x in json.loads((p/'selected'/f'{d}.json').read_text())['selected'])
if len(out)!=96 or len(set(out))!=96: raise SystemExit(f'bad union selection n={len(out)} unique={len(set(out))}')
(p/'selected'/'union.json').write_text(json.dumps({'selected':out},indent=2))
PY
  [ -s "$SD/selected/union.jsonl" ] || "$PY" "$PROJECT_DIR/scripts/subset_probe_pool.py" --probes "$PROBES" --selected "$SD/selected/union.json" --output "$SD/selected/union.jsonl"
  STAGE="seed_${s}_full_target_probe_responses"
  if [ ! -s "$SD/responses/base.npz" ] || [ ! -s "$SD/responses/target.npz" ]; then "$PY" "$SCRIPT_DIR/run_pair_probe_bank.py" --model "$BASE_MODEL" --adapter "$SD/adapters/subset_Math_Medical_Science" --probes "$SD/selected/union.jsonl" --encoder "$ENCODER" --base-output "$SD/responses/base.npz" --target-output "$SD/responses/target.npz" --max-new-tokens 96 --batch-size 4; fi
  STAGE="seed_${s}_coordinates"
  "$PY" "$SCRIPT_DIR/compute_stage6_coordinates.py" --stage3 "$STAGE3" --seed "$s" --base-selected "$SD/responses/base.npz" --target-selected "$SD/responses/target.npz" --selection Math="$SD/selected/math.json" --selection Medical="$SD/selected/medical.json" --selection Science="$SD/selected/science.json" --output "$SD/coordinates.json"
done
STAGE=aggregate
"$PY" "$SCRIPT_DIR/aggregate_stage6.py" --run-root "$RUN_ROOT" --output-json "$RUN_ROOT/aggregate_stage6.json" --output-rows "$RUN_ROOT/functional_validity_rows.csv"
SUCCESS=1; STAGE=complete
