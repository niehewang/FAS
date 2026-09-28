#!/usr/bin/env bash
FAS_WORK_ROOT="${FAS_WORK_ROOT:-$PWD}"
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${HOME}/FAS_TPAMI_SERVER_SHARED"; ENV_DIR="$SHARED_ROOT/envs/fas-py311"; PROJECT_DIR="$SHARED_ROOT/stage1_v3/project"; PY="$ENV_DIR/bin/python"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"; STAGE4="$PROJECT_DIR/runs/fas_g0_calibration_deterministic_v1"; STAGE5="$PROJECT_DIR/runs/fas_stage5_l5_mixture_kd_v1"; RUN_ROOT="$PROJECT_DIR/runs/fas_stage8_deep_ancestry_matchedscale_v1"
RETURN_ROOT="$SHARED_ROOT/stage8_deep_ancestry_matchedscale_v1_return"; RETURN_DIR="$RETURN_ROOT/packet"; LOG="$RETURN_ROOT/stage8.log"; FINAL="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE8_DEEP_ANCESTRY_MATCHEDSCALE_V1.tar.gz"; FREEZE="$SCRIPT_DIR/STAGE8_FREEZE.json"; SEEDS="${SEEDS:-23 47 71}"
export PATH="$ENV_DIR/bin:$PATH" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 HF_HOME="$SHARED_ROOT/hf_cache" HUGGINGFACE_HUB_CACHE="$SHARED_ROOT/hf_cache/hub" DATASETS_CACHE="$SHARED_ROOT/hf_cache/datasets" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SUCCESS=0; STAGE=bootstrap; mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_ROOT"; mkdir -p "$RETURN_DIR"; exec > >(tee -a "$LOG") 2>&1
status(){ "$PY" - "$RETURN_DIR/status.json" "$1" "$STAGE" "$2" <<'PY'
import sys,json,datetime;json.dump({'workflow':'FAS_STAGE8_DEEP_ANCESTRY_MATCHEDSCALE_V1','exit_code':int(sys.argv[2]),'stage':sys.argv[3],'message':sys.argv[4],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
}
collect(){ set +e; cp -f "$LOG" "$RETURN_DIR/"; cp -f "$FREEZE" "$RETURN_DIR/"; cp -f "$RUN_ROOT/aggregate_stage8.json" "$RETURN_DIR/" 2>/dev/null||true; cp -f "$RUN_ROOT/data_manifest.json" "$RETURN_DIR/" 2>/dev/null||true; for s in $SEEDS; do mkdir -p "$RETURN_DIR/s$s"; cp -f "$RUN_ROOT/s$s/evaluation.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/sft/training_metadata.json" "$RETURN_DIR/s$s/sft_training_metadata.json" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/student/training_metadata.json" "$RETURN_DIR/s$s/student_training_metadata.json" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/kd_teacher.jsonl.meta.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; done; df -hT / /data > "$RETURN_DIR/disk_usage.txt"; tar -czf "$FINAL" -C "$RETURN_DIR" .; }
trap 'rc=$?; set +e; if [ "$SUCCESS" = 1 ]; then rc=0; status 0 "Stage8 matched-scale deep-ancestry trajectory completed."; else [ "$rc" != 0 ] || rc=90; status "$rc" "Stage8 incomplete; rerun the same package to resume preserved artifacts."; fi; collect; echo RETURN_THIS_SINGLE_FILE=$FINAL; exit $rc' EXIT
for x in "$PY" "$PROJECT_DIR/data/probes/interventions.jsonl" "$STAGE3/shared/base_full.npz" "$STAGE4/aggregate_calibration.json"; do [ -e "$x" ] || { echo missing:$x; exit 11; }; done
command -v mergekit-yaml >/dev/null || { echo 'mergekit-yaml missing from environment'; exit 12; }
for s in $SEEDS; do for d in math code medical science; do [ -s "$PROJECT_DIR/runs/experts/s$s/$d/adapter/adapter_config.json" ] || { echo missing expert s$s $d; exit 13; }; [ -s "$STAGE3/s$s/banks/$d.npz" ] || exit 14; done; [ -s "$STAGE3/s$s/selections/active_B32.json" ] || exit 15; [ -s "$STAGE3/s$s/selections/random_B32.json" ] || exit 16; [ -s "$STAGE4/s$s/final/support_threshold.json" ] || exit 17; done
mkdir -p "$RUN_ROOT"; if [ -s "$RUN_ROOT/STAGE8_FREEZE.json" ]; then cmp -s "$FREEZE" "$RUN_ROOT/STAGE8_FREEZE.json" || { echo freeze mismatch; exit 18; }; else cp "$FREEZE" "$RUN_ROOT/"; fi
STAGE=preflight; "$PY" -m py_compile "$SCRIPT_DIR"/*.py; "$PY" - <<'PY'
import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0),torch.__version__)
PY
STAGE=prepare_data
if [ ! -s "$RUN_ROOT/holdout_2000.jsonl" ]; then if [ -s "$STAGE5/kd_prompts.jsonl" ]; then cp "$STAGE5/kd_prompts.jsonl" "$RUN_ROOT/holdout_2000.jsonl"; else "$PY" "$SCRIPT_DIR/prepare_holdout_v1.py" --project "$PROJECT_DIR" --output "$RUN_ROOT/holdout_2000.jsonl" --per-domain 500 --seed 20260924; fi; fi
[ -s "$RUN_ROOT/aux_prompts.jsonl" ] || "$PY" "$SCRIPT_DIR/prepare_deepchain_data_v1.py" --source "$RUN_ROOT/holdout_2000.jsonl" --aux "$RUN_ROOT/aux_prompts.jsonl" --kd "$RUN_ROOT/kd_prompts.jsonl" --manifest "$RUN_ROOT/data_manifest.json"
STAGE=aux_base_responses
[ -s "$RUN_ROOT/aux_sft.jsonl" ] || "$PY" "$SCRIPT_DIR/generate_single_teacher_resume_v1.py" --prompts "$RUN_ROOT/aux_prompts.jsonl" --model Qwen/Qwen3-4B-Base --output "$RUN_ROOT/aux_sft.jsonl" --max-new-tokens 256 --batch-size 4 --seed 20260927
for s in $SEEDS; do
 S="$RUN_ROOT/s$s"; mkdir -p "$S"; STAGE=seed_${s}_audit_union
 [ -s "$S/audit_union.jsonl" ] || "$PY" "$SCRIPT_DIR/build_probe_union_v2.py" --probes "$PROJECT_DIR/data/probes/interventions.jsonl" --selection "$STAGE3/s$s/selections/active_B32.json" --selection "$STAGE3/s$s/selections/random_B32.json" --output "$S/audit_union.jsonl"
 # If all downstream artifacts exist, skip heavy construction.
 if [ ! -s "$S/D4_kd4b.npz" ]; then
  PR="$S/temp_parents"; mkdir -p "$PR"
  for d in math code medical; do if [ ! -s "$PR/$d/config.json" ]; then STAGE=seed_${s}_materialize_$d; "$PY" "$SCRIPT_DIR/materialize_lora_v1.py" --base Qwen/Qwen3-4B-Base --adapter "$PROJECT_DIR/runs/experts/s$s/$d/adapter" --output "$PR/$d"; fi; done
  if [ ! -s "$S/merge/config.json" ]; then STAGE=seed_${s}_merge; "$PY" "$SCRIPT_DIR/make_merge_config_v1.py" --parent-root "$PR" --output "$S/merge.yaml"; mergekit-yaml "$S/merge.yaml" "$S/merge" --cuda --lazy-unpickle; fi
  STAGE=seed_${s}_audit_D1; [ -s "$S/D1_merge.npz" ] || "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v3.py" --model "$S/merge" --probes "$S/audit_union.jsonl" --output "$S/D1_merge.npz" --batch-size 8
  if [ ! -s "$S/sft/adapter/adapter_config.json" ]; then STAGE=seed_${s}_sft; resume=""; if compgen -G "$S/sft/checkpoint-*" >/dev/null; then resume=$(ls -d "$S"/sft/checkpoint-*|sort -V|tail -1); fi; "$PY" "$SCRIPT_DIR/make_train_config_v1.py" --model "$S/merge" --seed "$s" --data "$RUN_ROOT/aux_sft.jsonl" --output-dir "$S/sft" --config "$S/sft.yaml" --lr 0.0001 --domain DeepChainAuxSFT ${resume:+--resume "$resume"}; "$PY" "$SCRIPT_DIR/train_lora_v1.py" --config "$S/sft.yaml"; fi
  if [ ! -s "$S/sft_full/config.json" ]; then STAGE=seed_${s}_materialize_sft; "$PY" "$SCRIPT_DIR/materialize_lora_v1.py" --base "$S/merge" --adapter "$S/sft/adapter" --output "$S/sft_full"; fi
  STAGE=seed_${s}_audit_D2; [ -s "$S/D2_sft.npz" ] || "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v3.py" --model "$S/sft_full" --probes "$S/audit_union.jsonl" --output "$S/D2_sft.npz" --batch-size 8
  STAGE=seed_${s}_audit_D3; [ -s "$S/D3_int4.npz" ] || "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v3.py" --model "$S/sft_full" --probes "$S/audit_union.jsonl" --output "$S/D3_int4.npz" --batch-size 8 --load-in-4bit
  STAGE=seed_${s}_kd_teacher; [ -s "$S/kd_teacher.jsonl" ] || "$PY" "$SCRIPT_DIR/generate_single_teacher_resume_v1.py" --prompts "$RUN_ROOT/kd_prompts.jsonl" --model "$S/sft_full" --output "$S/kd_teacher.jsonl" --max-new-tokens 256 --batch-size 4 --load-in-4bit --seed "$s"
  if [ ! -s "$S/student/adapter/adapter_config.json" ]; then STAGE=seed_${s}_student; resume=""; if compgen -G "$S/student/checkpoint-*" >/dev/null; then resume=$(ls -d "$S"/student/checkpoint-*|sort -V|tail -1); fi; "$PY" "$SCRIPT_DIR/make_train_config_v1.py" --model Qwen/Qwen3-4B-Base --seed "$s" --data "$S/kd_teacher.jsonl" --output-dir "$S/student" --config "$S/student.yaml" --lr 0.0002 --domain DeepChainKD4B ${resume:+--resume "$resume"}; "$PY" "$SCRIPT_DIR/train_lora_v1.py" --config "$S/student.yaml"; fi
  STAGE=seed_${s}_audit_D4; "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v3.py" --model Qwen/Qwen3-4B-Base --adapter "$S/student/adapter" --probes "$S/audit_union.jsonl" --output "$S/D4_kd4b.npz" --batch-size 8
  rm -rf "$PR" "$S/merge" "$S/sft_full"
 fi
 STAGE=seed_${s}_evaluate; "$PY" "$SCRIPT_DIR/evaluate_stage8_v1.py" --seed "$s" --stage3 "$STAGE3" --stage4 "$STAGE4" --target D1_merge="$S/D1_merge.npz" --target D2_sft="$S/D2_sft.npz" --target D3_int4="$S/D3_int4.npz" --target D4_kd4b="$S/D4_kd4b.npz" --output "$S/evaluation.json"
done
STAGE=aggregate; "$PY" "$SCRIPT_DIR/summarize_stage8_v1.py" --run-root "$RUN_ROOT" --output "$RUN_ROOT/aggregate_stage8.json"; SUCCESS=1; STAGE=complete
