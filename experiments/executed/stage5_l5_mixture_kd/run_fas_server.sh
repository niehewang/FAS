#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${HOME}/FAS_TPAMI_SERVER_SHARED"; ENV_DIR="$SHARED_ROOT/envs/fas-py311"; PROJECT_DIR="$SHARED_ROOT/stage1_v3/project"; PY="$ENV_DIR/bin/python"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"; STAGE4="$PROJECT_DIR/runs/fas_g0_calibration_deterministic_v1"; RUN_ROOT="$PROJECT_DIR/runs/fas_stage5_l5_mixture_kd_v1"
RETURN_ROOT="$SHARED_ROOT/stage5_l5_mixture_kd_v1_return"; RETURN_DIR="$RETURN_ROOT/packet"; LOG="$RETURN_ROOT/stage5.log"; FINAL="$SCRIPT_DIR/FAS_SERVER_RETURN_STAGE5_L5_MIXTUREKD_STRONG_BASELINE_V1_2.tar.gz"; FREEZE="$SCRIPT_DIR/STAGE5_FREEZE_L5_DNA_V1.json"; SEEDS="${SEEDS:-23 47 71}"
export PATH="$ENV_DIR/bin:$PATH" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 HF_HOME="$SHARED_ROOT/hf_cache" HUGGINGFACE_HUB_CACHE="$SHARED_ROOT/hf_cache/hub" DATASETS_CACHE="$SHARED_ROOT/hf_cache/datasets" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SUCCESS=0; STAGE=bootstrap; mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_ROOT"; mkdir -p "$RETURN_DIR"; exec > >(tee -a "$LOG") 2>&1
status(){ "$PY" - "$RETURN_DIR/status.json" "$1" "$STAGE" "$2" <<'PY'
import sys,json,datetime;json.dump({'workflow':'FAS_STAGE5_L5_MIXTUREKD_STRONG_BASELINE_V1_2','exit_code':int(sys.argv[2]),'stage':sys.argv[3],'message':sys.argv[4],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
}
collect(){ set +e; cp -f "$LOG" "$RETURN_DIR/"; cp -f "$FREEZE" "$RETURN_DIR/"; cp -f "$RUN_ROOT/aggregate_stage5.json" "$RETURN_DIR/" 2>/dev/null||true; cp -f "$RUN_ROOT/kd_prompts.jsonl.meta.json" "$RETURN_DIR/" 2>/dev/null||true; for s in $SEEDS; do mkdir -p "$RETURN_DIR/s$s"; cp -f "$RUN_ROOT/s$s/evaluation.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/student/training_metadata.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/teacher_data.jsonl.meta.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/dna/decomposition.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; cp -f "$RUN_ROOT/s$s/dna/baseline_status.json" "$RETURN_DIR/s$s/" 2>/dev/null||true; done; df -hT / /data > "$RETURN_DIR/disk_usage.txt"; tar -czf "$FINAL" -C "$RETURN_DIR" .; }
trap 'rc=$?; set +e; if [ "$SUCCESS" = 1 ]; then rc=0; status 0 "Stage5 L5 Mixture-KD completed; NumPy-safe evaluation patch active; strong LLM-DNA baseline attempted under released package."; else [ "$rc" != 0 ] || rc=90; status "$rc" "Stage5 incomplete; rerun same package to resume preserved teacher outputs, student checkpoints and probe responses."; fi; collect; echo RETURN_THIS_SINGLE_FILE=$FINAL; exit $rc' EXIT
for x in "$PY" "$PROJECT_DIR/data/probes/interventions.jsonl" "$STAGE3/aggregate_results.json" "$STAGE4/aggregate_calibration.json"; do [ -e "$x" ] || { echo missing:$x; exit 11; }; done
for d in math code medical science; do [ -s "$PROJECT_DIR/data/training/formal_raw/$d.jsonl" ] || exit 12; [ -s "$PROJECT_DIR/data/training/formal/$d.jsonl" ] || exit 13; done
for s in $SEEDS; do for d in math code medical science; do [ -s "$PROJECT_DIR/runs/experts/s$s/$d/adapter/adapter_config.json" ] || { echo missing expert s$s $d; exit 14; }; done; [ -s "$STAGE3/s$s/selections/active_B32.json" ] || exit 15; [ -s "$STAGE3/s$s/selections/random_B32.json" ] || exit 16; [ -s "$STAGE4/s$s/final/support_threshold.json" ] || exit 17; done
mkdir -p "$RUN_ROOT"; if [ -s "$RUN_ROOT/STAGE5_FREEZE_L5_DNA_V1.json" ]; then cmp -s "$FREEZE" "$RUN_ROOT/STAGE5_FREEZE_L5_DNA_V1.json" || { echo freeze mismatch; exit 18; }; else cp "$FREEZE" "$RUN_ROOT/"; fi
STAGE=preflight; df -hT / /data; "$PY" - <<'PY'
import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0),torch.__version__)
PY
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
STAGE=prepare_kd_prompts
if [ -s "$RUN_ROOT/kd_prompts.jsonl" ] && ! "$PY" "$SCRIPT_DIR/validate_stage5_jsonl_v1.py" --path "$RUN_ROOT/kd_prompts.jsonl" --expected 2000; then
  mv "$RUN_ROOT/kd_prompts.jsonl" "$RUN_ROOT/kd_prompts.jsonl.corrupt.$(date +%s)"
  rm -f "$RUN_ROOT/kd_prompts.jsonl.meta.json"
fi
[ -s "$RUN_ROOT/kd_prompts.jsonl" ] || "$PY" "$SCRIPT_DIR/prepare_kd_holdout_v2.py" --project "$PROJECT_DIR" --output "$RUN_ROOT/kd_prompts.jsonl" --per-domain 500
STAGE=build_probe_union; mkdir -p "$RUN_ROOT/shared"; if [ ! -s "$RUN_ROOT/shared/audit_union.jsonl" ]; then cmd=("$PY" "$SCRIPT_DIR/build_probe_union_v2.py" --probes "$PROJECT_DIR/data/probes/interventions.jsonl" --output "$RUN_ROOT/shared/audit_union.jsonl"); for s in $SEEDS; do cmd+=(--selection "$STAGE3/s$s/selections/active_B32.json" --selection "$STAGE3/s$s/selections/random_B32.json"); done; "${cmd[@]}"; fi
STAGE=base17_response; if [ ! -s "$RUN_ROOT/shared/base17.npz" ]; then "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v2.py" --model Qwen/Qwen3-1.7B-Base --probes "$RUN_ROOT/shared/audit_union.jsonl" --encoder sentence-transformers/all-mpnet-base-v2 --max-new-tokens 96 --batch-size 8 --chunk-size 100 --work-dir "$RUN_ROOT/shared/base17_parts" --output "$RUN_ROOT/shared/base17.npz"; rm -rf "$RUN_ROOT/shared/base17_parts"; fi
for s in $SEEDS; do
 S="$RUN_ROOT/s$s"; mkdir -p "$S"
 STAGE=teacher_data_s$s
 if [ -s "$S/teacher_data.jsonl" ] && ! "$PY" "$SCRIPT_DIR/validate_stage5_jsonl_v1.py" --path "$S/teacher_data.jsonl" --expected 2000; then
   mv "$S/teacher_data.jsonl" "$S/teacher_data.jsonl.corrupt.$(date +%s)"; rm -f "$S/teacher_data.jsonl.meta.json"
 fi
 if [ ! -s "$S/teacher_data.jsonl" ]; then "$PY" "$SCRIPT_DIR/generate_teacher_data_resume_v2.py" --prompts "$RUN_ROOT/kd_prompts.jsonl" --seed "$s" --adapter-root "$PROJECT_DIR/runs/experts/s$s" --output "$S/teacher_data.jsonl" --work-dir "$S/teacher_parts" --max-new-tokens 512 --batch-size 4; fi
 STAGE=train_student_s$s; STUD="$S/student"; if [ ! -s "$STUD/adapter/adapter_config.json" ]; then resume=""; if compgen -G "$STUD/checkpoint-*" >/dev/null; then resume=$(ls -d "$STUD"/checkpoint-* | sort -V | tail -1); fi; "$PY" "$SCRIPT_DIR/make_student_config_v1.py" --seed "$s" --data "$S/teacher_data.jsonl" --output-dir "$STUD" --config "$S/student.yaml" ${resume:+--resume "$resume"}; "$PY" "$SCRIPT_DIR/train_student_v1.py" --config "$S/student.yaml"; fi
 STAGE=student_probe_s$s; if [ ! -s "$S/audit_union.jsonl" ]; then "$PY" "$SCRIPT_DIR/build_probe_union_v2.py" --probes "$PROJECT_DIR/data/probes/interventions.jsonl" --selection "$STAGE3/s$s/selections/active_B32.json" --selection "$STAGE3/s$s/selections/random_B32.json" --output "$S/audit_union.jsonl"; fi; if [ ! -s "$S/student_response.npz" ]; then "$PY" "$SCRIPT_DIR/deterministic_probe_runner_v2.py" --model Qwen/Qwen3-1.7B-Base --adapter "$STUD/adapter" --probes "$S/audit_union.jsonl" --encoder sentence-transformers/all-mpnet-base-v2 --max-new-tokens 96 --batch-size 8 --chunk-size 100 --work-dir "$S/student_parts" --output "$S/student_response.npz"; rm -rf "$S/student_parts"; fi
 STAGE=evaluate_s$s; "$PY" "$SCRIPT_DIR/evaluate_l5_v1.py" --seed "$s" --stage3 "$STAGE3" --stage4 "$STAGE4" --base17 "$RUN_ROOT/shared/base17.npz" --student "$S/student_response.npz" --output "$S/evaluation.json"
 # Strong baseline: nonfatal if upstream package/runtime fails; preserve L5 core outputs.
 STAGE=dna_baseline_s$s; mkdir -p "$S/dna"; BENV="$SHARED_ROOT/envs/fas-baselines-py311"; if [ ! -x "$BENV/bin/python" ]; then "$PY" -m venv --system-site-packages "$BENV"; "$BENV/bin/pip" install -q 'llm-dna==0.2.3' 'wonderwords>=2.2.0' 'tiktoken>=0.7.0' 'python-dotenv>=1.0.0' || true; fi
 set +e
 if "$BENV/bin/python" -c 'import llm_dna' >/dev/null 2>&1; then
   tmp="$S/dna/tmp_full"; mkdir -p "$tmp"; ok=1; arr=();
   for d in math code medical science; do v="$S/dna/${d}.npy"; if [ ! -s "$v" ]; then rm -rf "$tmp/$d"; "$PY" "$SCRIPT_DIR/materialize_lora_v1.py" --base Qwen/Qwen3-4B-Base --adapter "$PROJECT_DIR/runs/experts/s$s/$d/adapter" --output "$tmp/$d" && "$BENV/bin/python" "$SCRIPT_DIR/run_llm_dna_local_v1.py" --model "$tmp/$d" --output "$v" --seed 20260924 || ok=0; rm -rf "$tmp/$d"; fi; arr+=("$v"); done
   tv="$S/dna/target.npy"; if [ ! -s "$tv" ]; then rm -rf "$tmp/student"; "$PY" "$SCRIPT_DIR/materialize_lora_v1.py" --base Qwen/Qwen3-1.7B-Base --adapter "$STUD/adapter" --output "$tmp/student" && "$BENV/bin/python" "$SCRIPT_DIR/run_llm_dna_local_v1.py" --model "$tmp/student" --output "$tv" --seed 20260924 || ok=0; rm -rf "$tmp/student"; fi
   if [ "$ok" = 1 ] && [ -s "$tv" ]; then "$PY" "$SCRIPT_DIR/evaluate_dna_v1.py" --ancestors "${arr[@]}" --target "$tv" --threshold .01 --output "$S/dna/decomposition.json" || ok=0; fi
   "$PY" - "$S/dna/baseline_status.json" "$ok" <<'PY'
import sys,json;json.dump({'llm_dna_version':'0.2.3','status':'ok' if sys.argv[2]=='1' else 'failed','nonfatal_to_core':True},open(sys.argv[1],'w'),indent=2)
PY
 else "$PY" - "$S/dna/baseline_status.json" <<'PY'
import sys,json;json.dump({'llm_dna_version':'0.2.3','status':'import_failed','nonfatal_to_core':True},open(sys.argv[1],'w'),indent=2)
PY
 fi
 set -e; rm -rf "$S/dna/tmp_full"
done
STAGE=aggregate; "$PY" "$SCRIPT_DIR/summarize_stage5_v1.py" --run-root "$RUN_ROOT" --seeds "$(echo $SEEDS|tr ' ' ',')" --output "$RUN_ROOT/aggregate_stage5.json"; SUCCESS=1; STAGE=complete
