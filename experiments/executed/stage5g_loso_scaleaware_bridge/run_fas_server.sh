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
STAGE5F="$PROJECT_DIR/runs/fas_stage5f_scalematched17_multiseed_v1"
RUN_ROOT="$PROJECT_DIR/runs/fas_stage5g_loso_scaleaware_bridge_v1"
RETURN_DIR="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5G_LOSO_SCALEAWARE_BRIDGE_V1"
FINAL="${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5G_LOSO_SCALEAWARE_BRIDGE_V1.tar.gz"
LOG="$RUN_ROOT/stage5g.log"
SUCCESS=0; STAGE=bootstrap
mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_DIR"; mkdir -p "$RETURN_DIR"
exec > >(tee -a "$LOG") 2>&1
status(){ "$PY" - "$RETURN_DIR/status.json" "$1" "$STAGE" "$2" <<'PY'
import sys,json,datetime
json.dump({'workflow':'FAS_STAGE5G_LOSO_SCALE_AWARE_BRIDGE_V1','exit_code':int(sys.argv[2]),'stage':sys.argv[3],'message':sys.argv[4],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),indent=2)
PY
}
collect(){ set +e
  cp -f "$SCRIPT_DIR/BRIDGE_FREEZE_STAGE5G_V1.json" "$RETURN_DIR/"
  cp -f "$RUN_ROOT/aggregate_stage5g.json" "$RETURN_DIR/" 2>/dev/null || true
  cp -f "$LOG" "$RETURN_DIR/" 2>/dev/null || true
  df -hT / /data > "$RETURN_DIR/disk_usage.txt" 2>/dev/null || true
  tar -czf "$FINAL" -C "$RETURN_DIR" .
  echo "RETURN_THIS_SINGLE_FILE=$FINAL"
}
trap 'rc=$?; set +e; if [ "$SUCCESS" = 1 ]; then rc=0; status 0 "Stage5G LOSO scale-aware bridge diagnostic completed."; else [ "$rc" != 0 ] || rc=90; status "$rc" "Stage5G incomplete; no prior experiment artifacts were modified."; fi; collect; exit $rc' EXIT

STAGE=preflight
for p in "$PY" "$STAGE3/shared/base_full.npz" "$STAGE5/shared/base17.npz"; do [ -s "$p" ] || { echo "MISSING:$p"; exit 21; }; done
for s in 23 47 71; do
  for p in "$STAGE5/s$s/student_response.npz" "$STAGE5/s$s/evaluation.json" "$STAGE3/s$s/selections/active_B32.json" "$STAGE3/s$s/selections/random_B32.json"; do [ -s "$p" ] || { echo "MISSING:$p"; exit 22; }; done
  for d in math code medical science; do [ -s "$STAGE3/s$s/banks/$d.npz" ] || { echo "MISSING:$STAGE3/s$s/banks/$d.npz"; exit 23; }; done
done
for d in math code medical science; do [ -s "$STAGE5E/s23/experts/$d/response.npz" ] || { echo "MISSING:$STAGE5E/s23/experts/$d/response.npz"; exit 24; }; done
for s in 47 71; do for d in math code medical science; do [ -s "$STAGE5F/s$s/experts/$d/response.npz" ] || { echo "MISSING:$STAGE5F/s$s/experts/$d/response.npz"; exit 25; }; done; done
"$PY" -m py_compile "$SCRIPT_DIR/evaluate_loso_bridge_v1.py"
STAGE=evaluate_loso
"$PY" "$SCRIPT_DIR/evaluate_loso_bridge_v1.py" --stage3 "$STAGE3" --stage5 "$STAGE5" --stage5e "$STAGE5E" --stage5f "$STAGE5F" --output "$RUN_ROOT/aggregate_stage5g.json"
SUCCESS=1; STAGE=complete
