#!/usr/bin/env bash
FAS_WORK_ROOT="${FAS_WORK_ROOT:-$PWD}"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$HOME/FAS_TPAMI_SERVER_SHARED/stage1_v3/project}"
PY="${PY:-$HOME/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python}"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"
STAGE4="$PROJECT_DIR/runs/fas_g0_calibration_deterministic_v1"
STAGE5="$PROJECT_DIR/runs/fas_stage5_l5_mixture_kd_v1"
RUN_ROOT="$PROJECT_DIR/runs/fas_stage5d_crossscale_anchor_diagnostic_v1"
RETURN_ROOT="${RETURN_ROOT:-${FAS_WORK_ROOT}}"
RETURN_DIR="$RETURN_ROOT/FAS_SERVER_RETURN_STAGE5D_CROSSSCALE_ANCHOR_DIAGNOSTIC_V1"
FINAL="$RETURN_ROOT/FAS_SERVER_RETURN_STAGE5D_CROSSSCALE_ANCHOR_DIAGNOSTIC_V1.tar.gz"
LOG="$RUN_ROOT/stage5d.log"
mkdir -p "$RUN_ROOT"; rm -rf "$RETURN_DIR"; mkdir -p "$RETURN_DIR"
exec > >(tee -a "$LOG") 2>&1
status(){ cat > "$RETURN_DIR/status.json" <<EOF
{"workflow":"FAS_STAGE5D_CROSSSCALE_ANCHOR_DIAGNOSTIC_V1","exit_code":$1,"stage":"$2","message":"$3"}
EOF
}
collect(){ set +e; cp -f "$SCRIPT_DIR/DIAGNOSTIC_FREEZE_V1.json" "$RETURN_DIR/"; cp -f "$RUN_ROOT/aggregate_diagnostic.json" "$RETURN_DIR/" 2>/dev/null || true; cp -f "$LOG" "$RETURN_DIR/" 2>/dev/null || true; for s in 23 47 71; do mkdir -p "$RETURN_DIR/s$s"; cp -f "$RUN_ROOT/s$s/diagnostic.json" "$RETURN_DIR/s$s/" 2>/dev/null || true; done; df -hT / /data > "$RETURN_DIR/disk_usage.txt" 2>/dev/null || true; tar -czf "$FINAL" -C "$RETURN_DIR" .; echo "RETURN_THIS_SINGLE_FILE=$FINAL"; }
trap 'rc=$?; set +e; if [ "$rc" = 0 ]; then status 0 complete "Stage5D cross-scale anchor diagnostic completed without retraining."; else status "$rc" failed "Stage5D diagnostic failed; existing Stage1-5 artifacts were not modified."; fi; collect; exit "$rc"' EXIT

for p in "$PY" "$STAGE3/shared/base_full.npz" "$STAGE5/shared/base17.npz"; do [ -e "$p" ] || { echo "MISSING: $p"; exit 21; }; done
for s in 23 47 71; do
  for p in "$STAGE5/s$s/student_response.npz" "$STAGE5/s$s/evaluation.json" "$STAGE3/s$s/banks/math.npz" "$STAGE3/s$s/banks/code.npz" "$STAGE3/s$s/banks/medical.npz" "$STAGE3/s$s/banks/science.npz" "$STAGE3/s$s/selections/active_B32.json" "$STAGE3/s$s/selections/random_B32.json"; do [ -s "$p" ] || { echo "MISSING: $p"; exit 22; }; done
  if [ ! -s "$STAGE4/s$s/final/support_threshold.json" ] && [ ! -s "$STAGE4/s$s/support_threshold.json" ]; then echo "MISSING support threshold for s$s"; exit 23; fi
done
cp -f "$SCRIPT_DIR/DIAGNOSTIC_FREEZE_V1.json" "$RUN_ROOT/DIAGNOSTIC_FREEZE_V1.json"
for s in 23 47 71; do
  mkdir -p "$RUN_ROOT/s$s"
  "$PY" "$SCRIPT_DIR/diagnose_crossscale_v1.py" --seed "$s" --stage3 "$STAGE3" --stage4 "$STAGE4" --stage5 "$STAGE5" --output "$RUN_ROOT/s$s/diagnostic.json"
done
"$PY" "$SCRIPT_DIR/summarize_diagnostic_v1.py" --run-root "$RUN_ROOT" --output "$RUN_ROOT/aggregate_diagnostic.json"
