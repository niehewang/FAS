#!/usr/bin/env bash
FAS_WORK_ROOT="${FAS_WORK_ROOT:-$PWD}"
set -euo pipefail
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$HOME/FAS_TPAMI_SERVER_SHARED/stage1_v3/project}"
PY="${PYTHON:-$HOME/FAS_TPAMI_SERVER_SHARED/envs/fas-py311/bin/python}"
STAGE3="$PROJECT_DIR/runs/fas_confirmatory_deterministic_v1"
STAGE4="$PROJECT_DIR/runs/fas_g0_calibration_deterministic_v1"
RUN_ROOT="$PROJECT_DIR/runs/fas_stage7_robust_audit_v1"
RETURN_ROOT="${RETURN_ROOT:-${FAS_WORK_ROOT}}"
RETURN="$RETURN_ROOT/FAS_SERVER_RETURN_STAGE7_ROBUST_AUDIT_V1.tar.gz"
export HF_HOME="${HF_HOME:-$HOME/FAS_TPAMI_SERVER_SHARED/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/stage7.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage7 robust audit start ==="
echo "project=$PROJECT_DIR"
echo "python=$PY"
for p in "$PROJECT_DIR" "$STAGE3" "$STAGE4" "$PROJECT_DIR/runs/experts"; do
  if [[ ! -e "$p" ]]; then
    echo "MISSING $p"
    cat > "$RUN_ROOT/status.json" <<EOF
{"workflow":"FAS_STAGE7_ROBUST_AUDIT_V1","exit_code":21,"stage":"preflight_failed","message":"missing required path: $p"}
EOF
    tar -czf "$RETURN" -C "$RUN_ROOT" status.json stage7.log
    exit 21
  fi
done
for s in 23 47 71; do
  test -s "$STAGE3/s$s/selections/active_B32.json"
  test -s "$STAGE3/s$s/selections/union_probes.jsonl"
  for a in math code medical science; do test -d "$PROJECT_DIR/runs/experts/s$s/$a/adapter"; done
done

set +e
"$PY" "$PKG_DIR/stage7_robust_audit.py" \
  --project "$PROJECT_DIR" --stage3 "$STAGE3" --stage4 "$STAGE4" --run-root "$RUN_ROOT" \
  --plan "$PKG_DIR/STAGE7_TARGET_PLAN.json" --freeze "$PKG_DIR/STAGE7_FREEZE.json"
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  cat > "$RUN_ROOT/status.json" <<EOF
{"workflow":"FAS_STAGE7_ROBUST_AUDIT_V1","exit_code":$RC,"stage":"failed","message":"stage7_robust_audit.py failed; inspect stage7.log; completed artifacts retained for resume"}
EOF
else
  cat > "$RUN_ROOT/status.json" <<EOF
{"workflow":"FAS_STAGE7_ROBUST_AUDIT_V1","exit_code":0,"stage":"complete","message":"Stage7 C6 encoder sensitivity + C8 selective/open-set/negative-controls completed."}
EOF
fi
cp "$PKG_DIR/STAGE7_FREEZE.json" "$RUN_ROOT/STAGE7_FREEZE.json"
cp "$PKG_DIR/STAGE7_TARGET_PLAN.json" "$RUN_ROOT/STAGE7_TARGET_PLAN.json"
(df -h / /data 2>/dev/null || true) > "$RUN_ROOT/disk_usage.txt"
# Return only compact evidence, not raw generations/embeddings.
FILES=(status.json stage7.log STAGE7_FREEZE.json STAGE7_TARGET_PLAN.json disk_usage.txt)
for f in aggregate_stage7.json per_seed_stage7.json negative_controls.json encoder_sensitivity_rows.csv open_set_rows.csv; do [[ -f "$RUN_ROOT/$f" ]] && FILES+=("$f"); done
tar -czf "$RETURN" -C "$RUN_ROOT" "${FILES[@]}"
echo "RETURN=$RETURN"
exit $RC
