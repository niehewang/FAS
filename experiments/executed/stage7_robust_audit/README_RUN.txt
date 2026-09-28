FAS Stage7 Robust Audit v1
==========================
Purpose: close C6 (encoder sensitivity) and C8 (selective/open-set + negative controls) in one bounded run.
No model training. Reuses frozen Stage3 experts and Stage4 calibration.

Run:
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage7_RobustAudit_v1.tar.gz
  cd FAS_Server_Stage7_RobustAudit_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Expected return:
  ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE7_ROBUST_AUDIT_V1.tar.gz

The run is resumable: completed per-seed raw generation/embedding files are reused.
