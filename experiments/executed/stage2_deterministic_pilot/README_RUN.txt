FAS TPAMI Stage2 Deterministic Pilot v1
=======================================

Purpose:
- No retraining.
- No rebuilding the full 1599-probe bank.
- Reuse existing Stage2-v4 Active/Random selections.
- Re-query only their union with greedy decoding (do_sample=False).
- Re-evaluate parent support, residual, deterministic geometry, and the
  construction-weight diagnostic.

This is a diagnostic measurement pilot, not a paper-ready protocol change.
Construction weights are NOT treated as exact functional ground truth.

Run:
  tar -xzf FAS_Server_Stage2_DeterministicPilot_v1.tar.gz
  cd FAS_Server_Stage2_DeterministicPilot_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Rerunning the same script resumes per-model chunk outputs.

Return:
  FAS_SERVER_RETURN_STAGE2_DETERMINISTIC_PILOT_V1.tar.gz
