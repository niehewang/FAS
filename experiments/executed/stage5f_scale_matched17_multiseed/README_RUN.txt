FAS Stage5F — Scale-Matched 1.7B Multi-Seed Diagnostic v1

Purpose
- Stage5E seed-23 pilot passed its frozen diagnostic gate.
- This package extends ONLY the same diagnostic to seeds 47 and 71.
- It does NOT retrain seed-23, Stage1-4, or any Stage5 KD student.
- It trains Qwen3-1.7B sibling surrogate experts for seeds 47/71 and evaluates the already-existing Stage5 KD students with a scale-matched surrogate dictionary.

Run
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage5F_ScaleMatched17_MultiSeed_v1.tar.gz
  cd FAS_Server_Stage5F_ScaleMatched17_MultiSeed_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Resume
- Safe to rerun the SAME package.
- Completed adapters, checkpoints, and response.npz files are reused.
- Seed-23 Stage5E artifacts are reference-only and never retrained here.

Return file
  ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5F_SCALEMATCHED17_MULTISEED_V1.tar.gz

Interpretation
- This is a diagnostic surrogate-dictionary test, not genealogical evidence.
- A pass does NOT make original cross-scale Stage5 C3 pass and does NOT authorize L7 by itself.
- If the frozen multi-seed diagnostic passes, the next scientific step is a frozen scale-aware representation/bridge test on the ACTUAL 4B->1.7B lineage.
