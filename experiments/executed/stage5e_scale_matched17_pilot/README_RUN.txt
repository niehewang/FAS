FAS Stage5E — Scale-Matched 1.7B Ancestor-Field Pilot v1

Purpose:
  Diagnose Stage5 C3 failure without entering L7. This pilot trains only four seed-23
  Qwen3-1.7B sibling experts using the exact frozen domain data / v0.11 expert recipe,
  then evaluates the already-trained seed-23 KD student with a same-scale surrogate FAS dictionary.

IMPORTANT:
  - Does NOT rerun Stage1-4.
  - Does NOT retrain seed-11.
  - Does NOT retrain the Stage5 KD student.
  - Does NOT overwrite original Stage5 results.
  - The 1.7B experts are diagnostic surrogates, NOT actual genealogical ancestors.

Run:
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage5E_ScaleMatched17_Pilot_v1.tar.gz
  cd FAS_Server_Stage5E_ScaleMatched17_Pilot_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Return only:
  ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5E_SCALEMATCHED17_PILOT_V1.tar.gz

The run is resumable: completed expert adapters / deterministic response NPZs are reused.
