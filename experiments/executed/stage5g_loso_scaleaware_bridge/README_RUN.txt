FAS Stage5G LOSO Scale-Aware Bridge v1

Purpose:
- Pure analysis/development diagnostic. No model training.
- Tests whether a bridge learned only from the OTHER two seeds can map actual 4B ancestor functional fields into a useful 1.7B coordinate system for the held-out Stage5 KD student.
- Held-out Active/Random B32 global probe IDs are explicitly removed from bridge training and lambda selection.
- Held-out 1.7B sibling experts are evaluation oracle only.

Run:
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage5G_LOSO_ScaleAwareBridge_v1.tar.gz
  cd FAS_Server_Stage5G_LOSO_ScaleAwareBridge_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Return:
  ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE5G_LOSO_SCALEAWARE_BRIDGE_V1.tar.gz

Scientific guardrail:
Passing Stage5G does NOT establish C3. Because this bridge was designed after observing Stage5, a genuinely new unseen seed is required for confirmatory evidence before C3 or L7.
