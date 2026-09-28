FAS Stage8 Deep Ancestry Matched-Scale v1

Purpose: final experiment block. Tests depth through a same-family 4B chain:
Math/Code/Medical sibling experts -> DARE-TIES merge -> base-neutral auxiliary SFT -> INT4 inference teacher -> 4B response-KD student.
This DOES NOT reopen the failed 4B->1.7B cross-scale C3 claim.

Run:
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage8_DeepAncestry_MatchedScale_v1.tar.gz
  cd FAS_Server_Stage8_DeepAncestry_MatchedScale_v1
  chmod +x run_fas_server.sh
  ./run_fas_server.sh

Return:
  ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE8_DEEP_ANCESTRY_MATCHEDSCALE_V1.tar.gz

The workflow is resumable. Existing Stage1-7 artifacts are read-only.
