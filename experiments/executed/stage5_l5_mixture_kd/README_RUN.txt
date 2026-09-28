FAS Stage5 — L5 Mixture-KD + strong baseline gate

Purpose:
- Reuse frozen seeds 23/47/71 experts; never retrain seed11.
- Build 2,000 source-disjoint KD prompts from held-out formal_raw rows (500/domain).
- Generate weighted teacher-exposure data with weights 0.4/0.3/0.2/0.1.
- Train Qwen3-1.7B QLoRA students, one per genealogy seed.
- Audit deterministic Active-B32 and Random-B32 using paired 1.7B base anchor and 4B ancestor dictionary.
- Compare FAS, Absolute+NNLS, Random-IFRF.
- Attempt released llm-dna==0.2.3 DNA-Decomp; baseline failure is nonfatal so expensive L5 outputs are preserved.
- G0 calibration is used only as G1 transfer diagnostic; no finite-sample claim for KD.

Run:
  cd ${FAS_WORK_ROOT}
  tar -xzf FAS_Server_Stage5_L5_MixtureKD_StrongBaseline_v1.tar.gz
  cd FAS_Server_Stage5_L5_MixtureKD_StrongBaseline_v1
  chmod +x run_fas_server.sh
  nohup ./run_fas_server.sh > stage5_outer.log 2>&1 < /dev/null &
  echo $! > stage5.pid

Return:
  FAS_SERVER_RETURN_STAGE5_L5_MIXTUREKD_STRONG_BASELINE_V1.tar.gz

PATCH v1.2: fixes NumPy int64 JSON serialization in L5 evaluation. Scientific freeze unchanged. Reuses the same run root and completed v1.1 artifacts. Return file: FAS_SERVER_RETURN_STAGE5_L5_MIXTUREKD_STRONG_BASELINE_V1_2.tar.gz
