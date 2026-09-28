# Stage5 Engineering Patch v1.2

This is an engineering-only resume patch. The scientific freeze file `STAGE5_FREEZE_L5_DNA_V1.json` is unchanged byte-for-byte from v1/v1.1.

## Failure fixed
Stage5 v1.1 completed seed-23 teacher generation, Qwen3-1.7B student training, and deterministic student-response measurement, then failed during `evaluation.json` serialization because `fas_detected` contained NumPy `int64` values. Python's standard JSON encoder does not serialize NumPy scalar integers.

## Patch
- converts detected support indices to native Python `int` in `sm()`;
- adds a conservative JSON fallback for NumPy integer/float/bool/array objects;
- changes no metric formula, threshold, probe set, model, seed, training data, or calibration rule;
- preserves and reuses the existing Stage5 run root, so completed seed-23 teacher data, student adapter, and `student_response.npz` are skipped automatically;
- return archive name is bumped to `...V1_2.tar.gz` for provenance.

## Resume behavior
Rerunning this package should start by validating the frozen inputs, reuse existing `kd_prompts`, `audit_union`, 1.7B base response, and seed-23 completed artifacts, then recompute only seed-23 evaluation before continuing to its nonfatal LLM-DNA baseline and seeds 47/71.
