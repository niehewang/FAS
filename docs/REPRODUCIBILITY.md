# Reproducibility guide

## Scope

This release separates three layers:

1. **Core method code** (`code/fas_core`, `scripts/`) — reusable numerical/protocol implementation.
2. **Frozen experiment specifications** (`configs/`, `experiments/executed/`) — paper-specific protocols and late-stage exact scripts.
3. **Compact evidence** (`results/paper_evidence/`) — small JSON/CSV records used to verify reported tables/figures.

Model weights and third-party raw datasets are intentionally excluded.

## Recommended environment

- Linux
- Python 3.10/3.11
- CUDA-capable PyTorch for full experiments
- `requirements-experiments.txt` for LLM experiments
- `requirements-merge.txt` for TIES/DARE-TIES merge experiments

Set:

```bash
export FAS_WORK_ROOT=/path/to/fast/workspace
export FAS_SHARED_ROOT=/path/to/shared/workspace
export HF_HOME=/path/to/hf_cache
```

## Core pipeline

### A. Data governance and intervention construction

Primary scripts:

- `scripts/prepare_formal_datasets.py`
- `scripts/audit_data_contamination.py`
- `scripts/build_formal_interventions.py`
- `scripts/validate_formal_interventions.py`
- `scripts/prepare_formal_probe_seeds.py`

The final protocol uses source-disjoint evaluation where specified and archives contamination checks.

### B. Ancestor bank construction

- train/fetch expert ancestors using `scripts/train_lora_expert.py` and the frozen expert configs;
- collect controlled response banks with `scripts/run_text_probe_bank.py`;
- form ancestor fields using `scripts/build_response_bank.py`.

### C. Active probing and decomposition

- `scripts/select_probes.py` / `scripts/select_domain_probes.py`
- `scripts/prepare_target_decomposition.py`
- `scripts/decompose_target.py`
- core NNLS logic in `code/fas_core/decompose.py`

### D. Selective calibration

- `scripts/generate_calibration_plan.py`
- `scripts/collect_calibration_scores.py`
- `scripts/calibrate_support_from_prepared.py`
- `code/fas_core/conformal.py`
- `code/fas_core/selective.py`

Stage 4 is the matched-calibration source of the finite-sample guarantee. Later shifted diagnostics must not inherit this guarantee automatically.

## Paper-stage map

| Paper evidence | Code location |
|---|---|
| Deterministic pilot | `experiments/executed/stage2_deterministic_pilot/` |
| Multi-parent confirmatory + active geometry | core scripts + `results/paper_evidence/stage3_confirmatory_aggregate_results.json` |
| Matched G0 calibration | core calibration scripts + `results/paper_evidence/stage4_g0_aggregate_calibration.json` |
| Cross-scale stress/diagnostics | `experiments/executed/stage5_*` |
| Exact-Shapley functional validity | `experiments/executed/stage6_functional_validity/` |
| Evaluator/open-set robustness | `experiments/executed/stage7_robust_audit/` |
| Deep matched-scale ancestry | `experiments/executed/stage8_deep_ancestry/` |

## Public-release path sanitization

The original server runs used machine-specific absolute paths. In this release those path strings were replaced by `FAS_WORK_ROOT` and `FAS_SHARED_ROOT`. Scientific hyperparameters, seeds, target definitions, thresholds, and freeze files were not changed.

## Evidence verification

Run:

```bash
python scripts/verify_public_release.py
```

The script checks that the final evidence ledger and the Stage 3/4/6/7/8 files expected by the paper are present and parseable.
