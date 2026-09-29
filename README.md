# FAS — Functional Ancestry Simplex

Official research code release for “Functional Ancestry of Foundation Models.”

FAS studies **API-only multi-parent functional ancestry decomposition**. Instead of treating provenance as a binary source-verification problem, FAS uses controlled semantic interventions to build an **Interventional Functional Response Field (IFRF)**, constructs a Functional Ancestry Dictionary from candidate ancestors, actively selects identifiable probes, and solves a non-negative inverse problem to recover ancestor support, domain-conditioned functional ancestry coordinates, residual/explainedness, and selective audit states.

## What is in this release

This repository is a cleaned public release assembled from the final frozen manuscript codebase and the exact experiment scripts used for the reported Stage 5–8 analyses.

- `code/fas_core/` — core FAS algorithms: response fields, decomposition, active probing, conformal/selective inference, bootstrap uncertainty, metrics, and data governance.
- `code/adapters/` — text/diffusion adapter interfaces.
- `scripts/` — experiment preparation, probing, decomposition, calibration, functional-validity, open-set, baseline, data-governance, and paper-asset utilities.
- `configs/` — frozen experiment, merge, KD, calibration, and benchmark configurations.
- `jobs/` — reproducible job generators and shell entry points.
- `experiments/executed/` — scripts and freeze files from the actually executed late-stage experiments.
- `results/paper_evidence/` — compact machine-readable evidence used to produce paper tables/figures. No model weights are included.
- `tests/` — unit/regression tests for the core numerical and protocol logic.
- `paper_assets/figures_python/` — code for data-driven experiment figures.
- `docs/` — experiment runbook, data governance, evidence map, and reproducibility notes.

## Main paper evidence frozen in this release

| Evidence block | Frozen result |
|---|---:|
| Active probing geometry | mean `sigma_min` 0.5955 vs 0.5144 (random) |
| Multi-parent recovery | Parent-F1 L1/L2 = 0.7111 / 0.8571 |
| Functional validity | FAS–Exact-Shapley Spearman = 0.5556 vs construction share = 0.1111 |
| Evaluator robustness | support Jaccard = 1.000, coordinate Spearman = 1.000 across MPNet/MiniLM/BGE |
| Deep matched-scale ancestry | D4 Active Parent-F1 = 0.7937, true-parent mass = 0.7453, residual = 0.7106 |
| Cross-scale stress test | 4B→1.7B transport is not supported by the frozen protocol |
| Broad open-set stress test | AUROC = 0.7460; frozen broad-open-set gate not passed |

The repository deliberately preserves both positive and negative/boundary results. In particular, **cross-scale ancestry transport** and **broad unknown-ancestor detection** are not presented as solved problems.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

For full LLM experiments:

```bash
pip install -r requirements-experiments.txt
```

Optional merge baselines:

```bash
pip install -r requirements-merge.txt
```

### 2. Run the light-weight core smoke tests

```bash
bash run_smoke.sh
```

### 3. Minimal synthetic sanity check

```bash
python scripts/sanity_demo.py
```

### 4. Inspect the frozen paper evidence

```bash
python scripts/verify_public_release.py
```

## Reproducing the paper

Full reproduction requires public foundation models, benchmark datasets, GPU resources, and the exact frozen protocols. The high-level order is:

1. prepare source-disjoint formal datasets and intervention pools;
2. train/fetch candidate expert ancestors;
3. build ancestor response banks;
4. select Active-B32 probes and freeze geometry;
5. generate target descendants and run black-box response collection;
6. run FAS decomposition and matched calibration;
7. run functional-validity, robustness/open-set, and deep-ancestry audits.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md).

## Executed experiment snapshots

`experiments/executed/` contains the exact late-stage scientific scripts and freeze files used in the final evidence chain:

- Stage 2 deterministic pilot
- Stage 5 mixture-KD strong baseline
- Stage 5D cross-scale anchor diagnostic
- Stage 5E/F scale-matched 1.7B diagnostics
- Stage 5G leave-one-seed-out scale-aware bridge diagnostic
- Stage 6 exact-Shapley functional validity
- Stage 7 evaluator robustness / open-set audit
- Stage 8 matched-scale deep ancestry

Machine-specific server paths were replaced by `FAS_WORK_ROOT` / `FAS_SHARED_ROOT` in this public release. Scientific configurations and frozen thresholds were not changed.

## Data and model policy

This repository **does not redistribute** foundation-model weights or third-party benchmark corpora. Scripts download or consume them according to their original licenses. See [`docs/DATA_AND_MODEL_LICENSES.md`](docs/DATA_AND_MODEL_LICENSES.md).

## Reproducibility and claims

The most authoritative compact evidence is under `results/paper_evidence/`, especially:

- `FAS_TPAMI_EXPERIMENT_FREEZE_20260927_FINAL.md`
- `FAS_TPAMI_RESULT_LEDGER_20260927_FINAL.json`
- `stage3_confirmatory_aggregate_results.json`
- `stage4_g0_aggregate_calibration.json`
- Stage 6/7/8 aggregate JSON/CSV files

Matched finite-sample calibration guarantees established in Stage 4 must **not** be transferred automatically to later diagnostic or shifted regimes.

## Citation

If you use this code, please cite the corresponding TPAMI paper. A machine-readable citation template is provided in [`CITATION.cff`](CITATION.cff).

## License

Code authored for this project is released under the MIT License. Third-party models, datasets, baseline packages, and generated artifacts remain subject to their respective licenses.
