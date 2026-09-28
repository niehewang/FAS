# Strong-baseline integration notes

This file records **how** the closest baselines should be run so the final paper does not weaken them through an inconsistent solver or a hand reimplementation.

## LLM DNA / RepTrace

- Upstream project: `Xtra-Computing/LLM-DNA`.
- Locked experiment package: `llm-dna==0.2.3` (Apache-2.0).
- FAS wrapper: `scripts/run_llm_dna_baseline.py`.
- Extract ancestor/target DNA using the upstream `DNAExtractionConfig`/`calc_dna` API, then pass the resulting vectors to `scripts/decompose_vectors.py` so DNA-Decomp and FAS use the same nonnegative decomposition layer.
- Record `dataset`, `max_samples`, `dna_dim`, reduction method, package version, model revision and random seed in run metadata.

## modelDNA

- Upstream package is pinned as `modeldna==0.1.0`; the wrapper invokes its released CLI rather than reimplementing weight fingerprints.
- Use modelDNA **only** on settings where the method's weight-access assumptions hold. Do not turn `N/A` in cross-scale KD into a numerical failure.
- For the final locked experiment, pin the exact upstream commit/package version in the environment manifest and retain the raw JSON/fingerprint output.
- Do not reimplement its weight fingerprints in FAS code. The paper should compare against the authors' released implementation/tooling.

## Distillation-relation baselines

DistillTrace/related black-box distillation detectors answer an edge/relation question rather than a share-decomposition question. Use them, where code is available and compatible, for parent-edge detection/support comparison; do not manufacture a quantitative ancestry coordinate from a detector that was not designed to output one.

## Fairness rule

The paper must distinguish three cases explicitly:

1. baseline applicable and run successfully;
2. baseline applicable but fails empirically (report the real result);
3. baseline not applicable under its published access/model assumptions (report `N/A`, with reason).

Never convert case (3) into zero accuracy.

## Automated paper-facing path

`generate_baseline_target_manifest.py` selects completed targets from the canonical experiment manifest. `generate_baseline_commands.py` then produces one executable job that extracts DNA/modelDNA outputs, converts them to the common row schema, evaluates support recovery, merges the resulting summaries through `sync_paper_results.py`, and rebuilds paper assets. Unsupported modelDNA rows stay N/A and are excluded from aggregate scores rather than counted as failures.
