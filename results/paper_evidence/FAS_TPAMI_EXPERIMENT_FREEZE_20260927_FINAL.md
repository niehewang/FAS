# FAS TPAMI — FINAL EXPERIMENT FREEZE (2026-09-27)

## Status

The server-experiment phase is **closed** after Stage8. Do not create Stage8A/8B/8C or reopen Stage5/C3 unless a reviewer later asks for a targeted check. The manuscript should now be completed from the current paper branch (Drive currently contains `FAS_TPAMI_Chinese_Draft_v0.18-paper.zip`) using the evidence below.

## Authoritative experimental evidence

### Stage3 — multi-seed deterministic confirmatory ancestry decomposition
Seeds: 23/47/71. Same-family Qwen3-4B setting.
- L1 FAS Parent-F1: Active 0.7111, Random 0.7022.
- L2 FAS Parent-F1: Active 0.8571, Random 0.8190.
- L1 false leakage: Active 0.2306, Random 0.2611.
- L2 false leakage: Active 0.1674, Random 0.2171.
- Geometry sigma_min: Active 0.5955, Random 0.5144.
- Geometry condition number: Active 2.7127, Random 3.2656.
- Absolute remains competitive and sometimes stronger on Parent-F1. Do not claim blanket superiority over Absolute.

### Stage4 — matched-G0 calibration
- Frozen support thresholds for seeds 23/47/71: [0.01, 0.01, 0.15].
- L1 has matched-G0 finite-sample calibration evidence.
- L2 is G1 transfer diagnostic only; do not claim finite-sample guarantee for L2.
- Deterministic calibration scores have ties / few unique values; disclose this limitation.

### Stage5 — cross-scale 4B -> 1.7B Mixture-KD stress test
- Cross-scale FAS residual is very high (~0.95–0.97); original C3 broad cross-scale claim is **not supported**.
- Same-scale 1.7B surrogate dictionaries reduce residual to pooled ~0.674, indicating a strong representation-scale mismatch component.
- A learned LOSO 4B->1.7B bridge fails to generalize (pooled residual remains ~0.957; oracle-gap recovery ~0.26%).
- Conclusion: cross-scale functional coordinates are not solved by the current method. Report as a boundary/stress-test result; do not reopen C3.

### Stage6 — functional validity (C4)
Exact Shapley realized contribution over 9 seed x domain cells:
- FAS mean Spearman: 0.5556.
- Construction-share mean Spearman: 0.1111.
- Mean FAS advantage vs construction: +0.4444.
- FAS wins or ties construction in 9/9 cells.
- Absolute-output mean Shapley Spearman: 0.5962, slightly above FAS.

Interpretation: FAS coordinates track realized functional contribution substantially better than nominal construction shares in the tested controlled family, but FAS is not uniformly best on every contribution-estimation metric.

### Stage7 — robust audit (C6 + C8)
C6 encoder sensitivity: **supported**.
- MPNet / MiniLM / BGE: support Jaccard = 1.000 and coordinate Spearman = 1.000 in the tested audit.
- The matched-G0 finite-sample guarantee remains tied to the MPNet primary evaluator; do not transfer that guarantee to alternate encoders.

C8 broad open-set unknown-ancestor detection: **not supported by the frozen gate**.
- Pooled AUROC = 0.7460 (< 0.75 gate).
- Unknown Bank-Insufficient trigger rate = 52.38% (< 67% gate).
- Known false Bank-Insufficient rate = 0%.
- Pair-shuffle collapses in 3/3 seeds; base-zero false-signal = 0%.

Interpretation: selective safeguards and negative controls behave sensibly, but broad unknown-ancestor detection should be framed as limited / incomplete rather than solved.

### Stage8 — matched-scale deep ancestry (FINAL BLOCK)
Chain: D1 DARE-TIES merge -> D2 base-neutral SFT -> D3 INT4/NF4 inference -> D4 same-scale Qwen3-4B response KD.
Seeds: 23/47/71.

Frozen Stage8 gate: **PASS**.
D4 Active FAS:
- mean Parent-F1 = 0.7936507937
- mean true-parent mass = 0.7453294354
- mean relative residual = 0.7105572346
- F1 retention vs D1 = ~1.00
- seeds with D4 Parent-F1 >= 0.60 = 3/3

Depth trajectory, Active FAS mean:
- D1 merge: F1 0.79365, true-parent mass 0.75010, residual 0.58462
- D2 SFT: F1 0.85714, true-parent mass 0.78435, residual 0.65762
- D3 INT4: F1 0.85714, true-parent mass 0.78609, residual 0.70459
- D4 4B KD: F1 0.79365, true-parent mass 0.74533, residual 0.71056

D4 comparison:
- FAS Active: F1 0.79365, true-parent mass 0.74533, residual 0.71056.
- Absolute Active: F1 0.85714, true-parent mass 0.70384, residual 0.76141.
- FAS Random: F1 0.83810, true-parent mass 0.82604, residual 0.73512.
- Absolute Random: F1 0.79365, true-parent mass 0.77598, residual 0.77236.

Important caveat: Stage4 matched-G0 p-values are only transfer diagnostics on D1-D4. Deep-ancestry support/recovery is supported, but a new finite-sample calibration guarantee for the deep chain was not established. D3/D4 selective states therefore must not be described as formally calibrated under Stage4.

## Final claim status
- C1 active probing / better-conditioned functional geometry: SUPPORTED.
- C2 multi-parent ancestry decomposition: SUPPORTED in the tested same-family setting; Absolute is a strong comparator.
- C3 broad cross-scale 4B->1.7B ancestry persistence: NOT SUPPORTED; keep as limitation/stress-test.
- C4 functional validity beyond construction shares: SUPPORTED, with Absolute nuance.
- C5 matched statistical calibration/selective decision: SUPPORTED for the matched L1 regime; L2/deep-chain transfer is diagnostic only.
- C6 encoder robustness: SUPPORTED for the tested MPNet/MiniLM/BGE evaluators.
- C8 broad unknown-ancestor/open-set detection: NOT SUPPORTED by the frozen gate; report partial safeguards only.
- Matched-scale deep ancestry through Merge->SFT->INT4->KD: SUPPORTED by Stage8 frozen gate.

## Stop rule
No new exploratory server experiments. Next work is manuscript integration, figures/tables, supplementary evidence map, claim-language audit, and reviewer-style consistency checking.
