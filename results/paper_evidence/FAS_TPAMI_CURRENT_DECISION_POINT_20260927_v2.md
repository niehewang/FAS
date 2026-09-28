# FAS TPAMI — Current Decision Point (2026-09-27 v2)

## Authoritative status
This file supersedes earlier CURRENT_DECISION_POINT files in the FAS Drive folder.

### Completed scientific stages
- Stage1 formal data freeze: complete.
- Stage2 development / deterministic pilot: complete.
- Stage3 confirmatory seeds 23/47/71: complete.
- Stage4 matched-G0 calibration: complete.
- Stage5 Mixture-KD cross-scale stress: complete; original 4B->1.7B C3 not supported.
- Stage5E/F same-scale diagnostic: complete; same-scale 1.7B surrogate dictionaries reduce pooled residual to ~0.674, localizing much of Stage5 failure to cross-scale representation mismatch.
- Stage5G LOSO linear scale bridge: failed; pooled residual essentially unchanged (~0.958 -> ~0.957). Cross-scale C3 remains closed. Stage5H was cancelled and must not be run.
- Stage6 Functional Validity: complete; C4 gate passed.
- Stage7 Robust Audit: complete; C6 passed, C8 failed the preregistered broad open-set gate.

## Stage6 C4 result
Formal Exact-Shapley held-out utility experiment, seeds 23/47/71, 9 seed×domain cells:
- FAS mean Spearman vs realized Shapley: 0.5556.
- Construction-share mean Spearman: 0.1111.
- Mean FAS advantage over construction share: +0.4444.
- FAS wins/ties construction in 9/9 cells.
- Absolute-output mean Shapley Spearman: 0.5962, slightly higher than FAS.
Interpretation: FAS coordinates track realized functional contribution substantially better than construction weights in this controlled family, but do not claim FAS universally beats Absolute-output on contribution correlation.

## Stage7 C6 result — SUPPORTED
Across seeds 23/47/71 and MPNet / MiniLM / BGE on the frozen L1 Active-B32 audit:
- mean alternative-encoder support Jaccard vs MPNet = 1.0000.
- mean alternative-encoder coordinate Spearman vs MPNet = 1.0000.
- MiniLM mean true-parent F1 = 1.0000.
- BGE mean true-parent F1 = 1.0000.
Claim allowed: ancestry support/ranking is qualitatively stable across the three tested sentence encoders.
Guardrail: Stage4 finite-sample calibration remains specific to the primary MPNet evaluator; do not transfer conformal guarantees to alternative encoders.

## Stage7 C8 result — DOWNGRADED / broad open-set claim NOT supported
Frozen restricted-bank setting: known bank = Math/Code/Medical, Science withheld.
- pooled signal-eligible known n = 24.
- pooled signal-eligible unknown n = 21.
- pooled open-set AUROC = 0.7460 (gate required >=0.75).
- known false Bank-Insufficient rate = 0.0000.
- unknown Bank-Insufficient rate = 0.5238 (gate required >=0.67).
- probe-pair shuffle collapsed all 3 seeds.
- base-zero false-signal rate = 0.0000.
Interpretation: selective safeguards are meaningful and known-target false abstention is well controlled in this experiment, but withheld-ancestor detection is not strong enough for a broad open-set claim. Keep Low-Signal / Bank-Insufficient as safeguards, report the open-set test as a boundary result, and do not tune the threshold after seeing Stage7.

## Only remaining server experiment block: Stage8
Package: `FAS_Server_Stage8_DeepAncestry_MatchedScale_v1.tar.gz`
Purpose: same-family deep ancestry trajectory, not cross-scale rescue.

Frozen chain per seed 23/47/71:
1. D1: Math/Code/Medical 4B sibling experts -> DARE-TIES merge (0.40/0.35/0.25, density=0.5).
2. D2: base-neutral auxiliary SFT of the merged 4B target using 100 source-disjoint held-out prompts/domain and frozen Qwen3-4B base responses.
3. D3: INT4/NF4 inference view of D2.
4. D4: response KD from D3 into a Qwen3-4B-Base QLoRA student using a disjoint 250 prompts/domain.

Audit: Active-B32 and Random-B32, four-ancestor seed-matched Stage3 bank, v0.11 global ancestor-norm normalization, Stage4 support threshold reused without retuning, Absolute-output baseline included. G0 p-values are transfer diagnostics only.

Stage8 gate is frozen before execution:
- D4 Active mean Parent-F1 >=0.60.
- D4 Active mean true-parent mass >=0.65.
- D4 Active mean relative residual <=0.80.
- D4 Active F1 retention vs D1 >=0.70.
- >=2/3 seeds have D4 Active Parent-F1 >=0.60.

Regardless of Stage8 outcome, 4B->1.7B cross-scale C3 remains closed.

## After Stage8
No further exploratory server stages. Freeze experimental evidence, update result ledger, and perform final TPAMI manuscript integration / reviewer audit. Negative or boundary Stage8 results are reportable; do not create Stage8A/B/C to rescue them.
