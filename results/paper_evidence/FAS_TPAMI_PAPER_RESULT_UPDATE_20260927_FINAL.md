# FAS TPAMI — FINAL PAPER RESULT UPDATE (after Stage8, 2026-09-27)

This file is intended for the dedicated manuscript-writing conversation. It supersedes earlier `PAPER_RESULT_UPDATE` files.

## Recommended final paper story

The strongest defensible story is no longer “ancestry survives every transformation.” It is:

1. **Functional ancestry is observable in black-box response geometry** when lineage candidates share a compatible functional representation family.
2. **Active intervention probes improve the ancestry geometry** and support decomposition under multi-parent mixtures.
3. **Matched null calibration provides selective decisions** with a finite-sample guarantee in the matched L1 regime.
4. **The inferred coordinates have functional meaning**: they track exact-Shapley realized contribution far better than nominal construction shares.
5. **The evidence is evaluator-stable** across MPNet, MiniLM, and BGE in the tested audit.
6. **Ancestry can remain recoverable through depth**: Merge -> SFT -> INT4 -> same-scale KD preserves meaningful parent recovery across three confirmatory seeds.
7. **Representation shift is a real boundary**: 4B->1.7B Mixture-KD breaks the present functional coordinate system, and a simple learned bridge does not solve it.
8. **Broad open-set unknown-ancestor detection is incomplete**: selective safeguards work, but the frozen open-set gate is not met.

This combination is appropriate for a mature TPAMI paper: strong central evidence + statistical controls + functional validation + deep-chain stress testing + explicit failure boundary.

## Numbers to integrate

### Core ancestry / geometry
- L1 FAS Parent-F1: Active 0.7111; Random 0.7022.
- L2 FAS Parent-F1: Active 0.8571; Random 0.8190.
- sigma_min: Active 0.5955 vs Random 0.5144.
- condition number: Active 2.7127 vs Random 3.2656.

### Functional validity
- FAS vs Exact-Shapley Spearman: 0.5556.
- construction share vs Exact-Shapley: 0.1111.
- FAS win/tie vs construction: 9/9 cells.
- Absolute-output vs Exact-Shapley: 0.5962.

### Evaluator robustness
- support Jaccard across tested encoders: 1.000.
- coordinate Spearman across tested encoders: 1.000.

### Open-set boundary
- pooled AUROC 0.7460.
- unknown Bank-Insufficient trigger 52.38%.
- known false Bank-Insufficient 0%.
- pair-shuffle collapse 3/3 seeds.

### Deep ancestry Stage8
D4 Active FAS: Parent-F1 0.79365; true-parent mass 0.74533; residual 0.71056; 3/3 seeds >=0.60 F1; ~100% mean F1 retention vs D1.

Active FAS trajectory:
D1: F1 0.79365 / mass 0.75010 / residual 0.58462.
D2: F1 0.85714 / mass 0.78435 / residual 0.65762.
D3: F1 0.85714 / mass 0.78609 / residual 0.70459.
D4: F1 0.79365 / mass 0.74533 / residual 0.71056.

At D4, Absolute Active has higher Parent-F1 (0.85714) but worse true-parent mass (0.70384) and worse residual (0.76141) than FAS Active. Do not cherry-pick one metric; show the full comparison.

## Claim-language guardrails

Use:
- “supports,” “in the tested same-family regime,” “matched-scale,” “transfer diagnostic,” “boundary case.”

Avoid:
- “universal,” “architecture invariant,” “cross-scale robust,” “solves open-set ancestry,” “formally calibrated at every depth.”

Finite-sample statement:
- Only the matched Stage4 L1 setting carries the matched-G0 finite-sample calibration claim.
- L2, Stage5 cross-scale, Stage7 alternate encoders, and Stage8 deep-chain states are transfer diagnostics unless separately recalibrated.

## Suggested experiment organization in the final paper

Main paper:
1. Core multi-parent attribution + Active vs Random geometry.
2. Matched calibration / selective decisions.
3. Functional validity (Exact Shapley).
4. Evaluator robustness.
5. Deep ancestry trajectory.
6. Stress-test boundary: cross-scale KD + open-set limits.

Supplementary:
- seed-level rows and thresholds;
- deterministic calibration tie analysis;
- Stage5 surrogate/bridge diagnosis;
- full Stage6 cell table;
- Stage7 open-set rows + negative controls;
- full Stage8 seed x depth x strategy table and data manifest.

## Next action for the paper-writing conversation

Use the current Drive manuscript branch (latest visible paper is `FAS_TPAMI_Chinese_Draft_v0.18-paper.zip`) and integrate the final frozen evidence. Do not request additional experiments from the experiment chat unless a concrete manuscript inconsistency or reviewer-critical missing control is discovered.
