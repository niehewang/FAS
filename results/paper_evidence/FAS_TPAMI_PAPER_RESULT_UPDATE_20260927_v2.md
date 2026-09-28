# FAS TPAMI — Paper Result Update (2026-09-27 v2)

This is the manuscript-facing update and supersedes earlier paper-result update files.

## Main evidence story now recommended
The paper should be centered on: active functional probing -> ancestry decomposition -> matched calibration / selective decision -> functional validity -> evaluator robustness -> depth stress, with explicit limits under cross-scale KD and broad open-set detection.

## Claims and writing strength
- C1 Active identifiability: supported by Stage2/3 geometry/conditioning evidence.
- C2 Multi-parent support recovery: supported in matched 4B controlled descendants; Absolute is a strong baseline and must remain visible.
- C3 Cross-scale nonlinear lineage: NOT supported for the tested 4B->1.7B Mixture-KD recipe. Treat as a stress-test limitation, not a success claim. Same-scale diagnostics show the failure is strongly representation-dependent, but LOSO bridge did not generalize.
- C4 Functional validity: SUPPORTED. FAS-Shapley mean Spearman 0.5556 vs construction 0.1111 across 9 cells; 9/9 wins/ties. Absolute = 0.5962, so do not claim universal FAS superiority on this metric.
- C5 Matched calibration/selective decision: supported in Stage4 G0 family; finite-sample wording must stay restricted to matched G0.
- C6 Evaluator robustness: SUPPORTED for MPNet/MiniLM/BGE support/ranking. Support Jaccard=1.0 and coordinate Spearman=1.0 vs primary across tested alternative encoders. No conformal guarantee transfer.
- C8 Broad open-set ancestry detection: NOT supported by frozen Stage7 gate. AUROC=0.7460; unknown Bank-Insufficient=0.5238; known false Bank-Insufficient=0.0. Keep the selective gates as safeguards and present open-set as a boundary analysis.

## Stage7 negative controls
- within-domain probe-pair shuffle collapsed all 3 seeds, with markedly increased residuals and Bank-Insufficient outcomes.
- exact base-zero sanity produced signal_p=1.0 and zero false-signal rate.
- genealogy-label permutation is diagnostic only and did not uniformly collapse support F1; do not use it as the sole shortcut-control headline.

## Final remaining experiment: Stage8 matched-scale deep ancestry
Stage8 tests whether ancestry evidence survives transformation depth when representation scale is held fixed. It does not revise the cross-scale C3 outcome.
Trajectory: 4B sibling parents -> DARE-TIES merge -> base-neutral auxiliary SFT -> INT4 inference -> same-scale 4B response-KD student.
Report Parent-F1, true-parent mass, residual, Active/Random comparison, Absolute baseline, and trajectory across D1-D4. Treat Stage4 p-values on this G1 chain as transfer diagnostics only.

## Manuscript positioning if Stage8 passes
Use a scoped statement: functional ancestry remains recoverable through multiple nonlinear/post-training transformations within a matched Qwen3-4B representation family, while cross-scale KD reveals a clear limitation.

## Manuscript positioning if Stage8 fails
Do not patch. Present the depth trajectory as an empirical boundary and restrict the principal claim to direct/shallower matched-family descendants. The paper remains supported by C1/C2/C4/C5/C6 plus rigorous failure analysis.

## Finalization rule
After Stage8 there should be no new exploratory experiment stage. Integrate all confirmed and boundary evidence into the TPAMI manuscript and supplementary material.
