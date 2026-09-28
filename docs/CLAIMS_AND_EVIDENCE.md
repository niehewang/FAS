# Claims and Evidence — v0.19-paper final experiment freeze

This ledger is internal and evidence-facing; publication prose uses positive TPAMI-style framing while preserving the numerical scope below.

- **C1 Active identifiability / geometry — supported.** Stage3: sigma_min 0.5955 vs 0.5144; condition 2.7127 vs 3.2656.
- **C2 Multi-parent functional ancestry decomposition — supported.** L1/L2 Active FAS Parent-F1 0.7111/0.8571 across seeds 23/47/71.
- **C4 Functional validity — supported.** FAS-Shapley rho 0.5556 vs construction 0.1111; 9/9 win/tie; FAS-LOPO rho 0.4444.
- **C5 Selective calibration — supported for matched Stage4 L1.**
- **C6 Evaluator robustness — supported.** MPNet/MiniLM/BGE support Jaccard and coordinate Spearman 1.000.
- **Deep matched-scale ancestry — supported.** D4 Active FAS Parent-F1 0.7937, true-parent mass 0.7453, residual 0.7106, ~100% F1 retention vs D1, 3/3 seeds >=0.60.
- **Cross-scale transport diagnostic.** 4B->1.7B Mixture-KD exposes response-geometry reconfiguration; same-scale surrogate residual 0.6744 vs cross-scale 0.9578.
- **Restricted-bank selective stress.** AUROC 0.746; withheld-Science Bank-Insufficient 52.4%; known false Bank-Insufficient 0%; pair-shuffle collapse 3/3.
- **Absolute Output+NNLS is retained as a strong complementary baseline** throughout support recovery, functional validity and deep ancestry.

Experiment phase frozen after Stage8.
