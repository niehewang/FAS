# Executed experiment snapshots

This directory archives the exact late-stage scripts/freeze files that generated the final scientific evidence. It complements the reusable implementation under `code/` and `scripts/`.

Included:

- `stage2_deterministic_pilot`
- `stage5_l5_mixture_kd`
- `stage5d_crossscale_anchor`
- `stage5e_scale_matched17_pilot`
- `stage5f_scale_matched17_multiseed`
- `stage5g_loso_scaleaware_bridge`
- `stage6_functional_validity`
- `stage7_robust_audit`
- `stage8_deep_ancestry`

Stage 5H is intentionally absent: it was designed during development but **cancelled before execution** and is not part of the final paper evidence.

Machine-specific absolute server paths have been sanitized to `FAS_WORK_ROOT` / `FAS_SHARED_ROOT`; frozen scientific settings are unchanged.
