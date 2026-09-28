FAS Stage6 Functional Validity Exact-Shapley v1.1

Why v1.1 exists
- v1 exited during preflight with exit_code=21 because data/evaluation/pilot_utility.jsonl was a legacy pilot helper output and was never part of the formal Stage1 freeze.
- No Stage6 model evaluation occurred in v1. Therefore v1.1 is an engineering/data-realization amendment made before any Stage6 scientific result.

Scientific design is unchanged
- Formal seeds: 23,47,71.
- Three-parent fixed-coefficient subset game: Math=.2, Medical=.5, Science=.3; all 2^3 subsets; no renormalization.
- Exact Shapley and leave-one-parent-out realized utility contributions.
- Compare construction weights, Absolute Output+NNLS, and domain-conditioned FAS coordinates.
- Frozen C4 gate is unchanged.

Formal held-out utility v1
- Uses the exact benchmark repositories/revisions frozen in Stage1.
- Excludes all examples already present in data/probes/formal_seed_pool.jsonl by stable seed_id.
- Screens against the final 3000-example/domain training corpora using the same Stage1 8-gram contamination thresholds.
- Selects 96 deterministic items/domain before any Stage6 model evaluation.
- Returns heldout_utility.jsonl and heldout_utility_manifest.json with SHA256/provenance.

Run: chmod +x run_fas_server.sh && ./run_fas_server.sh
Return: ${FAS_WORK_ROOT}/FAS_SERVER_RETURN_STAGE6_FUNCTIONAL_VALIDITY_V1_1.tar.gz
