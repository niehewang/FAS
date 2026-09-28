# Formal data governance (v0.10)

This file defines the controlled data protocol for the four Qwen3-4B sibling experts. The goal is not to maximize each expert's benchmark score; it is to create four reproducible, domain-distinct functional directions while keeping the training recipe and number of examples fixed.

## Locked training sources

| Domain | Default source | Intended split | License recorded by source | Formal role |
|---|---|---|---|---|
| Math | `nvidia/OpenMathInstruct-2` | `train_1M` | CC BY 4.0 | ancestor training only |
| Code | `m-a-p/Code-Feedback` | `train` | Apache-2.0 | ancestor training only |
| Medical | `openlifescienceai/medmcqa` | `train` | Apache-2.0 | ancestor training only |
| Science | `stemdataset/STEM` | `train`, text-only `subject=science` | Apache-2.0 | ancestor training only |

The preparation script resolves every Hugging Face repository to an immutable commit SHA and stores that SHA in `data/governance/formal_training_manifest.json`. Do not report only `main` in the final paper.

## Frozen probe/evaluation source families

The formal probe seed pool is source-disjoint from the four training repositories:

- Math: MATH test (`EleutherAI/hendrycks_math`).
- Code: HumanEval + MBPP+.
- Medical: PubMedQA labeled subset.
- Science: AI2 ARC Easy/Challenge test.

The math training source is derived from MATH/GSM8K *training* problems, so source-family separation alone is not enough. The project therefore applies a second contamination layer: exact normalized-string matching plus word-ngram Jaccard/containment screening against the frozen probe seed pool. The final paper must report the realized number of removed rows.

## Fixed training size and pseudoreplication control

1. Raw preparation samples 3,600 usable records/domain.
2. The probe seed pool is frozen before contamination filtering.
3. `audit_data_contamination.py` removes exact/high-overlap records.
4. The first 3,000 clean records/domain are retained deterministically.
5. The same four canonical corpora are used for all genealogy seeds 11/23/47; independence comes from independently initialized/trained LoRA adapters and data order, not from silently changing domain content across seeds.
6. All four domains use exactly 3,000 examples and the identical QLoRA optimizer recipe. Prompt/response token counts are recorded and reported; they are not claimed to be identical.

## Commands

```bash
python scripts/prepare_formal_probe_seeds.py
python scripts/prepare_formal_datasets.py
python scripts/audit_data_contamination.py
python scripts/check_formal_data.py
python scripts/generate_seeded_expert_configs.py
```

No formal expert should be trained if `check_formal_data.py` fails.

## Probe candidate review and freeze

`build_formal_interventions.py` creates a deterministic candidate pool from the frozen probe seeds. Automatically edited medical/code/science cases are deliberately tagged `needs_review`. The project does **not** require manual inspection of all candidates: `sample_probe_review.py` draws a stratified 20-row sample/domain (80 rows total). A formal freeze requires at least 10 reviewed rows/domain and a reject rate no larger than 20%; sampled rows explicitly rejected are removed. The final `interventions.jsonl` records rule version, source seed, source revision/license and review status.

The sample-based review is a quality-control gate over deterministic generation rules, not statistical evidence for the paper. If the sampled reject rate is high for a domain/intervention type, repair the rule and regenerate the complete pool rather than selectively editing only the visible examples.

## Reproducibility locks

The preparation stage writes:
- `formal_training_manifest.json`: resolved source revisions and token statistics before contamination filtering;
- `formal_probe_seed_manifest.json`: resolved probe-source revisions and SHA256 of the frozen seed pool;
- `contamination_report.json`: realized exact/ngram removals and clean-row trimming;
- `formal_data_lock.json`: SHA256 of the final 3,000-example/domain corpora;
- `probe_review_sample.tsv`: the human QA sample used to approve/reject deterministic intervention rules.

These manifests should be archived with every formal run. Do not overwrite them after a reported experiment; create a new versioned data lock instead.
