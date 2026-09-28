# Merge recipes

For **controlled linear merges**, use `scripts/merge_linear_models.py`; it is intentionally transparent and records exact weights.

For TIES/DARE-TIES, use the current official `mergekit` release rather than an in-repository reimplementation. This avoids small implementation differences becoming an experimental confound. Record:

- mergekit version/commit;
- exact YAML recipe;
- all model checkpoint hashes;
- density / normalization / weight parameters.

A final experiment run should copy each executed mergekit YAML into this directory and add its path to the genealogy manifest.
