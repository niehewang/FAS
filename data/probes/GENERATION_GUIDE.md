# Intervention probe generation guide

Each row must change one controlled semantic factor while keeping the remainder as invariant as possible. Required fields:
`probe_id`, `domain`, `type`, `base_query`, `edited_query`, `changed_factor`, `source`, `notes`.

Recommended types: Value, Entity, Constraint, Relation, Composition.

Quality workflow:
1. seed from data disjoint from expert training/evaluation;
2. generate candidate single-factor edits with an unrelated generator or rules;
3. run `python scripts/validate_probe_pool.py ...`;
4. manually inspect a stratified sample from every domain/type;
5. freeze the pool before querying descendants.

Avoid paraphrase-only edits, multiple simultaneous semantic changes, answer-format-only changes, and prompts that contain model/vendor names.
