# Expert training data

`prepare_pilot_datasets.py` can create a small first-pass Math/Code/Medical/Science corpus from benchmark *training* splits. These are only for the mechanism/sanity stage.

For final TPAMI experiments, lock a larger licensed corpus per domain and record:
- exact dataset name/revision/license;
- preprocessing script/config hash;
- number of examples/tokens;
- deduplication against probe/evaluation sets;
- any excluded benchmark-derived records.

Do not train on held-out evaluation or final probe examples.
