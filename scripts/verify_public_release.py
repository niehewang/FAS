#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "paper_evidence"
required = [
    "FAS_TPAMI_RESULT_LEDGER_20260927_FINAL.json",
    "FAS_TPAMI_EXPERIMENT_FREEZE_20260927_FINAL.md",
    "stage3_confirmatory_aggregate_results.json",
    "stage4_g0_aggregate_calibration.json",
    "FAS_STAGE6_FUNCTIONAL_VALIDITY_AGGREGATE_20260927.json",
    "FAS_STAGE7_ROBUST_AUDIT_AGGREGATE_20260927.json",
    "FAS_STAGE8_DEEP_ANCESTRY_AGGREGATE_20260927.json",
]
missing = [x for x in required if not (E / x).exists()]
if missing:
    raise SystemExit(f"Missing evidence files: {missing}")

ledger = json.loads((E / required[0]).read_text(encoding="utf-8"))
assert ledger["experiment_phase"] == "frozen"
assert ledger["stage8"]["gate"]["pass"] is True
assert ledger["claims"]["C4"]["status"] == "supported"
assert ledger["claims"]["C6"]["status"] == "supported"
assert ledger["claims"]["C3"]["status"] == "not_supported"
assert ledger["claims"]["C8"]["status"] == "not_supported_broadly"

# CSVs used for final paper figures/tables must be readable and non-empty.
for name in [
    "FAS_STAGE6_FUNCTIONAL_VALIDITY_ROWS_20260927.csv",
    "FAS_STAGE7_ENCODER_SENSITIVITY_ROWS_20260927.csv",
    "FAS_STAGE7_OPEN_SET_ROWS_20260927.csv",
    "FAS_STAGE8_DEEP_ANCESTRY_ROWS_20260927.csv",
]:
    p = E / name
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"Empty CSV: {name}"

print("FAS_PUBLIC_RELEASE_OK")
print("experiment_phase=frozen")
print("stage8_gate=PASS")
print("C3_cross_scale=NOT_SUPPORTED")
print("C8_broad_open_set=NOT_SUPPORTED")
