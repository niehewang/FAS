#!/usr/bin/env python3
"""Evaluate the low-signal gate and write paper-facing negative-control rows.

Input CSV columns:
  split: cal or test
  label: base_like or descendant
  score: ||y|| (or weighted signal norm)
  run_id: optional

The calibration distribution uses only split=cal,label=base_like. For every
held-out test row we compute a conformal p-value for a large signal score.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fas_core.selective import signal_pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV with split,label,score")
    ap.add_argument("--alphas", default="0.01,0.05,0.10")
    ap.add_argument("--output", default="results/derived/signal_gate_metrics.csv")
    ap.add_argument("--paper-output", default="results/templates/negative_controls.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    need = {"split", "label", "score"}
    if not need.issubset(df.columns):
        raise SystemExit(f"missing columns: {sorted(need - set(df.columns))}")
    cal = pd.to_numeric(
        df[(df["split"] == "cal") & (df["label"] == "base_like")]["score"],
        errors="coerce",
    ).dropna().to_numpy()
    if len(cal) == 0:
        raise SystemExit("no cal/base_like signal scores")

    test = df[df["split"] == "test"].copy()
    test["score"] = pd.to_numeric(test["score"], errors="coerce")
    test = test.dropna(subset=["score"])
    test["p_signal"] = [signal_pvalue(cal, x) for x in test["score"]]

    alphas = [float(x) for x in args.alphas.split(",")]
    rows = []
    paper = []
    for a in alphas:
        base = test[test["label"] == "base_like"]
        desc = test[test["label"] == "descendant"]
        false_signal = float((base["p_signal"] < a).mean()) if len(base) else float("nan")
        pass_rate = float((desc["p_signal"] < a).mean()) if len(desc) else float("nan")
        rows.append({"alpha": a, "false_signal_rate": false_signal, "descendant_pass_rate": pass_rate,
                     "n_cal": len(cal), "n_base_test": len(base), "n_desc_test": len(desc)})
        paper.append({"control": f"Low-signal gate (alpha={a:.2f})",
                      "false_positive_or_f1": false_signal,
                      "signal_pass_or_expected": pass_rate,
                      "interpretation": "false-signal / descendant-pass"})

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    pp = ROOT / args.paper_output
    pp.parent.mkdir(parents=True, exist_ok=True)
    # Preserve non-gate rows (e.g. permutation controls) if already filled.
    if pp.exists():
        old = pd.read_csv(pp, keep_default_na=False)
        keep = old[~old["control"].astype(str).str.startswith("Low-signal gate")]
        paper_df = pd.concat([pd.DataFrame(paper), keep], ignore_index=True)
    else:
        paper_df = pd.DataFrame(paper)
    paper_df.to_csv(pp, index=False)
    test.to_csv(out.with_name("signal_gate_scored_targets.csv"), index=False)
    print(f"wrote {out} and {pp}")


if __name__ == "__main__":
    main()
