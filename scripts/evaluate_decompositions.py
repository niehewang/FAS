#!/usr/bin/env python3
"""Compute FAS support/coordinate metrics from row-level decomposition outputs.

Input CSV columns (minimum):
  target_id, method, setting, parent_names, true_support, true_pi, pred_pi
Optional:
  seed, queries, rho, p_open

List-valued columns accept either JSON arrays or semicolon-separated strings.
`true_pi` and `pred_pi` must align with `parent_names`. For settings without a
meaningful coordinate target, leave true_pi empty; support metrics are still
computed when true_support is provided.

Outputs:
  <prefix>_rows.csv      row-level metrics
  <prefix>_summary.csv   mean/std/count by method/setting
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math
import numpy as np


def parse_list(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    if s.startswith("["):
        return [str(x) for x in json.loads(s)]
    return [x for x in s.split(";") if x != ""]


def parse_float_list(s: str) -> np.ndarray | None:
    vals = parse_list(s)
    if not vals:
        return None
    return np.asarray([float(x) for x in vals], dtype=float)


def safe_float(s: str) -> float | None:
    try:
        if s is None or str(s).strip() == "":
            return None
        return float(s)
    except Exception:
        return None


def support_metrics(true: set[str], pred: set[str]) -> tuple[float,float,float,float]:
    tp = len(true & pred); fp = len(pred - true); fn = len(true - pred)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not true else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not pred else 0.0)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    exact = float(true == pred)
    return precision, recall, f1, exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output-prefix", default="results/derived/decomposition")
    ap.add_argument("--support-threshold", type=float, default=0.01)
    ap.add_argument("--include-nondecomposable", action="store_true",
                    help="diagnostic only: score coordinates/support even when selective state is not decomposable")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for r in rows:
        state = (r.get("state", "") or "").strip().lower()
        eligible = args.include_nondecomposable or state in ("", "decomposable")
        names = parse_list(r.get("parent_names", ""))
        pred_pi = parse_float_list(r.get("pred_pi", ""))
        true_pi = parse_float_list(r.get("true_pi", ""))
        if pred_pi is not None and names and len(pred_pi) != len(names):
            raise ValueError(f"{r.get('target_id')}: pred_pi length != parent_names")
        if true_pi is not None and names and len(true_pi) != len(names):
            raise ValueError(f"{r.get('target_id')}: true_pi length != parent_names")
        true_support = set(parse_list(r.get("true_support", "")))
        if not eligible:
            pred_support = set()
        elif r.get("pred_support", "").strip():
            pred_support = set(parse_list(r["pred_support"]))
        elif pred_pi is not None:
            pred_support = {n for n,p in zip(names,pred_pi) if p > args.support_threshold}
        else:
            pred_support = set()
        if eligible:
            p, rec, f1, exact = support_metrics(true_support, pred_support)
            l1 = float(np.abs(pred_pi - true_pi).sum()) if pred_pi is not None and true_pi is not None else math.nan
            e2e_f1, e2e_exact = f1, exact
        else:
            p = rec = f1 = exact = l1 = math.nan
            # For a bank-complete benchmark target with non-empty ground-truth
            # ancestry, abstaining is an end-to-end support-recovery failure.
            # This prevents selective methods from improving headline F1 merely
            # by rejecting hard cases. Coordinate L1 remains conditional because
            # no coordinate was reported.
            e2e_f1 = 0.0 if true_support else math.nan
            e2e_exact = 0.0 if true_support else math.nan
        out = dict(r)
        out.update({
            "selective_eligible": int(eligible),
            "parent_precision": p, "parent_recall": rec, "parent_f1": f1,
            "exact_support": exact, "coordinate_l1": l1,
            "end_to_end_parent_f1": e2e_f1,
            "end_to_end_exact_support": e2e_exact,
        })
        out_rows.append(out)

    prefix = Path(args.output_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    row_path = prefix.with_name(prefix.name + "_rows.csv")
    fields = list(out_rows[0].keys()) if out_rows else []
    with row_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)

    groups: dict[tuple[str,str], list[dict]] = {}
    for r in out_rows:
        groups.setdefault((r.get("method", ""), r.get("setting", "")), []).append(r)
    summary=[]
    for (method, setting), grp in sorted(groups.items()):
        n_decomp=sum(int(g.get("selective_eligible",1)) for g in grp)
        states=[(g.get("state","") or "").strip().lower() for g in grp]
        row={"method":method,"setting":setting,"n":len(grp),"n_decomposable":n_decomp,
             "decomposable_rate":n_decomp/len(grp) if grp else "",
             "low_signal_rate":sum(x=="low_signal" for x in states)/len(grp) if grp else "",
             "bank_insufficient_rate":sum(x=="bank_insufficient" for x in states)/len(grp) if grp else ""}
        for key in ["parent_precision","parent_recall","parent_f1","exact_support","coordinate_l1","end_to_end_parent_f1","end_to_end_exact_support"]:
            vals=np.asarray([float(g[key]) for g in grp if g[key] not in ("",None) and np.isfinite(float(g[key]))], dtype=float)
            row[key+"_mean"] = float(vals.mean()) if len(vals) else ""
            row[key+"_std"] = float(vals.std(ddof=1)) if len(vals)>1 else (0.0 if len(vals)==1 else "")
        summary.append(row)
    sum_path=prefix.with_name(prefix.name+"_summary.csv")
    sfields=list(summary[0].keys()) if summary else []
    with sum_path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=sfields); w.writeheader(); w.writerows(summary)
    print(f"Wrote {row_path} and {sum_path}")


if __name__ == "__main__":
    main()
