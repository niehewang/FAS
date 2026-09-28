#!/usr/bin/env python3
"""Selective FAS decomposition for one black-box target.

The default paper-facing semantics are deliberately conservative:
  1) compute ancestry-specific signal ||y|| and calibrate it against base-like
     `cal-signal` controls;
  2) if signal is detected, solve the NNLS FAS decomposition;
  3) calibrate the residual against bank-complete `cal-open` descendants;
  4) report ancestry coordinates only in the Decomposable state.

This prevents a near-base target from receiving a spuriously sharp simplex
coordinate and prevents a bank-insufficient target from being forced into the
closed candidate set.

Calibration files may be CSV/TSV (column defaults to `score`), JSON lists or
objects containing a score list, NPY arrays, or NPZ files containing the named
column/key.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fas_core.decompose import fas_decompose
from fas_core.conformal import conformal_pvalue
from fas_core.selective import signal_score, selective_decision


def _score_vector(path: str, key: str = "score") -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p)
    elif suf == ".npz":
        z = np.load(p)
        if key not in z:
            raise KeyError(f"{p}: NPZ lacks key {key!r}; available={list(z.keys())}")
        arr = z[key]
    elif suf in {".csv", ".tsv"}:
        delim = "\t" if suf == ".tsv" else ","
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delim))
        if not rows or key not in rows[0]:
            raise KeyError(f"{p}: table lacks column {key!r}")
        arr = [float(r[key]) for r in rows if str(r.get(key, "")).strip()]
    elif suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if key not in obj:
                raise KeyError(f"{p}: JSON object lacks key {key!r}")
            obj = obj[key]
        arr = obj
    else:
        raise ValueError(f"unsupported calibration format: {p}")
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"no finite calibration scores in {p}")
    return arr


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="NPZ containing A [D,K] and y [D]")
    ap.add_argument("--target-id", default="", help="stable target/run identifier written into output JSON")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="support threshold on relative FAS coordinate pi; tune only on cal-support")
    ap.add_argument("--signal-calibration",
                    help="cal-signal score file (base-like controls). Required for paper-facing selective output.")
    ap.add_argument("--signal-key", default="score")
    ap.add_argument("--open-calibration",
                    help="cal-open residual score file (bank-complete descendants). Required after signal passes.")
    ap.add_argument("--open-key", default="score")
    ap.add_argument("--open-score", choices=["relative_residual", "residual_norm"], default="relative_residual")
    ap.add_argument("--alpha-signal", type=float, default=0.05)
    ap.add_argument("--alpha-open", type=float, default=0.05)
    ap.add_argument("--calibration-regime", choices=["matched","g0_transfer","g1_transfer","exploratory"], default="matched",
                    help="matched is the only regime carrying the finite-sample conformal interpretation; transfer regimes are stress tests")
    ap.add_argument("--unsafe-report-without-calibration", action="store_true",
                    help="debug only: expose coordinates without selective calibration; never use in paper results")
    ap.add_argument("--output", default="decomposition.json")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    A = np.asarray(z["A"], dtype=float)
    y = np.asarray(z["y"], dtype=float).reshape(-1)
    names = [str(x) for x in z["ancestor_names"].tolist()] if "ancestor_names" in z else [f"ancestor_{i}" for i in range(A.shape[1])]
    if len(names) != A.shape[1]:
        raise ValueError("ancestor_names length does not match A columns")

    weights = None
    if "weights" in z:
        weights = np.asarray(z["weights"], dtype=float)
    elif "whitening" in z:
        weights = np.asarray(z["whitening"], dtype=float)
    dec = fas_decompose(A, y, weights=weights, support_threshold=args.threshold)
    sig_score = signal_score(y, weights=weights)
    out = {k: _serializable(v) for k, v in dec.items()}
    out.update({
        "target_id": args.target_id,
        "ancestor_names": names,
        "support_names": [names[i] for i in dec["support"]],
        "signal_score": float(sig_score),
        "support_threshold": float(args.threshold),
        "alpha_signal": float(args.alpha_signal),
        "alpha_open": float(args.alpha_open),
        "open_score_name": args.open_score,
        "open_score": float(dec[args.open_score]),
        "weighted_geometry": bool(weights is not None),
        "calibration_regime": args.calibration_regime,
        "finite_sample_guarantee": bool(args.calibration_regime == 'matched'),
    })

    # Without signal calibration we intentionally avoid a publication-style
    # ancestry claim. The unsafe flag exists only for early engineering checks.
    if not args.signal_calibration:
        out.update({
            "state": "uncalibrated",
            "report_coordinates": bool(args.unsafe_report_without_calibration),
            "signal_p": None,
            "open_p": None,
        })
        if not args.unsafe_report_without_calibration:
            out["reported_pi"] = None
            out["reported_support"] = None
        else:
            warnings.warn("Reporting coordinates without calibration is unsafe and must not be used in paper results.")
            out["reported_pi"] = out["pi"]
            out["reported_support"] = out["support_names"]
    else:
        sig_cal = _score_vector(args.signal_calibration, args.signal_key)
        p_signal = conformal_pvalue(sig_cal, sig_score)
        # Low-signal targets do not proceed to the bank-sufficiency test.
        if p_signal >= args.alpha_signal:
            decision = selective_decision(p_signal, alpha_signal=args.alpha_signal, alpha_open=args.alpha_open)
        else:
            if not args.open_calibration:
                raise SystemExit("signal gate passed but --open-calibration was not provided")
            open_cal = _score_vector(args.open_calibration, args.open_key)
            p_open = conformal_pvalue(open_cal, dec[args.open_score])
            decision = selective_decision(
                p_signal, p_open,
                alpha_signal=args.alpha_signal,
                alpha_open=args.alpha_open,
            )
        out.update(decision)
        if decision["report_coordinates"]:
            out["reported_pi"] = out["pi"]
            out["reported_support"] = out["support_names"]
        else:
            out["reported_pi"] = None
            out["reported_support"] = None

    if args.calibration_regime != 'matched' and args.signal_calibration:
        warnings.warn(f"calibration_regime={args.calibration_regime}: p-values are transfer diagnostics, not a matched-exchangeability finite-sample guarantee")

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "state": out["state"],
        "signal_p": out.get("signal_p"),
        "open_p": out.get("open_p"),
        "report_coordinates": out["report_coordinates"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
