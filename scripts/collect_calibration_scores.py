#!/usr/bin/env python3
"""Collect signal or bank-sufficiency calibration scores from prepared NPZs.

Input NPZs are outputs of ``prepare_target_decomposition.py`` and contain
``A`` and ``y``. Optional ``weights`` stores a diagonal whitening vector or
full whitening matrix.  The paper-facing protocol is deliberately selective:

* ``signal`` writes ||W y|| for base-like/no-task-delta controls.
* ``open`` first applies the Low-Signal gate using an *independent* cal-signal
  score file, then writes residual scores only for descendants that pass.

This mirrors Theorem 4 in the manuscript: open-set conformal calibration is
conditional on the target having passed the signal gate.  Mixing low-signal
residuals into cal-open is therefore rejected by default.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fas_core.conformal import conformal_pvalue
from fas_core.decompose import fas_decompose
from fas_core.selective import signal_score


def score_vector(path: str, key: str = "score") -> np.ndarray:
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
        raise ValueError(f"unsupported score format: {p}")
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"no finite scores in {p}")
    return arr


def npz_weights(z) -> np.ndarray | None:
    for key in ("weights", "whitening"):
        if key in z:
            return np.asarray(z[key], dtype=float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["signal", "open"])
    ap.add_argument("inputs", nargs="+", help="NPZ paths or glob patterns")
    ap.add_argument("--open-score", choices=["relative_residual", "residual_norm"], default="relative_residual")
    ap.add_argument("--signal-calibration", help="independent cal-signal score file; required for paper-facing open mode")
    ap.add_argument("--signal-key", default="score")
    ap.add_argument("--alpha-signal", type=float, default=0.05)
    ap.add_argument("--unsafe-unconditioned-open", action="store_true",
                    help="debug only: include open calibration residuals without first applying the Low-Signal gate")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.mode == "open" and not args.signal_calibration and not args.unsafe_unconditioned_open:
        raise SystemExit(
            "paper-facing open calibration must be conditioned on the Low-Signal gate; "
            "provide --signal-calibration or use --unsafe-unconditioned-open for debugging only"
        )
    sig_cal = score_vector(args.signal_calibration, args.signal_key) if args.signal_calibration else None

    files: list[str] = []
    for x in args.inputs:
        matches = glob.glob(x)
        files.extend(matches if matches else [x])
    files = sorted(set(files))

    rows = []
    skipped_low_signal = 0
    for x in files:
        p = Path(x)
        if not p.exists():
            raise FileNotFoundError(p)
        z = np.load(p, allow_pickle=False)
        y = np.asarray(z["y"], dtype=float).reshape(-1)
        weights = npz_weights(z)
        sig = signal_score(y, weights=weights)
        p_sig = conformal_pvalue(sig_cal, sig) if sig_cal is not None else None

        if args.mode == "signal":
            score = sig
            state = "signal_calibration"
        else:
            if p_sig is not None and p_sig >= args.alpha_signal:
                skipped_low_signal += 1
                continue
            dec = fas_decompose(np.asarray(z["A"], dtype=float), y, weights=weights)
            score = float(dec[args.open_score])
            state = "signal_pass"

        rows.append({
            "target_id": p.stem,
            "score": float(score),
            "signal_score": float(sig),
            "signal_p": "" if p_sig is None else float(p_sig),
            "state": state,
            "source": str(p),
        })

    if not rows:
        raise SystemExit("no calibration rows remained after filtering")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["target_id", "score", "signal_score", "signal_p", "state", "source"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(
        f"wrote {out}: {len(rows)} {args.mode} calibration scores"
        + (f"; skipped_low_signal={skipped_low_signal}" if args.mode == "open" else "")
    )


if __name__ == "__main__":
    main()
