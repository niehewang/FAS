#!/usr/bin/env python3
"""Fail if a compiled PDF exceeds an internal page budget."""
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path


def pdf_pages(path: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.STDOUT)
    m = re.search(r"^Pages:\s+(\d+)", out, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not read page count from {path}")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main.pdf")
    ap.add_argument("--limit", type=int, default=18, help="hard submission working ceiling")
    ap.add_argument("--soft", type=int, default=12, help="standard regular-paper length before MOPC")
    ap.add_argument("--target", type=int, default=15, help="internal full-study manuscript target")
    args = ap.parse_args()
    path = Path(args.main)
    n = pdf_pages(path)
    status = "OK" if n <= args.limit else "FAIL"
    print(f"{status}: {path} has {n} pages (soft regular length {args.soft}; internal full-study target {args.target}; hard working ceiling {args.limit}).")
    if n > args.limit:
        raise SystemExit(2)
    if n > args.soft:
        print("WARN: manuscript exceeds the 12-page regular-paper length and may incur mandatory overlength charges if accepted in this form.")
    if n > args.target:
        print("NOTE: manuscript is above the current internal full-study target; move nonessential detail to supplementary before submission.")

if __name__ == "__main__":
    main()
