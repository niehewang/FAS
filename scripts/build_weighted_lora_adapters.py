#!/usr/bin/env python3
"""Build many weighted LoRA calibration descendants in one model-loading session.

The script consumes one or more calibration-plan CSVs.  Each row must contain
``calibration_id``, ``parents`` (JSON list) and ``weights`` (JSON mapping).
Weights are preserved exactly by default; use ``--normalize-weights`` only when
the experimental definition explicitly calls for renormalized subsets.
All source adapters are expected to share the same base model and PEFT task.

We intentionally use PEFT's ``add_weighted_adapter`` with ``combination_type=cat``
by default.  Concatenation represents the weighted sum of LoRA delta matrices
without introducing cross terms and avoids materializing a full 4B checkpoint.
The base model is loaded once, source adapters are loaded once, and each merged
adapter is saved separately under ``--output-dir/<calibration_id>``.

This is a construction helper, not part of FAS itself.  Final experiments must
record the installed PEFT/Transformers versions in the reproducibility manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable


def parse_mapping(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for x in items:
        if "=" not in x:
            raise ValueError(f"--adapter expects NAME=PATH, got {x!r}")
        k, v = x.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with Path(p).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("parents") and r.get("weights"):
                    rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--plan", nargs="+", required=True, help="calibration plan CSV(s)")
    ap.add_argument("--adapter", action="append", default=[], help="NAME=PATH; repeat for each parent")
    ap.add_argument("--output-dir", default="runs/calibration/adapters")
    ap.add_argument("--combination-type", default="cat", choices=["cat", "linear", "svd"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="cpu", help="default cpu: calibration adapter construction is offline")
    ap.add_argument("--normalize-weights", action="store_true", help="renormalize each row to sum to 1; default preserves the plan coefficients exactly")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    amap = parse_mapping(args.adapter)
    rows = load_rows(args.plan)
    if not rows:
        raise SystemExit("no rows with parents/weights found in plans")

    requested = sorted({p for r in rows for p in json.loads(r["parents"])})
    missing = [p for p in requested if p not in amap]
    if missing:
        raise SystemExit(f"missing --adapter mappings for: {missing}")

    # Delayed heavy imports so repository tests do not require GPU packages.
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    dtype = getattr(torch, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    first = requested[0]
    model = PeftModel.from_pretrained(base, amap[first], adapter_name=first, is_trainable=False)
    for name in requested[1:]:
        model.load_adapter(amap[name], adapter_name=name, is_trainable=False)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    built = 0
    for r in rows:
        cid = r["calibration_id"]
        parents = list(json.loads(r["parents"]))
        wmap = dict(json.loads(r["weights"]))
        weights = [float(wmap[p]) for p in parents]
        s = sum(weights)
        if s <= 0:
            raise ValueError(f"{cid}: non-positive weight sum")
        if args.normalize_weights:
            weights = [x / s for x in weights]
        dest = out_root / cid
        meta = dest / "construction_metadata.json"
        if meta.exists() and not args.overwrite:
            continue

        merged_name = f"merged_{cid}"
        if merged_name in getattr(model, "peft_config", {}):
            model.delete_adapter(merged_name)
        model.add_weighted_adapter(
            adapters=parents,
            weights=weights,
            adapter_name=merged_name,
            combination_type=args.combination_type,
        )
        dest.mkdir(parents=True, exist_ok=True)
        # Current PEFT supports selected_adapters.  If an older supported PEFT
        # is used, fail loudly rather than silently saving source adapters too.
        try:
            model.save_pretrained(str(dest), selected_adapters=[merged_name])
        except TypeError as e:
            raise RuntimeError(
                "installed PEFT lacks save_pretrained(selected_adapters=...); "
                "upgrade to the pinned experiment environment"
            ) from e
        nested = dest / merged_name
        if not (dest / 'adapter_config.json').exists() and (nested / 'adapter_config.json').exists():
            # Non-default PEFT adapters are saved in a named subdirectory.
            # Move that selected adapter to the requested root so downstream
            # PeftModel.from_pretrained(base, dest) works without special cases.
            for child in list(nested.iterdir()):
                target = dest / child.name
                if target.exists():
                    raise RuntimeError(f'{dest}: refusing to overwrite {target.name}')
                child.replace(target)
            nested.rmdir()
        if not (dest / 'adapter_config.json').exists():
            raise RuntimeError(f'{dest}: merged adapter was not saved in a directly loadable layout')
        info = {
            "calibration_id": cid,
            "base": args.base,
            "parents": parents,
            "weights": dict(zip(parents, weights)),
            "weights_normalized": bool(args.normalize_weights),
            "weight_sum": float(sum(weights)),
            "combination_type": args.combination_type,
            "source_adapters": {p: amap[p] for p in parents},
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            },
        }
        meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        model.delete_adapter(merged_name)
        built += 1
        print(f"built {cid} -> {dest}")

    print(json.dumps({"built": built, "requested_rows": len(rows), "output_dir": str(out_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
