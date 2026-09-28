#!/usr/bin/env python3
"""Expand configs/experiment_matrix.yaml into a deterministic run manifest CSV.

The manifest is deliberately model-agnostic: it records scenario, seed, parent
set, construction parameters, student scale, and post-processing. Concrete
checkpoint paths can be added later without changing experiment IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path
from typing import Any
import yaml


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_matrix.yaml")
    ap.add_argument("--output", default="data/metadata/experiment_run_manifest.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = cfg.get("seeds", [0])
    rows: list[dict[str, str]] = []
    for sc in cfg.get("scenarios", []):
        variants = sc.get("variants", [{}])
        students = sc.get("student_models", [""])
        for vidx, variant in enumerate(variants):
            for student in students:
                for seed in seeds:
                    suffix = variant.get("name", f"v{vidx+1}")
                    run_id = f"{sc['id']}__{suffix}__{student or 'same'}__s{seed}"
                    parents = variant.get("parents", variant.get("known_parents", []))
                    identity = {
                        "scenario": sc["id"], "variant": suffix, "seed": seed,
                        "parents": parents, "unseen_parents": variant.get("unseen_parents", []),
                        "student": student, "construction": sc.get("construction", ""),
                        "weights": variant.get("weights", []), "exposure": variant.get("exposure", []),
                        "router": variant.get("router", ""),
                        "postprocess": variant.get("postprocess", sc.get("postprocess", [])),
                    }
                    run_hash = hashlib.sha256(_j(identity).encode("utf-8")).hexdigest()[:12]
                    rows.append({
                        "run_id": run_id,
                        "run_hash": run_hash,
                        "plan_version": str(cfg.get("version", "")),
                        "scenario": sc["id"],
                        "family": sc.get("family", ""),
                        "variant": suffix,
                        "seed": str(seed),
                        "parents": ";".join(parents),
                        "unseen_parents": ";".join(variant.get("unseen_parents", [])),
                        "student": student,
                        "construction": sc.get("construction", ""),
                        "weights": _j(variant.get("weights", [])),
                        "exposure": _j(variant.get("exposure", [])),
                        "router": str(variant.get("router", "")),
                        "postprocess": ";".join(variant.get("postprocess", sc.get("postprocess", []))),
                        "ground_truth": ";".join(sc.get("ground_truth", [])),
                        "artifact_dir": f"runs/checkpoints/{run_id}",
                        "response_dir": f"runs/responses/{run_id}",
                        "metadata_path": f"runs/metadata/{run_id}.json",
                        "status": "planned",
                        "checkpoint_or_api": "",
                        "notes": "",
                    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} planned runs -> {out}")


if __name__ == "__main__":
    main()
