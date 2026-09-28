#!/usr/bin/env python3
"""Generate an executable G0 calibration-descendant/audit job.

The generated job implements the paper-facing matched calibration protocol for
same-scale sibling experiments:
  1. materialize only the selected B probes;
  2. create lightweight bank-complete weighted-LoRA descendants for
     cal-support and cal-open;
  3. create independent stochastic base-like cal-signal episodes;
  4. audit every calibration target on selected probes only;
  5. prepare decomposition NPZs;
  6. collect signal/open scores and tune the support threshold.

Heavy KD/deep-chain descendants are deliberately excluded here.  They remain
G1 shift tests unless a separate matched calibration plan is explicitly built.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shq(x: str) -> str:
    return "'" + str(x).replace("'", "'\\''") + "'"


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-dir", default="runs/calibration/plan")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--bank", default="runs/pilot/response/response_bank.estimate.npz")
    ap.add_argument("--selected", default="runs/pilot/selected/probes_B32.json")
    ap.add_argument("--probes", default="data/probes/interventions_pilot.jsonl")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--output", default="jobs/run_g0_calibration.sh")
    ap.add_argument("--run-root", default="runs/calibration", help="artifact root; use runs/calibration/s<seed> for formal genealogy-specific calibration")
    ap.add_argument("--ancestor-root", default="runs/experts", help="ancestor adapter root; contains Math/Code/Medical/Science domain directories")
    args = ap.parse_args()

    plan = ROOT / args.plan_dir
    sig = read(plan / "cal_signal_plan.csv")
    sup = read(plan / "cal_support_plan.csv")
    opn = read(plan / "cal_open_candidates.csv")
    if not (sig and sup and opn):
        raise SystemExit("calibration plan is missing/empty; run generate_calibration_plan.py first")

    run_root=args.run_root.rstrip('/')
    anc_root=args.ancestor_root.rstrip('/')
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        "",
        "# Auto-generated same-scale G0 calibration job.",
        "# This job is intentionally separate from nonlinear G1 KD/deep-chain stress tests.",
        f"BASE=${{BASE:-{shq(args.base)}}}",
        f"ENCODER=${{ENCODER:-{shq(args.encoder)}}}",
        f"SAMPLES=${{SAMPLES:-{args.samples}}}",
        f"TEMPERATURE=${{TEMPERATURE:-{args.temperature}}}",
        f"TOP_P=${{TOP_P:-{args.top_p}}}",
        f"BANK=${{BANK:-{shq(args.bank)}}}",
        f"SELECTED=${{SELECTED:-{shq(args.selected)}}}",
        f"PROBES=${{PROBES:-{shq(args.probes)}}}",
        f"mkdir -p {run_root}/adapters {run_root}/responses {run_root}/prepared {run_root}/scores",
        f"python scripts/subset_probe_pool.py --probes \"$PROBES\" --selected \"$SELECTED\" --output {run_root}/selected_probes.jsonl",
        "",
        "# Build all bank-complete lightweight descendants in one base-model loading session.",
        "python scripts/build_weighted_lora_adapters.py \\",
        "  --base \"$BASE\" \\",
        f"  --plan {shq(str(Path(args.plan_dir)/'cal_support_plan.csv'))} {shq(str(Path(args.plan_dir)/'cal_open_candidates.csv'))} \\",
        f"  --adapter Math={anc_root}/math/adapter \\",
        f"  --adapter Code={anc_root}/code/adapter \\",
        f"  --adapter Medical={anc_root}/medical/adapter \\",
        f"  --adapter Science={anc_root}/science/adapter \\",
        f"  --output-dir {run_root}/adapters",
        "",
        "# One independent anchor realization per target seed is generated below; paired seeds reduce variance.",
    ]

    # Signal episodes: target and anchor are independently sampled from the same base.
    for i, r in enumerate(sig):
        cid = r["calibration_id"]
        seed = int(r["seed"])
        target = f"{run_root}/responses/{cid}.target.npz"
        anchor = f"{run_root}/responses/{cid}.anchor.npz"
        prep = f"{run_root}/prepared/{cid}.npz"
        fam=r.get('family',''); quant = ' --load-in-4bit' if 'int4' in fam else (' --load-in-8bit' if 'int8' in fam else '')
        L += [
            f"echo '=== {cid} signal ==='",
            f"python scripts/run_text_probe_bank.py --model \"$BASE\"{quant} --probes {run_root}/selected_probes.jsonl --encoder \"$ENCODER\" --samples \"$SAMPLES\" --seed {seed} --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\" --paired-seeds --output {shq(target)}",
            f"python scripts/run_text_probe_bank.py --model \"$BASE\"{quant} --probes {run_root}/selected_probes.jsonl --encoder \"$ENCODER\" --samples \"$SAMPLES\" --seed {seed + 10000019} --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\" --paired-seeds --output {shq(anchor)}",
            f"python scripts/prepare_target_decomposition.py --bank \"$BANK\" --base {shq(anchor)} --target {shq(target)} --selected \"$SELECTED\" --target-selected-only --output {shq(prep)}",
        ]

    # Support/open descendants.  Each receives an independent base anchor realization.
    for role, rows in (("support", sup), ("open", opn)):
        for r in rows:
            cid = r["calibration_id"]
            seed = int(r["seed"])
            target = f"{run_root}/responses/{cid}.target.npz"
            anchor = f"{run_root}/responses/{cid}.anchor.npz"
            prep = f"{run_root}/prepared/{cid}.npz"
            adapter = f"{run_root}/adapters/{cid}"
            L += [
                f"echo '=== {cid} {role} ==='",
                f"python scripts/run_text_probe_bank.py --model \"$BASE\" --adapter {shq(adapter)} --probes {run_root}/selected_probes.jsonl --encoder \"$ENCODER\" --samples \"$SAMPLES\" --seed {seed} --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\" --paired-seeds --output {shq(target)}",
                f"python scripts/run_text_probe_bank.py --model \"$BASE\" --probes {run_root}/selected_probes.jsonl --encoder \"$ENCODER\" --samples \"$SAMPLES\" --seed {seed + 10000019} --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\" --paired-seeds --output {shq(anchor)}",
                f"python scripts/prepare_target_decomposition.py --bank \"$BANK\" --base {shq(anchor)} --target {shq(target)} --selected \"$SELECTED\" --target-selected-only --output {shq(prep)}",
            ]

    L += [
        "",
        f"python scripts/collect_calibration_scores.py signal '{run_root}/prepared/csig_*.npz' --output {run_root}/scores/signal_scores.csv",
        f"python scripts/collect_calibration_scores.py open '{run_root}/prepared/copen_*.npz' --signal-calibration {run_root}/scores/signal_scores.csv --alpha-signal 0.05 --output {run_root}/scores/open_scores.csv",
        f"python scripts/calibrate_support_from_prepared.py '{run_root}/prepared/csup_*.npz' --truth-plan " + shq(str(Path(args.plan_dir)/"cal_support_plan.csv")) + f" --output {run_root}/scores/support_threshold.json",
        f"python scripts/check_calibration_resolution.py --actual-signal {run_root}/scores/signal_scores.csv --actual-open {run_root}/scores/open_scores.csv --require-planned-count",
        "echo 'G0 calibration construction/audit complete.'",
    ]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.chmod(0o755)
    print(out)


if __name__ == "__main__":
    main()
