#!/usr/bin/env python3
"""Select probes from a saved task-field tensor.

Expected NPZ key: task_fields with shape [N probes, K ancestors, d].
By default uses unconstrained greedy D-optimal selection. With ``--balanced``
and a probe JSONL, selection is performed under partition quotas using a
frozen metadata field such as ``domain`` or ``intervention_type``.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from fas_core.response import probe_blocks
from fas_core.active_probe import greedy_logdet_select, greedy_logdet_select_partitioned


def load_groups(path: str, key: str, n: int):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if ln: rows.append(json.loads(ln))
    if len(rows)!=n:
        raise ValueError(f'probe metadata rows={len(rows)} != task_fields probes={n}')
    groups=[]
    for i,r in enumerate(rows):
        if key not in r or r[key] in (None,''):
            raise ValueError(f'missing group key {key!r} at probe row {i}')
        groups.append(str(r[key]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--output", default="selected_probes.json")
    ap.add_argument("--balanced", action="store_true", help="use partition-quota Balanced D-optimal selection")
    ap.add_argument("--probe-jsonl", help="probe metadata JSONL aligned to task_fields")
    ap.add_argument("--group-key", default="domain", help="metadata key used as partition group")
    args = ap.parse_args()
    fields = np.load(args.npz)["task_fields"]
    blocks, norms = probe_blocks(fields)
    meta={"budget":args.budget,"ridge":args.ridge,"balanced":bool(args.balanced)}
    if args.balanced:
        if not args.probe_jsonl:
            raise SystemExit('--balanced requires --probe-jsonl')
        groups=load_groups(args.probe_jsonl,args.group_key,len(blocks))
        uniq=list(dict.fromkeys(groups)); base,extra=divmod(args.budget,len(uniq))
        quotas={g:base+(i<extra) for i,g in enumerate(uniq)}
        selected,gains=greedy_logdet_select_partitioned(blocks,args.budget,groups,quotas=quotas,ridge=args.ridge)
        counts={g:0 for g in uniq}
        for i in selected: counts[groups[i]]+=1
        meta.update({"group_key":args.group_key,"groups":uniq,"quotas":quotas,"selected_group_counts":counts,"probe_jsonl":args.probe_jsonl})
    else:
        selected,gains=greedy_logdet_select(blocks,args.budget,args.ridge)
    meta.update({"selected":selected,"gains":gains,"norms":norms.tolist()})
    Path(args.output).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(args.output)

if __name__ == "__main__":
    main()
