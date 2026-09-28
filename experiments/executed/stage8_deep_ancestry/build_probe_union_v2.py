#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from jsonl_utils_v1 import strict_read_jsonl,atomic_write_jsonl
ap=argparse.ArgumentParser();ap.add_argument('--probes',required=True);ap.add_argument('--selection',action='append',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
rows=strict_read_jsonl(a.probes);sel=[]
for p in a.selection:
    o=json.loads(Path(p).read_text(encoding='utf-8'));sel += [int(x) for x in o['selected']]
ids=sorted(set(sel));recs=[]
for i in ids:
    r=dict(rows[i]);r['global_probe_index']=i;recs.append(r)
atomic_write_jsonl(a.output,recs);print(len(ids),a.output)
