#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--probes',required=True);ap.add_argument('--selection',action='append',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
rows=[json.loads(x) for x in Path(a.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
sel=[]
for p in a.selection:
    o=json.loads(Path(p).read_text()); sel += [int(x) for x in o['selected']]
ids=sorted(set(sel)); out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as f:
    for i in ids:
        r=dict(rows[i]);r['global_probe_index']=i;f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(len(ids),out)
