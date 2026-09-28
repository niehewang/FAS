#!/usr/bin/env python3
"""Subset a full response NPZ to selected global probe indices without new model queries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--selected',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    z=np.load(args.input,allow_pickle=False)
    sel=[int(x) for x in json.loads(Path(args.selected).read_text(encoding='utf-8'))['selected']]
    n=z['base_embeddings'].shape[0]
    if any(i<0 or i>=n for i in sel): raise SystemExit('selected index outside response NPZ')
    payload={}
    for k in z.files:
        a=z[k]
        if k in {'base_embeddings','edited_embeddings','probe_ids','global_probe_indices'} and getattr(a,'shape',()) and a.shape[0]==n:
            payload[k]=a[sel]
        else:
            payload[k]=a
    payload['global_probe_indices']=np.asarray(sel,dtype=np.int64)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**payload)
    print(f'{args.input}: {n} -> {len(sel)} probes -> {out}')
if __name__=='__main__':main()
