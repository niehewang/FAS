#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--names',required=True,help='comma-separated ancestor names to keep');ap.add_argument('--output',required=True);args=ap.parse_args()
    z=np.load(args.input,allow_pickle=False);fields=np.asarray(z['task_fields']);names=[str(x) for x in z['ancestor_names'].tolist()]
    keep=[x.strip() for x in args.names.split(',') if x.strip()];idx=[]
    for k in keep:
        if k not in names: raise SystemExit(f'{k} not in ancestor_names={names}')
        idx.append(names.index(k))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,task_fields=fields[:,idx,:],ancestor_names=np.asarray(keep))
    print(out)
if __name__=='__main__':main()
