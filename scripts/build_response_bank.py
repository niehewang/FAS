#!/usr/bin/env python3
"""Build centered IFRF task fields from precomputed output embeddings.

Each model NPZ must contain:
  base_embeddings   [N, m, d] or [N, d] for q
  edited_embeddings [N, m, d] or [N, d] for q ⊕ delta

Usage:
  python scripts/build_response_bank.py \
    --base base_model.npz \
    --ancestors math.npz code.npz medical.npz science.npz \
    --names Math Code Medical Science \
    --output response_bank.npz
"""
from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from fas_core.response import mean_embedding


def response(z):
    b=np.asarray(z["base_embeddings"],dtype=float)
    e=np.asarray(z["edited_embeddings"],dtype=float)
    if b.ndim==2: return e-b
    return e.mean(axis=1)-b.mean(axis=1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--ancestors",nargs="+",required=True)
    ap.add_argument("--names",nargs="+")
    ap.add_argument("--output",default="response_bank.npz")
    args=ap.parse_args()
    base_r=response(np.load(args.base))
    fields=[]
    for path in args.ancestors:
        r=response(np.load(path))
        if r.shape!=base_r.shape: raise ValueError(f"shape mismatch: {path} {r.shape} vs base {base_r.shape}")
        fields.append(r-base_r)
    task_fields=np.stack(fields,axis=1)  # [N,K,d]
    names=args.names or [Path(x).stem for x in args.ancestors]
    if len(names)!=len(fields): raise ValueError("--names length must match --ancestors")
    np.savez_compressed(args.output,task_fields=task_fields,ancestor_names=np.asarray(names))
    print(f"saved {args.output}: task_fields={task_fields.shape}")

if __name__=="__main__": main()
