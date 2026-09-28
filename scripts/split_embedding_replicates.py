#!/usr/bin/env python3
"""Split stochastic response replicates into bank-select and bank-estimate sets.

Use disjoint ancestor generations for adaptive probe selection and for the final
ancestry dictionary/geometry. This avoids reusing the stochastic noise that made
a probe look attractive during selection (adaptive winner's-curse bias).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--select-count',type=int);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--select-output');ap.add_argument('--estimate-output');args=ap.parse_args()
    p=Path(args.input);z=np.load(p,allow_pickle=False)
    b=np.asarray(z['base_embeddings']);e=np.asarray(z['edited_embeddings'])
    if b.ndim!=3 or e.ndim!=3:raise SystemExit('replicate split requires embeddings shaped [N,m,d]')
    if b.shape!=e.shape:raise ValueError('base/edited shape mismatch')
    m=b.shape[1]
    if m<2:raise SystemExit('need at least two stochastic replicates for bank cross-fitting')
    nsel=args.select_count if args.select_count is not None else m//2
    if not 1<=nsel<m:raise ValueError('--select-count must be between 1 and m-1')
    rng=np.random.default_rng(args.seed);perm=rng.permutation(m);sel=np.sort(perm[:nsel]);est=np.sort(perm[nsel:])
    so=Path(args.select_output or p.with_name(p.stem+'.select.npz'));eo=Path(args.estimate_output or p.with_name(p.stem+'.estimate.npz'))
    meta={k:z[k] for k in z.files if k not in {'base_embeddings','edited_embeddings'}}
    np.savez_compressed(so,base_embeddings=b[:,sel],edited_embeddings=e[:,sel],replicate_indices=sel,**meta)
    np.savez_compressed(eo,base_embeddings=b[:,est],edited_embeddings=e[:,est],replicate_indices=est,**meta)
    print(f'{p}: m={m} -> select={sel.tolist()} estimate={est.tolist()}')
    print(so);print(eo)
if __name__=='__main__':main()
