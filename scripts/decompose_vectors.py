#!/usr/bin/env python3
"""Generic NNLS decomposition for baseline vectors (e.g. DNA-Decomp)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ancestors',required=True,help='NPZ vectors [K,d]'); ap.add_argument('--target',required=True,help='NPY or NPZ single vector'); ap.add_argument('--output',default='vector_decomp.json'); args=ap.parse_args()
    z=np.load(args.ancestors); V=np.asarray(z['vectors'],float); names=z['names'].astype(str) if 'names' in z else np.asarray([str(i) for i in range(len(V))])
    t=np.load(args.target)
    if isinstance(t,np.lib.npyio.NpzFile):
        if 'vector' in t:
            y=np.asarray(t['vector'],float).reshape(-1)
        elif 'vectors' in t:
            vv=np.asarray(t['vectors'],float)
            if vv.ndim!=2 or vv.shape[0]!=1:
                raise ValueError(f'target NPZ vectors must have shape [1,d], got {vv.shape}')
            y=vv[0].reshape(-1)
        else:
            raise KeyError(f'target NPZ lacks vector/vectors keys: {list(t.keys())}')
    else:
        y=np.asarray(t,float).reshape(-1)
    if V.shape[1]!=len(y): raise ValueError((V.shape,y.shape))
    # Unit-norm columns so the coefficients reflect direction rather than arbitrary vector scale.
    A=(V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)).T
    yy=y/(np.linalg.norm(y)+1e-12)
    coef,res=nnls(A,yy); pi=coef/(coef.sum()+1e-12)
    out={'names':names.tolist(),'coefficients':coef.tolist(),'coordinates':pi.tolist(),'residual_norm':float(res)}
    Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(args.output)
if __name__=='__main__': main()
