#!/usr/bin/env python3
"""Estimate a stable diagonal whitening vector from repeated black-box outputs.

The input NPZ must contain ``base_embeddings`` and ``edited_embeddings`` shaped
[N probes, m stochastic replicates, d].  For each selected probe/coordinate we
estimate the variance of the intervention difference across replicates.  Since
FAS decomposes the *mean* response over m samples, the diagonal variance of the
mean is estimated as sample_variance / m.  The returned weight is
1/sqrt(var_mean + ridge), optionally clipped to robust quantiles.

This is deliberately a diagonal reference implementation: with typical
embedding dimension d >> number of stochastic replicates m, an unregularized
full covariance is singular and easy to overfit.  Full/shrinkage block
covariance can be added as a later ablation without changing the FAS API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',required=True,help='NPZ with repeated base/edited embeddings [N,m,d]')
    ap.add_argument('--selected',required=True,help='selected_probes.json')
    ap.add_argument('--ridge',type=float,default=1e-6)
    ap.add_argument('--clip-low',type=float,default=0.01)
    ap.add_argument('--clip-high',type=float,default=0.99)
    ap.add_argument('--output',default='weights.npy')
    args=ap.parse_args()
    z=np.load(args.reference,allow_pickle=False)
    b=np.asarray(z['base_embeddings'],dtype=float); e=np.asarray(z['edited_embeddings'],dtype=float)
    if b.ndim!=3 or e.ndim!=3 or b.shape!=e.shape:
        raise SystemExit('reference embeddings must share shape [N,m,d] with m>=2')
    if b.shape[1]<2: raise SystemExit('at least two stochastic replicates are required')
    selected=json.loads(Path(args.selected).read_text(encoding='utf-8'))['selected']
    delta=e-b
    var=delta.var(axis=1,ddof=1)/delta.shape[1]  # [N,d] variance of sample mean
    v=np.concatenate([var[i] for i in selected],axis=0)
    w=1.0/np.sqrt(np.maximum(v,args.ridge))
    if 0 <= args.clip_low < args.clip_high <= 1:
        lo,hi=np.quantile(w,[args.clip_low,args.clip_high]); w=np.clip(w,lo,hi)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);np.save(out,w)
    meta={'reference':args.reference,'selected':args.selected,'m':int(delta.shape[1]),'d':int(delta.shape[2]),'ridge':args.ridge,'clip':[args.clip_low,args.clip_high],'weight_min':float(w.min()),'weight_median':float(np.median(w)),'weight_max':float(w.max())}
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'wrote {out}: shape={w.shape}; min/median/max={w.min():.4g}/{np.median(w):.4g}/{w.max():.4g}')

if __name__=='__main__': main()
