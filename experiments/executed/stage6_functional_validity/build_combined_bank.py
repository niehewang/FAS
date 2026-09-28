#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
DOMS=['Math','Code','Medical','Science']; EPS=1e-12

def delta(z):
    b=np.asarray(z['base_embeddings'],float); e=np.asarray(z['edited_embeddings'],float)
    if b.ndim==3: b=b.mean(1); e=e.mean(1)
    return e-b

def gids(z): return [int(x) for x in np.asarray(z['global_probe_indices']).reshape(-1)] if 'global_probe_indices' in z else list(range(delta(z).shape[0]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage3',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    S=Path(a.stage3); zb=np.load(S/'shared/base_full.npz'); db=delta(zb); gb=gids(zb); mb={g:i for i,g in enumerate(gb)}
    banks={d:np.load(S/f's{a.seed}/banks/{d.lower()}.npz') for d in DOMS}; maps={d:{g:i for i,g in enumerate(gids(banks[d]))} for d in DOMS}; de={d:delta(banks[d]) for d in DOMS}
    common=[g for g in gb if all(g in maps[d] for d in DOMS)]
    if len(common)!=len(gb): raise ValueError(f'combined bank missing gids: base={len(gb)} common={len(common)}')
    F=np.stack([np.stack([de[d][maps[d][g]]-db[mb[g]] for d in DOMS],axis=0) for g in common],axis=0)
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(p,task_fields=F.astype(np.float32),ancestor_names=np.asarray(DOMS),global_probe_indices=np.asarray(common,dtype=np.int64))
    print(f'{p}: {F.shape}')
if __name__=='__main__':main()
