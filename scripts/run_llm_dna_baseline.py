#!/usr/bin/env python3
"""Optional wrapper for the public `llm-dna` package (ICLR 2026 baseline).

Install `requirements-baselines.txt`. This wrapper intentionally delegates DNA
extraction to the authors' package rather than reimplementing RepTrace.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',required=True); ap.add_argument('--names',nargs='+'); ap.add_argument('--dataset',default='rand'); ap.add_argument('--max-samples',type=int,default=100); ap.add_argument('--dna-dim',type=int,default=128); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--output',default='dna_vectors.npz'); args=ap.parse_args()
    try:
        from llm_dna import DNAExtractionConfig, calc_dna
    except Exception as e:
        raise SystemExit('Install requirements-baselines.txt (llm-dna) first') from e
    vecs=[]
    for model in args.models:
        cfg=DNAExtractionConfig(model_name=model,dataset=args.dataset,gpu_id=args.gpu,max_samples=args.max_samples,dna_dim=args.dna_dim,reduction_method='random_projection',trust_remote_code=True)
        res=calc_dna(cfg); vecs.append(np.asarray(res.vector,dtype=np.float32))
    names=args.names or args.models
    if len(names)!=len(vecs): raise ValueError('--names length mismatch')
    np.savez_compressed(args.output,vectors=np.stack(vecs),names=np.asarray(names),models=np.asarray(args.models))
    print(args.output)
if __name__=='__main__': main()
