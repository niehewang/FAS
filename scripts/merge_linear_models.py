#!/usr/bin/env python3
"""Exact weighted linear merge for same-architecture checkpoints.

This is deliberately a transparent reference implementation for controlled L1
experiments. For TIES/DARE use mergekit (see configs/merges/README.md) so the
published implementation is not silently reimplemented differently.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--normalize', action='store_true')
    args=ap.parse_args()
    if len(args.models)!=len(args.weights): raise ValueError('models/weights mismatch')
    w=list(args.weights)
    if args.normalize:
        s=sum(w)
        if s==0: raise ValueError('zero sum weights')
        w=[x/s for x in w]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base=AutoModelForCausalLM.from_pretrained(args.models[0],torch_dtype=torch.float32,device_map='cpu',trust_remote_code=True)
    state=base.state_dict()
    for k in state: state[k].zero_()
    for path,weight in zip(args.models,w):
        m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,device_map='cpu',trust_remote_code=True)
        sd=m.state_dict()
        if sd.keys()!=state.keys(): raise ValueError(f'architecture mismatch: {path}')
        for k in state:
            state[k].add_(sd[k], alpha=float(weight))
        del m, sd
    base.load_state_dict(state)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    base.save_pretrained(out,safe_serialization=True,max_shard_size='5GB')
    AutoTokenizer.from_pretrained(args.models[0],trust_remote_code=True).save_pretrained(out)
    (out/'merge_metadata.json').write_text(json.dumps({'models':args.models,'weights':w,'method':'linear'},indent=2),encoding='utf-8')
    print(out)
if __name__=='__main__': main()
