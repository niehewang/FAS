#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model and save a full checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dtype', default='bfloat16')
    args=ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype=getattr(torch,args.dtype)
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=dtype,device_map='cpu',trust_remote_code=True)
    model=PeftModel.from_pretrained(model,args.adapter)
    model=model.merge_and_unload()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out,safe_serialization=True,max_shard_size='5GB')
    tok=AutoTokenizer.from_pretrained(args.base,trust_remote_code=True)
    tok.save_pretrained(out)
    print(out)
if __name__=='__main__': main()
