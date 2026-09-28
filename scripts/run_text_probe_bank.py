#!/usr/bin/env python3
"""Run a JSONL intervention pool against one text model and save output embeddings.

Output NPZ is directly consumable by scripts/build_response_bank.py and contains
base_embeddings / edited_embeddings with shape [N, m, d], plus probe IDs.
The model can be a full checkpoint or a base+PEFT adapter.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='full HF model path/id, or base model when --adapter is used')
    ap.add_argument('--adapter')
    ap.add_argument('--load-in-4bit', action='store_true', help='load target/base through bitsandbytes 4-bit quantization')
    ap.add_argument('--load-in-8bit', action='store_true', help='load target/base through bitsandbytes 8-bit quantization')
    ap.add_argument('--probes', required=True)
    ap.add_argument('--encoder', required=True, help='sentence-transformers model id/path')
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.7, help='stochastic audit temperature; use >0 when samples>1 so bank cross-fitting has independent replicates')
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--paired-seeds', action='store_true')
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    if args.samples < 1:
        raise SystemExit('--samples must be >= 1')
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit('choose at most one of --load-in-4bit/--load-in-8bit')
    if args.samples > 1 and args.temperature <= 0:
        raise SystemExit('samples>1 with deterministic decoding creates duplicate replicates and invalidates bank cross-fitting; use --temperature > 0 or --samples 1')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
    from peft import PeftModel

    rows=[json.loads(x) for x in Path(args.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else (BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None))
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=True)
    if args.adapter: model=PeftModel.from_pretrained(model,args.adapter)
    model.eval(); enc=SentenceTransformer(args.encoder)

    do_sample=args.temperature>0
    gen={'max_new_tokens':args.max_new_tokens,'do_sample':do_sample}
    if do_sample: gen.update({'temperature':args.temperature,'top_p':args.top_p})

    def text(q, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        x=tok(q,return_tensors='pt').to(model.device)
        with torch.inference_mode(): y=model.generate(**x,**gen)
        return tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip()

    base_all=[]; edit_all=[]; ids=[]; global_indices=[]
    for i,row in enumerate(rows):
        q=row.get('base_query') or row.get('q') or row.get('prompt')
        qe=row.get('edited_query') or row.get('q_edit')
        if qe is None: raise KeyError(f'probe {i} lacks edited_query')
        bt=[]; et=[]
        seed_index=int(row.get('global_probe_index', i))
        for rep in range(args.samples):
            # Preserve the same deterministic probe-level seed namespace even
            # when the online audit materializes only a selected subset.
            s=args.seed + seed_index*100003 + rep
            sb=s
            se=s if args.paired_seeds else s+500000003
            bt.append(text(q,sb)); et.append(text(qe,se))
        base_all.append(enc.encode(bt,normalize_embeddings=True,convert_to_numpy=True))
        edit_all.append(enc.encode(et,normalize_embeddings=True,convert_to_numpy=True))
        ids.append(str(row.get('probe_id',i)))
        global_indices.append(int(row.get('global_probe_index', i)))
        if (i+1)%25==0: print(f'{i+1}/{len(rows)} probes')
    np.savez_compressed(args.output,
        base_embeddings=np.asarray(base_all,dtype=np.float32),
        edited_embeddings=np.asarray(edit_all,dtype=np.float32),
        probe_ids=np.asarray(ids),
        global_probe_indices=np.asarray(global_indices,dtype=np.int64),
        model=np.asarray([args.model]),
        adapter=np.asarray([args.adapter or '']),
        encoder=np.asarray([args.encoder]))
    print(args.output)
if __name__=='__main__': main()
