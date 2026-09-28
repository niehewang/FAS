#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import numpy as np

def read_rows(path):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def concat_parts(parts, output):
    objs=[np.load(p,allow_pickle=False) for p in parts]
    kw={}
    for key in ["base_embeddings","edited_embeddings","probe_ids","global_probe_indices"]:
        kw[key]=np.concatenate([z[key] for z in objs],axis=0)
    for key in ["model","adapter","encoder","runner","decode"]:
        kw[key]=objs[0][key]
    np.savez_compressed(output,**kw)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--probes",required=True)
    ap.add_argument("--encoder",default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--max-new-tokens",type=int,default=96)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--chunk-size",type=int,default=100)
    ap.add_argument("--work-dir",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    rows=read_rows(args.probes)
    if not rows:
        raise SystemExit("empty probe pool")
    work=Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from peft import PeftModel

    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    tok.padding_side="left"

    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map="auto",
        trust_remote_code=True,low_cpu_mem_usage=True
    )
    if args.adapter:
        model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)
    model.eval()
    enc=SentenceTransformer(args.encoder,device="cuda" if torch.cuda.is_available() else "cpu")

    def generate_batch(texts):
        x=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=4096).to(model.device)
        inlen=x["input_ids"].shape[1]
        with torch.inference_mode():
            y=model.generate(
                **x,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return tok.batch_decode(y[:,inlen:],skip_special_tokens=True)

    parts=[]
    n_chunks=(len(rows)+args.chunk_size-1)//args.chunk_size
    for ci,start in enumerate(range(0,len(rows),args.chunk_size)):
        sub=rows[start:start+args.chunk_size]
        part=work/f"part_{ci:04d}.npz"
        parts.append(part)
        if part.exists():
            try:
                z=np.load(part,allow_pickle=False)
                if len(z["probe_ids"])==len(sub):
                    print(f"resume chunk {ci+1}/{n_chunks}: {part}")
                    continue
            except Exception:
                part.unlink(missing_ok=True)

        base_store=[]; edit_store=[]
        for bs in range(0,len(sub),args.batch_size):
            batch=sub[bs:bs+args.batch_size]
            base_text=[r.get("base_query") or r.get("q") or r.get("prompt") for r in batch]
            edit_text=[r.get("edited_query") or r.get("q_edit") for r in batch]
            if any(x is None for x in edit_text):
                raise KeyError("probe lacks edited_query")
            bo=generate_batch(base_text)
            eo=generate_batch(edit_text)
            be=enc.encode(bo,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            ee=enc.encode(eo,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            base_store.extend(be); edit_store.extend(ee)
            print(f"chunk {ci+1}/{n_chunks} batch {bs//args.batch_size+1}")

        np.savez_compressed(
            part,
            base_embeddings=np.asarray(base_store,dtype=np.float32),
            edited_embeddings=np.asarray(edit_store,dtype=np.float32),
            probe_ids=np.asarray([str(r.get("probe_id",start+j)) for j,r in enumerate(sub)]),
            global_probe_indices=np.asarray(
                [int(r.get("global_probe_index",start+j)) for j,r in enumerate(sub)],dtype=np.int64
            ),
            model=np.asarray([args.model]),adapter=np.asarray([args.adapter or ""]),
            encoder=np.asarray([args.encoder]),
            runner=np.asarray(["stage2_deterministic_pilot_v1"]),
            decode=np.asarray(["greedy_do_sample_false"])
        )
        print("saved",part)

    concat_parts(parts,out)
    print(f"DONE {out} n={len(rows)} decode=greedy")
    del model,enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__=="__main__":
    main()
