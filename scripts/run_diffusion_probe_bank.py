#!/usr/bin/env python3
"""Run text-to-image intervention probes and emit the common FAS NPZ schema.

Designed for optional cross-modality validation.  The downstream FAS pipeline is
unchanged: build_response_bank.py, select_probes.py and decomposition consume
this file exactly like text embeddings.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--lora');ap.add_argument('--probes',required=True);ap.add_argument('--encoder',default='openai/clip-vit-large-patch14');ap.add_argument('--samples',type=int,default=2);ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--height',type=int,default=768);ap.add_argument('--width',type=int,default=768);ap.add_argument('--steps',type=int,default=30);ap.add_argument('--guidance',type=float,default=5.0);ap.add_argument('--output',required=True);args=ap.parse_args()
    if args.samples<1: raise SystemExit('--samples must be >=1')
    import torch
    from diffusers import StableDiffusionXLPipeline
    from transformers import AutoImageProcessor,AutoModel
    rows=[json.loads(x) for x in Path(args.probes).read_text(encoding='utf-8').splitlines() if x.strip()]
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    pipe=StableDiffusionXLPipeline.from_pretrained(args.model,torch_dtype=dtype,use_safetensors=True)
    if args.lora: pipe.load_lora_weights(args.lora)
    device='cuda' if torch.cuda.is_available() else 'cpu';pipe=pipe.to(device);pipe.set_progress_bar_config(disable=True)
    proc=AutoImageProcessor.from_pretrained(args.encoder);enc=AutoModel.from_pretrained(args.encoder).to(device).eval()
    def embed(images):
        x=proc(images=images,return_tensors='pt').to(device)
        with torch.inference_mode():
            if hasattr(enc,'get_image_features'): z=enc.get_image_features(**x)
            else:
                o=enc(**x);z=getattr(o,'pooler_output',None)
                if z is None: z=o.last_hidden_state.mean(dim=1)
        z=torch.nn.functional.normalize(z.float(),dim=-1);return z.cpu().numpy()
    def gen(prompt,seed):
        g=torch.Generator(device=device).manual_seed(int(seed))
        return pipe(prompt,generator=g,height=args.height,width=args.width,num_inference_steps=args.steps,guidance_scale=args.guidance).images[0]
    base_all=[];edit_all=[];ids=[];global_indices=[]
    for i,r in enumerate(rows):
        q=r.get('base_query') or r.get('prompt');qe=r.get('edited_query')
        if not q or not qe: raise KeyError(f'probe {i} requires base_query/prompt and edited_query')
        gi=int(r.get('global_probe_index',i));bi=[];ei=[]
        for rep in range(args.samples):
            s=args.seed+gi*100003+rep
            bi.append(gen(q,s));ei.append(gen(qe,s))  # paired latent seed
        base_all.append(embed(bi));edit_all.append(embed(ei));ids.append(str(r.get('probe_id',i)));global_indices.append(gi)
        if (i+1)%10==0: print(f'{i+1}/{len(rows)} probes')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,base_embeddings=np.asarray(base_all,np.float32),edited_embeddings=np.asarray(edit_all,np.float32),probe_ids=np.asarray(ids),global_probe_indices=np.asarray(global_indices,np.int64),model=np.asarray([args.model]),adapter=np.asarray([args.lora or '']),encoder=np.asarray([args.encoder]))
    print(out)
if __name__=='__main__':main()
