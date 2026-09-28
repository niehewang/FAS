#!/usr/bin/env python3
import argparse,random,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seed',type=int,default=20260924);a=ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed)
try:
 import torch; torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
 from llm_dna import DNAExtractionConfig, calc_dna
 cfg=DNAExtractionConfig(model_name=a.model,dataset='rand',gpu_id=0,max_samples=100,dna_dim=128,reduction_method='random_projection',trust_remote_code=True)
 r=calc_dna(cfg); v=np.asarray(r.vector,dtype=np.float32); Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.save(a.output,v); print(json.dumps({'model':a.model,'shape':list(v.shape),'output':a.output}))
except Exception as e:
 Path(a.output+'.error.txt').write_text(repr(e),encoding='utf-8'); raise
