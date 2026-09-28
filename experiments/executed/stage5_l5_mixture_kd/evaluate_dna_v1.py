#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
N=['Math','Code','Medical','Science'];TRUE=set(range(4))
ap=argparse.ArgumentParser();ap.add_argument('--ancestors',nargs=4,required=True);ap.add_argument('--target',required=True);ap.add_argument('--output',required=True);ap.add_argument('--threshold',type=float,default=.01);a=ap.parse_args()
V=np.stack([np.load(p).reshape(-1) for p in a.ancestors]); y=np.load(a.target).reshape(-1); A=(V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)).T; yy=y/(np.linalg.norm(y)+1e-12);c,res=nnls(A,yy);pi=c/(c.sum()+1e-12);pred=set(np.where(pi>a.threshold)[0]);tp=len(pred&TRUE);pr=tp/len(pred) if pred else 0.;re=tp/4;f=2*pr*re/(pr+re) if pr+re else 0.;o={'names':N,'coordinates':pi.tolist(),'residual_norm':float(res),'support_threshold':a.threshold,'pred_support':[N[i] for i in sorted(pred)],'parent_f1':f,'exact_support':float(pred==TRUE)};Path(a.output).write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2))
