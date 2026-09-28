#!/usr/bin/env python3
"""Extract raw FAS geometric coordinates for controlled analysis only.

Unlike decompose_target.py this script does not perform selective audit or make
any conformal claim.  It is used for controlled mechanism/counterfactual
analysis (e.g. domain-wise correlation with exact Shapley) where the descendant
is known by construction to be bank-complete and the quantity of interest is
the geometry itself.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from fas_core.decompose import fas_decompose

def main():
    ap=argparse.ArgumentParser();ap.add_argument('npz');ap.add_argument('--domain',default='');ap.add_argument('--output',required=True);args=ap.parse_args()
    z=np.load(args.npz,allow_pickle=False);A=np.asarray(z['A'],float);y=np.asarray(z['y'],float);w=np.asarray(z['weights'],float) if 'weights' in z else None
    names=[str(x) for x in z['ancestor_names'].tolist()] if 'ancestor_names' in z else [str(i) for i in range(A.shape[1])]
    d=fas_decompose(A,y,weights=w)
    obj={'analysis_type':'controlled_raw_coordinate','domain':args.domain,'parent_names':names,'coordinates':[float(x) for x in d['pi']],'rho':float(d['rho']),'relative_residual':float(d['relative_residual']),'no_conformal_claim':True}
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');print(p)
if __name__=='__main__':main()
