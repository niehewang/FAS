#!/usr/bin/env python3
"""Prepare A and y for target decomposition from embeddings and selected probes."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from fas_core.response import global_ancestor_norms, stack_selected


def response(path):
    z=np.load(path)
    b=np.asarray(z["base_embeddings"],dtype=float)
    e=np.asarray(z["edited_embeddings"],dtype=float)
    return (e-b) if b.ndim==2 else (e.mean(axis=1)-b.mean(axis=1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bank",required=True,help="response_bank.npz")
    ap.add_argument("--base",required=True,help="base model embedding NPZ")
    ap.add_argument("--target",required=True,help="target model embedding NPZ")
    ap.add_argument("--selected",required=True,help="selected_probes.json")
    ap.add_argument("--target-selected-only", action="store_true", help="base/target NPZ contain exactly B selected probes in selected-order; avoids querying the full candidate pool online")
    ap.add_argument("--weights", help="optional .npy or .npz whitening vector/matrix aligned with stacked selected coordinates")
    ap.add_argument("--weights-key", default="weights")
    ap.add_argument("--output",default="target_decomposition_input.npz")
    args=ap.parse_args()
    bank=np.load(args.bank)
    fields=bank["task_fields"]
    selected=json.loads(Path(args.selected).read_text())["selected"]
    norms=global_ancestor_norms(fields)
    A=stack_selected(fields,selected,norms)
    target_z=np.load(args.target)
    base_z=np.load(args.base)
    yall=response(args.target)-response(args.base)
    if args.target_selected_only:
        if yall.shape[0] != len(selected):
            raise ValueError(f"selected-only target/base have {yall.shape[0]} probes but selection has {len(selected)}")
        # Guard against silent selected-probe reordering. Modern response NPZs
        # persist the global indices written by subset_probe_pool.py.
        for label,zobj in (("target",target_z),("base",base_z)):
            if "global_probe_indices" in zobj:
                got=[int(x) for x in np.asarray(zobj["global_probe_indices"]).reshape(-1).tolist()]
                if got != [int(x) for x in selected]:
                    raise ValueError(f"{label} selected-only global probe indices do not match selected.json: got={got[:8]} expected={selected[:8]}")
        if "probe_ids" in target_z and "probe_ids" in base_z:
            t_ids=[str(x) for x in target_z["probe_ids"].tolist()]
            b_ids=[str(x) for x in base_z["probe_ids"].tolist()]
            if t_ids != b_ids:
                raise ValueError("selected-only target/base probe_ids differ")
        y=np.concatenate([yall[i] for i in range(len(selected))],axis=0)
    else:
        if max(selected, default=-1) >= yall.shape[0]:
            raise ValueError("selected global index exceeds target/base probe count; use --target-selected-only for B-probe online files")
        y=np.concatenate([yall[i] for i in selected],axis=0)
    names = bank["ancestor_names"] if "ancestor_names" in bank else np.asarray([f"ancestor_{i}" for i in range(A.shape[1])])
    payload=dict(A=A,y=y,selected=np.asarray(selected),norms=norms,ancestor_names=names)
    if args.weights:
        wp=Path(args.weights)
        if wp.suffix.lower()==".npy":
            W=np.load(wp)
        else:
            wz=np.load(wp)
            if args.weights_key not in wz:
                raise KeyError(f"{wp}: missing key {args.weights_key!r}")
            W=wz[args.weights_key]
        W=np.asarray(W,dtype=float)
        if W.ndim==1 and W.shape[0]!=y.shape[0]:
            raise ValueError(f"1D weights length {W.shape[0]} != y dimension {y.shape[0]}")
        if W.ndim==2 and W.shape!=(y.shape[0],y.shape[0]):
            raise ValueError(f"2D whitening shape {W.shape} incompatible with y dimension {y.shape[0]}")
        payload["weights"]=W
    np.savez_compressed(args.output,**payload)
    print(f"saved {args.output}: A={A.shape}, y={y.shape}")

if __name__=="__main__": main()
