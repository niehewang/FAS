#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

ANCESTORS=["Math","Code","Medical","Science"]
TRUTH_SUPPORT={
    "L1-linear-2p":{0,1},
    "L2-ties-3p":{0,1,3},
}
CONSTRUCTION={
    "L1-linear-2p":np.asarray([.5,.5,0,0],float),
    "L2-ties-3p":np.asarray([.34,.33,0,.33],float),
}

def delta(z):
    return np.asarray(z["edited_embeddings"],float)-np.asarray(z["base_embeddings"],float)

def baseout(z):
    return np.asarray(z["base_embeddings"],float)

def f1(pi,true,thr=.05):
    pred=set(np.where(np.asarray(pi)>=thr)[0].tolist())
    if not pred and not true:return 1.0
    p=len(pred&true)/len(pred) if pred else 0.0
    r=len(pred&true)/len(true) if true else 0.0
    return 2*p*r/(p+r) if p+r else 0.0

def solve(F,y):
    # raw-dictionary NNLS: coefficients remain in original functional scale
    A=np.concatenate([F[i].T for i in range(F.shape[0])],axis=0)
    yy=np.concatenate([y[i] for i in range(y.shape[0])],axis=0)
    c,_=nnls(A,yy)
    s=c.sum()
    pi=c/s if s>1e-12 else np.zeros_like(c)
    res=float(np.linalg.norm(A@c-yy)/(np.linalg.norm(yy)+1e-12))
    return c,pi,res

def geometry(F):
    X=np.stack([F[:,j,:].reshape(-1) for j in range(F.shape[1])],axis=1)
    X=X/(np.linalg.norm(X,axis=0,keepdims=True)+1e-12)
    sv=np.linalg.svd(X,compute_uv=False)
    gram=X.T@X
    coh=float(np.max(np.abs(gram-np.eye(gram.shape[0]))))
    return float(sv[-1]),coh,float(sv[0]/max(sv[-1],1e-12))

def restricted_res(F,y,support):
    idx=sorted(support)
    A=np.concatenate([F[i][:,idx].T for i in range(F.shape[0])],axis=0)
    yy=np.concatenate([y[i] for i in range(y.shape[0])],axis=0)
    c,_=nnls(A,yy)
    return float(np.linalg.norm(A@c-yy)/(np.linalg.norm(yy)+1e-12))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",required=True)
    ap.add_argument("--det-root",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    R=Path(args.run_root); D=Path(args.det_root)

    zbase=np.load(D/"base.npz",allow_pickle=False)
    zs={
      "Math":np.load(D/"math.npz",allow_pickle=False),
      "Code":np.load(D/"code.npz",allow_pickle=False),
      "Medical":np.load(D/"medical.npz",allow_pickle=False),
      "Science":np.load(D/"science.npz",allow_pickle=False),
    }
    ztargets={
      "L1-linear-2p":np.load(D/"l1.npz",allow_pickle=False),
      "L2-ties-3p":np.load(D/"l2.npz",allow_pickle=False),
    }

    gids=[int(x) for x in zbase["global_probe_indices"].tolist()]
    pos={g:i for i,g in enumerate(gids)}
    d0=delta(zbase)
    F=np.stack([delta(zs[a])-d0 for a in ANCESTORS],axis=1)
    absF=np.stack([baseout(zs[a])-baseout(zbase) for a in ANCESTORS],axis=1)

    rows=[]
    for target,zt in ztargets.items():
        yall=delta(zt)-d0
        yaall=baseout(zt)-baseout(zbase)
        truth=CONSTRUCTION[target]
        support=TRUTH_SUPPORT[target]

        # union-level construction-linearity diagnostic only
        pred=np.sum(F*truth[None,:,None],axis=1)
        lin_rel=float(np.linalg.norm(yall-pred)/(np.linalg.norm(yall)+1e-12))
        lin_cos=float(np.dot(yall.reshape(-1),pred.reshape(-1))/
                      ((np.linalg.norm(yall)+1e-12)*(np.linalg.norm(pred)+1e-12)))

        for B in [8,16,32,64,128]:
            for strategy in ["active","random"]:
                s=json.loads((R/"selections"/f"{strategy}_B{B}.json").read_text())
                sel=[int(x) for x in s["selected"]]
                loc=[pos[g] for g in sel]
                Fs=F[loc]; ys=yall[loc]
                _,pi,res=solve(Fs,ys)
                sig,coh,cond=geometry(Fs)
                rres=restricted_res(Fs,ys,support)

                _,api,ares=solve(absF[loc],yaall[loc])

                rows.append({
                  "target":target,"strategy":strategy,"budget":B,
                  "sigma_min_deterministic":sig,
                  "coherence_deterministic":coh,
                  "condition_number_deterministic":cond,
                  "fas_pi_raw_dictionary":pi.tolist(),
                  "fas_parent_f1":f1(pi,support),
                  "fas_coordinate_l1_vs_construction_diagnostic":float(np.abs(pi-truth).sum()),
                  "fas_relative_residual":res,
                  "truth_support_restricted_residual":rres,
                  "restricted_minus_full_residual":float(rres-res),
                  "absolute_pi":api.tolist(),
                  "absolute_parent_f1":f1(api,support),
                  "absolute_coordinate_l1_vs_construction_diagnostic":float(np.abs(api-truth).sum()),
                  "absolute_relative_residual":ares,
                  "construction_linearity_union_relative_mismatch":lin_rel,
                  "construction_linearity_union_cosine":lin_cos,
                })

    summary={}
    for target in ztargets:
        summary[target]={}
        for strategy in ["active","random"]:
            rr=[r for r in rows if r["target"]==target and r["strategy"]==strategy]
            summary[target][strategy]={
              "mean_parent_f1":float(np.mean([r["fas_parent_f1"] for r in rr])),
              "mean_coord_l1_construction_diagnostic":float(np.mean([r["fas_coordinate_l1_vs_construction_diagnostic"] for r in rr])),
              "mean_relative_residual":float(np.mean([r["fas_relative_residual"] for r in rr])),
              "mean_sigma_min_deterministic":float(np.mean([r["sigma_min_deterministic"] for r in rr])),
              "mean_absolute_parent_f1":float(np.mean([r["absolute_parent_f1"] for r in rr])),
              "mean_absolute_coord_l1_construction_diagnostic":float(np.mean([r["absolute_coordinate_l1_vs_construction_diagnostic"] for r in rr])),
              "mean_absolute_relative_residual":float(np.mean([r["absolute_relative_residual"] for r in rr])),
            }

    out={
      "stage":"seed11_deterministic_measurement_pilot",
      "paper_result_ready":False,
      "decode":"greedy_do_sample_false",
      "selection_source":"Existing Stage2-v4 stochastic selection bank; only measurement/audit responses are regenerated deterministically.",
      "interpretation_guardrail":"Construction weights are parameter-construction diagnostics, not asserted functional ground truth. Parent support remains construction-known.",
      "ancestor_names":ANCESTORS,
      "union_probe_count":len(gids),
      "summary":summary,
      "rows":rows,
    }
    Path(args.output).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({"union_probe_count":len(gids),"summary":summary},indent=2))

if __name__=="__main__":
    main()
