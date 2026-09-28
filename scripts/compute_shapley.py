#!/usr/bin/env python3
"""Compute exact or permutation-approximate Shapley values from subset utilities.

Input CSV:
  subset,utility[,domain]
where subset is semicolon-separated parent names; the empty subset is an empty
string. Exact computation is used when all 2^n subsets are present. Otherwise,
`--permutations` estimates Shapley values from available prefix utilities and
fails clearly if a required prefix is missing.
"""
from __future__ import annotations
import argparse, csv, itertools, math, random
from pathlib import Path


def subset_key(s: str) -> frozenset[str]:
    return frozenset(x for x in (s or "").split(";") if x)


def exact_shapley(players, U):
    n=len(players); out={p:0.0 for p in players}
    fact=math.factorial
    for p in players:
        rest=[x for x in players if x!=p]
        for r in range(len(rest)+1):
            for comb in itertools.combinations(rest,r):
                S=frozenset(comb); Sp=frozenset(set(S)|{p})
                w=fact(len(S))*fact(n-len(S)-1)/fact(n)
                out[p]+=w*(U[Sp]-U[S])
    return out


def approx_shapley(players,U,n_perm,seed):
    rng=random.Random(seed); out={p:0.0 for p in players}; used=0
    for _ in range(n_perm):
        perm=players[:]; rng.shuffle(perm); S=frozenset(); ok=True; contrib=[]
        for p in perm:
            Sp=frozenset(set(S)|{p})
            if S not in U or Sp not in U: ok=False; break
            contrib.append((p,U[Sp]-U[S])); S=Sp
        if ok:
            used+=1
            for p,v in contrib: out[p]+=v
    if used==0: raise ValueError("No permutation had all required prefix utilities")
    return {p:v/used for p,v in out.items()}, used


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--output",default="results/derived/shapley.csv")
    ap.add_argument("--permutations",type=int,default=10000); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    rows=list(csv.DictReader(open(args.input,encoding="utf-8",newline="")))
    domains=sorted(set(r.get("domain","") for r in rows)); outputs=[]
    for domain in domains:
        rr=[r for r in rows if r.get("domain","")==domain]
        U={subset_key(r["subset"]):float(r["utility"]) for r in rr}
        players=sorted(set().union(*U.keys()))
        expected=2**len(players)
        if len(U)==expected and all(frozenset(c) in U for k in range(len(players)+1) for c in itertools.combinations(players,k)):
            vals=exact_shapley(players,U); mode="exact"; used=""
        else:
            vals,used=approx_shapley(players,U,args.permutations,args.seed); mode="permutation" 
        total=sum(vals.values()); norm={p:(v/total if abs(total)>1e-12 else 0.0) for p,v in vals.items()}
        for p in players: outputs.append({"domain":domain,"parent":p,"shapley":vals[p],"normalized_shapley":norm[p],"mode":mode,"permutations_used":used})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(outputs[0].keys()) if outputs else []
    with out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(outputs)
    print(f"Wrote {len(outputs)} rows -> {out}")
if __name__=="__main__": main()
