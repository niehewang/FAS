#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ANCESTORS=['Math','Code','Medical','Science']
EXP=np.array([0.4,0.3,0.2,0.1],float)
EPS=1e-12

def jdefault(o):
    if isinstance(o,np.integer): return int(o)
    if isinstance(o,np.floating): return float(o)
    if isinstance(o,np.bool_): return bool(o)
    if isinstance(o,np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)

def delta(z):
    b=np.asarray(z['base_embeddings'],float); e=np.asarray(z['edited_embeddings'],float)
    if b.ndim==3: b=b.mean(1); e=e.mean(1)
    return e-b

def baseout(z):
    b=np.asarray(z['base_embeddings'],float)
    if b.ndim==3: b=b.mean(1)
    return b

def gid_map(z):
    return {int(g):i for i,g in enumerate(np.asarray(z['global_probe_indices']).reshape(-1).tolist())}

def norms(F):
    return np.maximum(np.sqrt(np.sum(np.asarray(F,float)**2,axis=(0,2))),EPS)

def solve(Fsel,y,N):
    A=np.concatenate([(Fsel[i]/N[:,None]).T for i in range(Fsel.shape[0])],axis=0)
    yy=np.asarray(y,float).reshape(-1)
    c,_=nnls(A,yy); pred=A@c
    pi=c/(c.sum()+EPS)
    res=float(np.linalg.norm(pred-yy)/(np.linalg.norm(yy)+EPS))
    return pi,res

def support(pi,thr=0.01):
    det=np.where(np.asarray(pi)>thr)[0].astype(int).tolist(); tp=len(det)
    # All four Stage5 teachers were exposed; this is diagnostic only, not calibrated support inference.
    rec=tp/4.0; prec=1.0 if tp else 0.0
    f1=(2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    return {'threshold_diagnostic':float(thr),'detected':det,'parent_f1_if_scored':float(f1),'support_size':len(det)}

def cosflat(a,b):
    x=np.asarray(a,float).reshape(-1); y=np.asarray(b,float).reshape(-1)
    return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+EPS))

def relerr(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/(np.linalg.norm(np.asarray(b))+EPS))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage3',required=True); ap.add_argument('--stage5',required=True); ap.add_argument('--run-root',required=True)
    ap.add_argument('--seed',type=int,default=23); ap.add_argument('--output',required=True)
    a=ap.parse_args(); s=a.seed
    S3=Path(a.stage3); S5=Path(a.stage5); R=Path(a.run_root)
    zb17=np.load(S5/'shared/base17.npz'); zt=np.load(S5/f's{s}/student_response.npz')
    mb=gid_map(zb17); mt=gid_map(zt); db=delta(zb17); dt=delta(zt)
    zexp={n:np.load(R/f's{s}/experts/{n.lower()}/response.npz') for n in ANCESTORS}
    me={n:gid_map(zexp[n]) for n in ANCESTORS}; de={n:delta(zexp[n]) for n in ANCESTORS}
    # Full union available for the pilot; order by base17 gid.
    common=set(mb)&set(mt)
    for n in ANCESTORS: common &= set(me[n])
    common=sorted(common)
    if len(common)<32: raise ValueError(f'only {len(common)} common probes')
    # Build same-scale functional and absolute fields on common union.
    F17=np.stack([np.stack([de[n][me[n][g]]-db[mb[g]] for g in common]) for n in ANCESTORS],axis=1)
    AF17=np.stack([np.stack([baseout(zexp[n])[me[n][g]]-baseout(zb17)[mb[g]] for g in common]) for n in ANCESTORS],axis=1)
    N17=norms(F17); AN17=norms(AF17)
    pos_common={g:i for i,g in enumerate(common)}
    old=json.loads((S5/f's{s}/evaluation.json').read_text())
    old_by={r['strategy']:r for r in old['rows']}

    # Cross-scale field alignment context against original 4B Stage3 banks on the same probe gids.
    zb4=np.load(S3/'shared/base_full.npz'); m4=gid_map(zb4); d4=delta(zb4)
    align={}
    for n in ANCESTORS:
        z4=np.load(S3/f's{s}/banks/{n.lower()}.npz'); m4e=gid_map(z4); d4e=delta(z4)
        gids=[g for g in common if g in m4 and g in m4e]
        f4=np.stack([d4e[m4e[g]]-d4[m4[g]] for g in gids])
        f17=np.stack([de[n][me[n][g]]-db[mb[g]] for g in gids])
        # Norm-matched relative error prevents pure scale magnitude from dominating this diagnostic.
        sc=(np.linalg.norm(f17)+EPS)/(np.linalg.norm(f4)+EPS)
        align[n]={'n':len(gids),'field_cosine_flat':cosflat(f4,f17),
                  'raw_relative_error_4b_to_1p7b':relerr(f4,f17),
                  'norm_matched_relative_error':relerr(f4*sc,f17),
                  'norm_ratio_1p7b_over_4b':float(sc)}

    rows=[]
    for strat in ['active','random']:
        sel=json.loads((S3/f's{s}/selections/{strat}_B32.json').read_text())
        gids=[int(x) for x in sel['selected']]
        if not set(gids)<=set(common): raise ValueError(f'{strat}: selection not in common pilot union')
        p=[pos_common[g] for g in gids]; bp=[mb[g] for g in gids]; tp=[mt[g] for g in gids]
        y=np.stack([dt[i] for i in tp])-np.stack([db[i] for i in bp])
        pi,res=solve(F17[p],y,N17)
        ay=np.stack([baseout(zt)[i] for i in tp])-np.stack([baseout(zb17)[i] for i in bp])
        api,ares=solve(AF17[p],ay,AN17)
        oldres=float(old_by[strat]['fas_relative_residual'])
        rows.append({'strategy':strat,'budget':32,
                     'cross_scale_4b_dictionary_reference_residual':oldres,
                     'same_scale_1p7b_surrogate_fas':{'pi':pi.tolist(),'relative_residual':res,
                         'improvement_vs_cross_scale':oldres-res,'exposure_l1_diagnostic':float(np.abs(pi-EXP).sum()),**support(pi,0.01)},
                     'same_scale_1p7b_absolute_context':{'pi':api.tolist(),'relative_residual':ares,**support(api,0.01)}})
    oldmean=float(np.mean([r['cross_scale_4b_dictionary_reference_residual'] for r in rows]))
    newmean=float(np.mean([r['same_scale_1p7b_surrogate_fas']['relative_residual'] for r in rows]))
    impr=oldmean-newmean
    each=[r['same_scale_1p7b_surrogate_fas']['improvement_vs_cross_scale'] for r in rows]
    gate=bool(newmean<=0.80 and impr>=0.15 and min(each)>=0.10)
    out={'protocol':'FAS_STAGE5E_SCALEMATCHED17_PILOT_V1','seed':s,
         'scientific_role':'diagnostic_only_scale_matched_surrogate_dictionary_not_genealogical_ancestor_evidence',
         'rows':rows,'cross_scale_field_alignment':align,
         'pilot_gate':{'same_scale_mean_residual':newmean,'cross_scale_reference_mean_residual':oldmean,
                       'mean_absolute_improvement':impr,'per_strategy_improvement':each,
                       'frozen_rule':'expand_to_seeds_47_71 only if same_scale_mean_residual<=0.80 AND mean_improvement>=0.15 AND each_strategy_improvement>=0.10',
                       'pass':gate,
                       'recommended_next':'expand_scale_matched_diagnostic_to_seeds_47_71' if gate else 'do_not_expand; revise IFRF/cross-scale representation before L7'},
         'guardrails':['1.7B sibling experts are diagnostic surrogates, not actual 4B genealogical ancestors.',
                       'No Stage4 matched-G0 finite-sample calibration transfers to this pilot.',
                       'Teacher exposure weights are construction metadata, not functional-coordinate ground truth.',
                       'Original Stage5 results remain unchanged.']}
    Path(a.output).write_text(json.dumps(out,indent=2,default=jdefault),encoding='utf-8')
    print(json.dumps(out,indent=2,default=jdefault))
if __name__=='__main__': main()
