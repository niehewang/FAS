#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

ANCESTORS = ['Math','Code','Medical','Science']
EXP = np.array([0.4,0.3,0.2,0.1], dtype=float)
EPS = 1e-12


def jdefault(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o).__name__)


def delta(z):
    b=np.asarray(z['base_embeddings'],float); e=np.asarray(z['edited_embeddings'],float)
    if b.ndim==3:
        b=b.mean(axis=1); e=e.mean(axis=1)
    return e-b


def baseout(z):
    b=np.asarray(z['base_embeddings'],float)
    if b.ndim==3: b=b.mean(axis=1)
    return b


def gid_map(z):
    if 'global_probe_indices' not in z:
        raise KeyError('response NPZ missing global_probe_indices')
    return {int(g):i for i,g in enumerate(np.asarray(z['global_probe_indices']).reshape(-1).tolist())}


def norms(fields):
    # fields [N,K,d], identical to v0.11 global_ancestor_norms
    return np.maximum(np.sqrt(np.sum(np.asarray(fields,float)**2, axis=(0,2))), EPS)


def solve(fields_sel, y, global_norms):
    # fields_sel [B,K,d] -> A [B*d,K], with v0.11 global ancestor normalization
    A=np.concatenate([(fields_sel[i]/global_norms[:,None]).T for i in range(fields_sel.shape[0])], axis=0)
    yy=np.asarray(y,float).reshape(-1)
    c,_=nnls(A,yy)
    pi=c/(c.sum()+EPS)
    pred=A@c
    res=float(np.linalg.norm(pred-yy)/(np.linalg.norm(yy)+EPS))
    return pi,res


def solve_legacy(fields_sel, y):
    A=np.concatenate([fields_sel[i].T for i in range(fields_sel.shape[0])], axis=0)
    yy=np.asarray(y,float).reshape(-1)
    c,_=nnls(A,yy)
    pi=c/(c.sum()+EPS)
    pred=A@c
    res=float(np.linalg.norm(pred-yy)/(np.linalg.norm(yy)+EPS))
    return pi,res


def support_metrics(pi, threshold):
    pred=set(int(x) for x in np.where(np.asarray(pi)>float(threshold))[0].tolist())
    truth=set(range(4)); tp=len(pred&truth)
    pr=tp/len(pred) if pred else 0.0; re=tp/4.0
    f1=2*pr*re/(pr+re) if pr+re else 0.0
    return {'threshold':float(threshold),'detected':sorted(pred),'parent_f1_if_scored':float(f1),'support_size':len(pred)}


def unit_rows(x):
    x=np.asarray(x,float)
    return x/(np.linalg.norm(x,axis=-1,keepdims=True)+EPS)


def cosine_rows(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+EPS
    return np.sum(a*b,axis=1)/den


def relerr(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/(np.linalg.norm(np.asarray(b))+EPS))


def fit_diag(X,Y):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    power=np.mean(np.sum(X*X,axis=1))/max(X.shape[1],1)
    lam=max(power*1e-3,1e-10)
    beta=np.sum(X*Y,axis=0)/(np.sum(X*X,axis=0)+lam)
    beta=np.clip(beta,-5.0,5.0)
    return beta


def ridge_alpha(X,Y,lam):
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    K=X@X.T
    return np.linalg.solve(K+float(lam)*np.eye(K.shape[0]),Y)


def ridge_predict(V,X,alpha):
    return (np.asarray(V,float)@np.asarray(X,float).T)@np.asarray(alpha,float)


def choose_ridge_lambda(X,Y):
    X=np.asarray(X,float);Y=np.asarray(Y,float); n=X.shape[0]
    scale=max(float(np.mean(np.sum(X*X,axis=1))),1e-10)
    factors=[1e-4,1e-3,1e-2,1e-1,1.0,10.0]
    if n < 6:
        return scale*1e-2, {'cv_available':False,'reason':'fewer_than_6_anchor_pairs'}
    folds=np.arange(n)%3
    scores=[]
    for fac in factors:
        lam=scale*fac; vals=[]
        for f in range(3):
            tr=folds!=f; va=folds==f
            if tr.sum()<2 or va.sum()==0: continue
            al=ridge_alpha(X[tr],Y[tr],lam)
            pred=ridge_predict(X[va],X[tr],al)
            vals.append(relerr(pred,Y[va]))
        scores.append((float(np.mean(vals)) if vals else float('inf'),lam,fac))
    scores.sort(key=lambda x:x[0])
    best=scores[0]
    return best[1], {'cv_available':True,'selected_factor':best[2],'selected_lambda':best[1],
                     'grid':[{'factor':fac,'lambda':lam,'cv_relerr':sc} for sc,lam,fac in scores]}


def anchor_stats(X4,X17,pred_diag=None,pred_ridge=None):
    out={
      'n':int(len(X4)),
      'raw_mean_cosine':float(np.mean(cosine_rows(X4,X17))) if len(X4) else None,
      'raw_median_cosine':float(np.median(cosine_rows(X4,X17))) if len(X4) else None,
      'raw_relative_error':relerr(X4,X17) if len(X4) else None,
      'median_norm_ratio_1p7b_over_4b':float(np.median((np.linalg.norm(X17,axis=1)+EPS)/(np.linalg.norm(X4,axis=1)+EPS))) if len(X4) else None,
    }
    if pred_diag is not None:
      out.update({'diag_mean_cosine':float(np.mean(cosine_rows(pred_diag,X17))), 'diag_relative_error':relerr(pred_diag,X17)})
    if pred_ridge is not None:
      out.update({'ridge_mean_cosine':float(np.mean(cosine_rows(pred_ridge,X17))), 'ridge_relative_error':relerr(pred_ridge,X17)})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--stage3',required=True)
    ap.add_argument('--stage4',required=True)
    ap.add_argument('--stage5',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); s=a.seed
    S3=Path(a.stage3);S4=Path(a.stage4);S5=Path(a.stage5)

    zb4=np.load(S3/'shared/base_full.npz'); m4=gid_map(zb4); d4=delta(zb4)
    z17=np.load(S5/'shared/base17.npz'); m17=gid_map(z17); d17=delta(z17)
    zt=np.load(S5/f's{s}/student_response.npz'); mt=gid_map(zt); dt=delta(zt)
    banks=[np.load(S3/f's{s}/banks/{n.lower()}.npz') for n in ANCESTORS]
    # Stage3 banks are full-pool and aligned to shared base_full.
    Fraw=np.stack([delta(z)-d4 for z in banks],axis=1)  # [N,K,d]
    Nraw=norms(Fraw)

    # Absolute baseline field, retained for context only.
    AFraw=np.stack([baseout(z)-baseout(zb4) for z in banks],axis=1)
    ANraw=norms(AFraw)

    common=sorted(set(m4)&set(m17))
    common_set=set(common)
    cal=json.loads((S4/f's{s}/final/support_threshold.json').read_text()) if (S4/f's{s}/final/support_threshold.json').exists() else json.loads((S4/f's{s}/support_threshold.json').read_text())
    thr=float(cal['selected']['threshold'])
    old_eval=json.loads((S5/f's{s}/evaluation.json').read_text())
    old_by={r['strategy']:r for r in old_eval['rows']}

    rows=[]
    for strat in ['active','random']:
        sel=json.loads((S3/f's{s}/selections/{strat}_B32.json').read_text())
        gids=[int(x) for x in sel['selected']]
        if not set(gids)<=common_set:
            raise ValueError(f'{strat}: selected probes not all in paired 4B/1.7B anchors')
        fpos=[m4[g] for g in gids]; bpos=[m17[g] for g in gids]; tpos=[mt[g] for g in gids]
        fit_gids=[g for g in common if g not in set(gids)]
        if len(fit_gids)<8:
            raise ValueError(f'{strat}: only {len(fit_gids)} non-evaluation anchor probes; refuse target-overlapping transfer fit')
        xfit=np.stack([d4[m4[g]] for g in fit_gids]); yfit=np.stack([d17[m17[g]] for g in fit_gids])
        xev=np.stack([d4[m4[g]] for g in gids]); yev=np.stack([d17[m17[g]] for g in gids])

        beta=fit_diag(xfit,yfit)
        pred_diag=xev*beta[None,:]
        lam,cv=choose_ridge_lambda(xfit,yfit)
        alpha=ridge_alpha(xfit,yfit,lam)
        pred_ridge=ridge_predict(xev,xfit,alpha)

        # Current paired-anchor target.
        ypaired=dt[tpos]-d17[bpos]
        # Controls: no paired anchor, wrong-scale 4B anchor, and shuffled same-scale 1.7B anchor.
        yno=dt[tpos]
        ywrong4=dt[tpos]-d4[fpos]
        sh=np.roll(np.arange(len(bpos)),1)
        yshuf=dt[tpos]-d17[np.asarray(bpos)[sh]]

        Fsel=Fraw[fpos]
        pi_legacy,res_legacy=solve_legacy(Fsel,ypaired)
        pi_raw,res_raw=solve(Fsel,ypaired,Nraw)
        pi_no,res_no=solve(Fsel,yno,Nraw)
        pi_w4,res_w4=solve(Fsel,ywrong4,Nraw)
        pi_sh,res_sh=solve(Fsel,yshuf,Nraw)

        # Protocol-consistent direction-only diagnostic.
        base4u=unit_rows(d4); targ4u=np.stack([unit_rows(delta(z)) for z in banks],axis=1)
        Funit=targ4u-base4u[:,None,:]
        Nunit=norms(Funit)
        yunit=unit_rows(dt[tpos])-unit_rows(d17[bpos])
        pi_unit,res_unit=solve(Funit[fpos],yunit,Nunit)

        # Anchor-only diagonal transfer; target never used in fit.
        Fdiag=Fraw*beta[None,None,:]
        Ndiag=norms(Fdiag)
        pi_diag,res_diag=solve(Fdiag[fpos],ypaired,Ndiag)

        # Anchor-only linear ridge transfer in semantic feature space; no target used in fit/CV.
        flat=Fraw.reshape(-1,Fraw.shape[-1])
        Fridge=ridge_predict(flat,xfit,alpha).reshape(Fraw.shape)
        Nridge=norms(Fridge)
        pi_ridge,res_ridge=solve(Fridge[fpos],ypaired,Nridge)

        # Protocol-normalized absolute baseline context.
        ay=baseout(zt)[tpos]-baseout(z17)[bpos]
        api,ares=solve(AFraw[fpos],ay,ANraw)

        anchor=anchor_stats(xev,yev,pred_diag=pred_diag,pred_ridge=pred_ridge)
        row={
          'seed':s,'strategy':strat,'budget':32,'support_threshold_stage4':thr,
          'anchor_fit_n':len(fit_gids),'anchor_eval_n':len(gids),'ridge_cv':cv,'anchor_eval':anchor,
          'legacy_stage5':{
             'reported_state':old_by[strat]['state'],
             'reported_pi':old_by[strat]['fas_pi'],
             'reported_residual':old_by[strat]['fas_relative_residual'],
             'recomputed_legacy_pi':pi_legacy.tolist(),'recomputed_legacy_residual':res_legacy
          },
          'protocol_normed_paired':{'pi':pi_raw.tolist(),'relative_residual':res_raw,**support_metrics(pi_raw,thr),'exposure_l1_diagnostic':float(np.abs(pi_raw-EXP).sum())},
          'anchor_controls':{
             'no_anchor':{'pi':pi_no.tolist(),'relative_residual':res_no},
             'wrong_4b_anchor':{'pi':pi_w4.tolist(),'relative_residual':res_w4},
             'shuffled_1p7b_anchor':{'pi':pi_sh.tolist(),'relative_residual':res_sh}
          },
          'unit_direction_diagnostic':{'pi':pi_unit.tolist(),'relative_residual':res_unit,**support_metrics(pi_unit,0.01)},
          'anchor_diagonal_transfer':{'pi':pi_diag.tolist(),'relative_residual':res_diag,**support_metrics(pi_diag,0.01)},
          'anchor_ridge_transfer':{'pi':pi_ridge.tolist(),'relative_residual':res_ridge,**support_metrics(pi_ridge,0.01)},
          'absolute_protocol_normed_context':{'pi':api.tolist(),'relative_residual':ares,**support_metrics(api,thr)},
          'guardrail':'Transfer variants are post-Stage5 diagnostics. Only paired base anchors outside the evaluated probe set are used to fit/cross-validate transfer; KD target responses and exposure weights are never used for fitting.'
        }
        rows.append(row)

    out={'protocol':'FAS_STAGE5D_CROSSSCALE_ANCHOR_DIAGNOSTIC_V1','seed':s,'target':'L5-mixture-kd-v2-1p7b','rows':rows,
         'implementation_note':'Stage5 legacy evaluator omitted v0.11 global ancestor-norm normalization. This can change pi/support but cannot change the NNLS cone residual or rescue the observed Bank-Insufficient residual failure.',
         'calibration_note':'No transformed diagnostic inherits Stage4 matched-G0 calibration. Residual/support numbers for transfer variants are diagnostic only.'}
    Path(a.output).write_text(json.dumps(out,indent=2,default=jdefault),encoding='utf-8')
    print(json.dumps(out,indent=2,default=jdefault))

if __name__=='__main__': main()
