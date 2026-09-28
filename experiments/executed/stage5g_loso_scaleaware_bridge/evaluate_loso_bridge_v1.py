#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

SEEDS=[23,47,71]
DOMAINS=['Math','Code','Medical','Science']
EPS=1e-12
FACTORS=[1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0]

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

def gid_map(z):
    return {int(g):i for i,g in enumerate(np.asarray(z['global_probe_indices']).reshape(-1).tolist())}

def cosflat(a,b):
    x=np.asarray(a,float).reshape(-1); y=np.asarray(b,float).reshape(-1)
    return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+EPS))

def relerr(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/(np.linalg.norm(np.asarray(b))+EPS))

def global_norms(F):
    return np.maximum(np.sqrt(np.sum(np.asarray(F,float)**2,axis=(0,2))),EPS)

def solve_scaled(Fsel,y,N):
    A=np.concatenate([(Fsel[i]/N[:,None]).T for i in range(Fsel.shape[0])],axis=0)
    yy=np.asarray(y,float).reshape(-1)
    c,_=nnls(A,yy); pred=A@c
    pi=c/(c.sum()+EPS)
    res=float(np.linalg.norm(pred-yy)/(np.linalg.norm(yy)+EPS))
    return pi,res

def ridge_alpha_residual(X,Y,lam):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    K=X@X.T
    return np.linalg.solve(K+float(lam)*np.eye(K.shape[0]),Y-X)

def ridge_predict_residual(V,X,alpha):
    V=np.asarray(V,float); X=np.asarray(X,float)
    return V + (V@X.T)@alpha

def pair_file(stage5e,stage5f,seed,domain):
    root=stage5e if seed==23 else stage5f
    return root/f's{seed}'/'experts'/domain.lower()/'response.npz'

def load_seed_pairs(seed,S3,S5,S5E,S5F,exclude_gids):
    zb4=np.load(S3/'shared/base_full.npz'); d4=delta(zb4); mb4=gid_map(zb4)
    zb17=np.load(S5/'shared/base17.npz'); d17=delta(zb17); mb17=gid_map(zb17)
    by_domain={}
    Xall=[]; Yall=[]; Gall=[]; Dall=[]
    for d in DOMAINS:
        z4=np.load(S3/f's{seed}/banks/{d.lower()}.npz'); m4=gid_map(z4); f4=delta(z4)
        z17=np.load(pair_file(S5E,S5F,seed,d)); m17=gid_map(z17); f17=delta(z17)
        gids=sorted((set(m4)&set(m17)&set(mb4)&set(mb17))-set(exclude_gids))
        if not gids: raise ValueError(f'seed {seed} domain {d}: no bridge pairs after leakage exclusion')
        X=np.stack([f4[m4[g]]-d4[mb4[g]] for g in gids])
        Y=np.stack([f17[m17[g]]-d17[mb17[g]] for g in gids])
        by_domain[d]={'X':X,'Y':Y,'gids':gids}
        Xall.append(X);Yall.append(Y);Gall += gids;Dall += [d]*len(gids)
    return {'X':np.concatenate(Xall,axis=0),'Y':np.concatenate(Yall,axis=0),'gids':Gall,'domains':Dall,'by_domain':by_domain}

def choose_factor(train_seeds,pairs):
    rows=[]
    for fac in FACTORS:
        fold=[]
        for valid in train_seeds:
            tr=[s for s in train_seeds if s!=valid]
            Xtr=np.concatenate([pairs[s]['X'] for s in tr],axis=0)
            Ytr=np.concatenate([pairs[s]['Y'] for s in tr],axis=0)
            Xv=pairs[valid]['X']; Yv=pairs[valid]['Y']
            scale=max(float(np.mean(np.sum(Xtr*Xtr,axis=1))),1e-12)
            lam=scale*fac
            a=ridge_alpha_residual(Xtr,Ytr,lam)
            pred=ridge_predict_residual(Xv,Xtr,a)
            fold.append(relerr(pred,Yv))
        rows.append({'factor':fac,'mean_seed_cv_relative_error':float(np.mean(fold)),'fold_relative_errors':fold})
    rows.sort(key=lambda r:(r['mean_seed_cv_relative_error'],r['factor']))
    return float(rows[0]['factor']),rows

def eval_seed(s,S3,S5,S5E,S5F):
    train=[x for x in SEEDS if x!=s]
    sel_gids={}
    exclude=set()
    for strat in ['active','random']:
        p=json.loads((S3/f's{s}/selections/{strat}_B32.json').read_text())
        gs=[int(x) for x in p['selected']]; sel_gids[strat]=gs; exclude.update(gs)
    pairs={t:load_seed_pairs(t,S3,S5,S5E,S5F,exclude) for t in train}
    factor,cv=choose_factor(train,pairs)
    Xtr=np.concatenate([pairs[t]['X'] for t in train],axis=0)
    Ytr=np.concatenate([pairs[t]['Y'] for t in train],axis=0)
    scale=max(float(np.mean(np.sum(Xtr*Xtr,axis=1))),1e-12); lam=scale*factor
    alpha=ridge_alpha_residual(Xtr,Ytr,lam)

    # Training-only per-domain norm ratios for approximate global standardization.
    ratios={}
    for d in DOMAINS:
        xd=np.concatenate([pairs[t]['by_domain'][d]['X'] for t in train],axis=0)
        yd=np.concatenate([pairs[t]['by_domain'][d]['Y'] for t in train],axis=0)
        ratios[d]=float(np.sqrt(np.sum(yd*yd)+EPS)/np.sqrt(np.sum(xd*xd)+EPS))

    zb4=np.load(S3/'shared/base_full.npz'); d4=delta(zb4); mb4=gid_map(zb4)
    zb17=np.load(S5/'shared/base17.npz'); d17=delta(zb17); mb17=gid_map(zb17)
    zstu=np.load(S5/f's{s}/student_response.npz'); dstu=delta(zstu); mstu=gid_map(zstu)
    z4={d:np.load(S3/f's{s}/banks/{d.lower()}.npz') for d in DOMAINS}
    m4={d:gid_map(z4[d]) for d in DOMAINS}; de4={d:delta(z4[d]) for d in DOMAINS}
    F4full=np.stack([de4[d]-d4 for d in DOMAINS],axis=1)
    N4=global_norms(F4full)
    Nbridge=N4*np.asarray([ratios[d] for d in DOMAINS],float)

    # Held-out same-scale sibling fields are evaluation oracle only; never used above.
    z17e={d:np.load(pair_file(S5E,S5F,s,d)) for d in DOMAINS}
    m17e={d:gid_map(z17e[d]) for d in DOMAINS}; de17e={d:delta(z17e[d]) for d in DOMAINS}
    common=set(mb17)&set(mstu)
    for d in DOMAINS: common &= set(m17e[d])
    common=sorted(common)
    F17=np.stack([np.stack([de17e[d][m17e[d][g]]-d17[mb17[g]] for g in common]) for d in DOMAINS],axis=1)
    N17=global_norms(F17); pos17={g:i for i,g in enumerate(common)}

    old=json.loads((S5/f's{s}/evaluation.json').read_text()); old_by={r['strategy']:r for r in old['rows']}
    rows=[]; align=[]
    for strat in ['active','random']:
        gids=sel_gids[strat]
        if not set(gids)<=set(common): raise ValueError(f'seed{s} {strat}: selected gids not all in held-out same-scale oracle union')
        fpos=[mb4[g] for g in gids]; bpos=[mb17[g] for g in gids]; tpos=[mstu[g] for g in gids]
        raw=np.stack([[de4[d][m4[d][g]]-d4[mb4[g]] for d in DOMAINS] for g in gids],axis=0)
        bridge=ridge_predict_residual(raw.reshape(-1,raw.shape[-1]),Xtr,alpha).reshape(raw.shape)
        y=np.stack([dstu[i] for i in tpos])-np.stack([d17[i] for i in bpos])
        pi_b,res_b=solve_scaled(bridge,y,Nbridge)
        raw_res=float(old_by[strat]['fas_relative_residual'])
        # Same-scale oracle context on identical target/anchor.
        p=[pos17[g] for g in gids]
        pi_o,res_o=solve_scaled(F17[p],y,N17)
        # Held-out field alignment on these probes.
        oracle=F17[p]
        raw_cos=cosflat(raw,oracle); bridge_cos=cosflat(bridge,oracle)
        raw_er=relerr(raw,oracle); bridge_er=relerr(bridge,oracle)
        align.append({'strategy':strat,'raw_field_cosine':raw_cos,'bridge_field_cosine':bridge_cos,
                      'cosine_improvement':bridge_cos-raw_cos,'raw_field_relative_error':raw_er,
                      'bridge_field_relative_error':bridge_er,'relative_error_improvement':raw_er-bridge_er})
        rows.append({'strategy':strat,'budget':32,'raw_cross_scale_residual':raw_res,
                     'bridge_residual':res_b,'improvement_vs_raw':raw_res-res_b,
                     'bridge_pi_descriptive_only':pi_b.tolist(),
                     'same_scale_oracle_residual':res_o,'same_scale_oracle_pi_diagnostic':pi_o.tolist(),
                     'fraction_of_oracle_gap_recovered':float((raw_res-res_b)/(raw_res-res_o+EPS)),
                     'field_alignment':align[-1]})
    mean_raw=float(np.mean([r['raw_cross_scale_residual'] for r in rows])); mean_b=float(np.mean([r['bridge_residual'] for r in rows])); mean_o=float(np.mean([r['same_scale_oracle_residual'] for r in rows]))
    imp=mean_raw-mean_b; each=[r['improvement_vs_raw'] for r in rows]
    frac=float((mean_raw-mean_b)/(mean_raw-mean_o+EPS)); cosimp=float(np.mean([a['cosine_improvement'] for a in align]))
    seed_pass=bool(mean_b<=0.82 and imp>=0.10 and min(each)>=0.07)
    return {'seed':s,'held_out_seed':s,'bridge_train_seeds':train,'held_out_eval_probe_ids':sorted(exclude),
            'bridge_training_rows':int(len(Xtr)),'selected_ridge_factor':factor,'selected_lambda':lam,
            'inner_cross_seed_cv':cv,'training_only_domain_norm_ratios':ratios,'rows':rows,
            'summary':{'raw_mean_residual':mean_raw,'bridge_mean_residual':mean_b,'same_scale_oracle_mean_residual':mean_o,
                       'mean_improvement_vs_raw':imp,'fraction_of_oracle_gap_recovered':frac,
                       'mean_field_cosine_improvement':cosimp,'seed_gate_pass':seed_pass}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage3',required=True); ap.add_argument('--stage5',required=True)
    ap.add_argument('--stage5e',required=True); ap.add_argument('--stage5f',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); S3=Path(a.stage3); S5=Path(a.stage5); S5E=Path(a.stage5e); S5F=Path(a.stage5f)
    seeds=[eval_seed(s,S3,S5,S5E,S5F) for s in SEEDS]
    allrows=[r for z in seeds for r in z['rows']]
    pooled_raw=float(np.mean([r['raw_cross_scale_residual'] for r in allrows])); pooled_b=float(np.mean([r['bridge_residual'] for r in allrows])); pooled_o=float(np.mean([r['same_scale_oracle_residual'] for r in allrows]))
    pooled_frac=float((pooled_raw-pooled_b)/(pooled_raw-pooled_o+EPS))
    pooled_cos=float(np.mean([r['field_alignment']['cosine_improvement'] for r in allrows]))
    gate=bool(all(z['summary']['seed_gate_pass'] for z in seeds) and pooled_b<=0.80 and pooled_frac>=0.50 and pooled_cos>=0.10)
    out={'protocol':'FAS_STAGE5G_LOSO_SCALE_AWARE_BRIDGE_V1','scientific_role':'post_stage5_development_diagnostic_only',
         'bridge_family':'shared_identity_regularized_residual_ridge','seeds':seeds,
         'pooled':{'raw_mean_residual':pooled_raw,'bridge_mean_residual':pooled_b,'same_scale_oracle_mean_residual':pooled_o,
                   'mean_improvement_vs_raw':pooled_raw-pooled_b,'fraction_of_oracle_gap_recovered':pooled_frac,
                   'mean_field_cosine_improvement':pooled_cos},
         'gate':{'frozen_rule':'each seed bridge mean residual<=0.82, mean improvement>=0.10, each Active/Random improvement>=0.07; pooled bridge mean residual<=0.80; pooled oracle-gap recovery>=0.50; pooled field-cosine improvement>=0.10',
                 'pass':gate,
                 'recommended_next':'freeze_bridge_v1_then_new_unseen_seed_confirmatory_before_C3_or_L7' if gate else 'do_not_enter_L7_revise_cross_scale_representation'},
         'guardrails':['LOSO bridge is development evidence because it was designed after viewing Stage5.',
                       'Held-out 1.7B sibling fields are used only as evaluation oracle, never for bridge fitting or lambda selection.',
                       'No Stage4 G0 support threshold is applied to bridge coefficients.',
                       'Original Stage5 and Absolute/LLM-DNA baseline results remain unchanged and reportable.',
                       'A new unseen seed is mandatory before a strong C3 claim.']}
    Path(a.output).write_text(json.dumps(out,indent=2,default=jdefault),encoding='utf-8')
    print(json.dumps(out,indent=2,default=jdefault))
if __name__=='__main__': main()
