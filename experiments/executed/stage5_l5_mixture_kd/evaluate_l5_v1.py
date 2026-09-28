#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ANCESTORS=['Math','Code','Medical','Science']; TRUE=set(range(4)); EXP=np.array([.4,.3,.2,.1],float)
def delta(z):return np.asarray(z['edited_embeddings'],float)-np.asarray(z['base_embeddings'],float)
def baseout(z):return np.asarray(z['base_embeddings'],float)
def mp(z):return {int(g):i for i,g in enumerate(z['global_probe_indices'].tolist())}
def conformal(cal,x):a=np.asarray(cal,float);return float((1+np.sum(a>=float(x)))/(len(a)+1))
def solve(F,y):
 A=np.concatenate([F[i].T for i in range(F.shape[0])]); yy=y.reshape(-1); c,_=nnls(A,yy); pi=c/(c.sum()+1e-12); res=float(np.linalg.norm(A@c-yy)/(np.linalg.norm(yy)+1e-12)); return pi,res
def sm(pi,thr):
 pred=set(int(x) for x in np.where(np.asarray(pi)>thr)[0].tolist()); tp=len(pred&TRUE); pr=tp/len(pred) if pred else 0.; re=tp/4.; f=2*pr*re/(pr+re) if pr+re else 0.;return {'parent_f1':float(f),'exact_support':float(pred==TRUE),'detected':sorted(pred)}
def json_default(o):
 if isinstance(o,np.integer): return int(o)
 if isinstance(o,np.floating): return float(o)
 if isinstance(o,np.bool_): return bool(o)
 if isinstance(o,np.ndarray): return o.tolist()
 raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--stage3',required=True);ap.add_argument('--stage4',required=True);ap.add_argument('--base17',required=True);ap.add_argument('--student',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 S3=Path(a.stage3); S4=Path(a.stage4); zb4=np.load(S3/'shared/base_full.npz'); m4=mp(zb4); d04=delta(zb4)
 banks=[np.load(S3/f's{a.seed}/banks/{n.lower()}.npz') for n in ANCESTORS]
 Ffull=np.stack([delta(z)-d04 for z in banks],axis=1); AFfull=np.stack([baseout(z)-baseout(zb4) for z in banks],axis=1)
 z17=np.load(a.base17); ms=mp(z17); zt=np.load(a.student); mt=mp(zt)
 cal=json.loads((S4/f's{a.seed}/final/support_threshold.json').read_text()); thr=float(cal['selected']['threshold'])
 sig=[float(r['score']) for r in csv.DictReader(open(S4/f's{a.seed}/signal_scores.csv'))]; op=[float(r['score']) for r in csv.DictReader(open(S4/f's{a.seed}/final/open_scores.csv'))]
 rows=[]
 for strat in ['active','random']:
  s=json.loads((S3/f's{a.seed}/selections/{strat}_B32.json').read_text()); gids=[int(x) for x in s['selected']]
  fpos=[m4[g] for g in gids]; bpos=[ms[g] for g in gids]; tpos=[mt[g] for g in gids]
  y=(delta(zt)[tpos]-delta(z17)[bpos]); ay=(baseout(zt)[tpos]-baseout(z17)[bpos]); F=Ffull[fpos]; AF=AFfull[fpos]
  pi,res=solve(F,y); api,ares=solve(AF,ay); ss=float(np.linalg.norm(y.reshape(-1))); ps=conformal(sig,ss); po=conformal(op,res) if ps<.05 else None; state='Low-Signal' if ps>=.05 else ('Bank-Insufficient' if po<.05 else 'Decomposable')
  fm=sm(pi,thr) if state=='Decomposable' else {'parent_f1':0.,'exact_support':0.,'detected':[]}; am=sm(api,thr)
  rows.append({'seed':a.seed,'strategy':strat,'budget':32,'support_threshold':thr,'calibration_regime':'g1_transfer_diagnostic_no_finite_sample_claim','state':state,'signal_score':ss,'signal_p':ps,'open_p':po,'fas_pi':pi.tolist(),'fas_relative_residual':res,'fas_parent_f1':fm['parent_f1'],'fas_exact_support':fm['exact_support'],'fas_detected':fm['detected'],'fas_exposure_l1_diagnostic':float(np.abs(pi-EXP).sum()),'absolute_pi':api.tolist(),'absolute_relative_residual':ares,'absolute_parent_f1':am['parent_f1'],'absolute_exact_support':am['exact_support'],'absolute_detected':am['detected'],'absolute_exposure_l1_diagnostic':float(np.abs(api-EXP).sum())})
 out={'seed':a.seed,'target':'L5-mixture-kd-v2-1p7b','true_support':ANCESTORS,'exposure_diagnostic':EXP.tolist(),'guardrail':'Exposure weights are diagnostic metadata, not realized functional-coordinate ground truth. G0 conformal values are transfer diagnostics only.','rows':rows};Path(a.output).write_text(json.dumps(out,indent=2,default=json_default),encoding='utf-8');print(json.dumps(out,indent=2,default=json_default))
if __name__=='__main__':main()
