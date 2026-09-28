#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
NAMES=['Math','Code','Medical','Science']; TRUE={0,1,2}
def d(z):return np.asarray(z['edited_embeddings'],float)-np.asarray(z['base_embeddings'],float)
def b(z):return np.asarray(z['base_embeddings'],float)
def mp(z):return {int(g):i for i,g in enumerate(z['global_probe_indices'].tolist())}
def conformal(cal,x):v=np.asarray(cal,float);return float((1+np.sum(v>=float(x)))/(len(v)+1))
def solve(F,y):
 A=F.transpose(1,0,2).reshape(F.shape[1],-1).T;yy=y.reshape(-1);c,_=nnls(A,yy);pi=c/(c.sum()+1e-12);res=float(np.linalg.norm(A@c-yy)/(np.linalg.norm(yy)+1e-12));return pi,res
def support(pi,t):
 P={i for i,x in enumerate(pi) if x>t};tp=len(P&TRUE);pr=tp/len(P) if P else 0.;re=tp/len(TRUE);f=2*pr*re/(pr+re) if pr+re else 0.;top=set(np.argsort(-np.asarray(pi))[:3].tolist());tt=len(top&TRUE);topf=tt/3.0
 return {'parent_f1':float(f),'detected':[NAMES[i] for i in sorted(P)],'true_parent_mass':float(sum(pi[i] for i in TRUE)),'false_parent_mass':float(sum(pi[i] for i in set(range(4))-TRUE)),'top3_overlap_fraction':float(topf)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--stage3',required=True);ap.add_argument('--stage4',required=True);ap.add_argument('--target',action='append',required=True,help='DEPTH=NPZ');ap.add_argument('--output',required=True);a=ap.parse_args();S3=Path(a.stage3);S4=Path(a.stage4)
 z0=np.load(S3/'shared/base_full.npz');M=mp(z0);D0=d(z0);B0=b(z0);banks=[np.load(S3/f's{a.seed}/banks/{n.lower()}.npz') for n in NAMES]
 FF=np.stack([d(z)-D0 for z in banks],axis=1);AF=np.stack([b(z)-B0 for z in banks],axis=1)
 fn=np.sqrt(np.sum(FF**2,axis=(0,2)));an=np.sqrt(np.sum(AF**2,axis=(0,2)));FF=FF/np.maximum(fn[None,:,None],1e-12);AF=AF/np.maximum(an[None,:,None],1e-12)
 thr=float(json.loads((S4/f's{a.seed}/final/support_threshold.json').read_text())['selected']['threshold']);sig=[float(r['score']) for r in csv.DictReader(open(S4/f's{a.seed}/signal_scores.csv'))];op=[float(r['score']) for r in csv.DictReader(open(S4/f's{a.seed}/final/open_scores.csv'))]
 targets=[]
 for x in a.target:
  dep,p=x.split('=',1);z=np.load(p);targets.append((dep,z,mp(z)))
 rows=[]
 for strat in ['active','random']:
  ss=json.loads((S3/f's{a.seed}/selections/{strat}_B32.json').read_text());g=[int(x) for x in ss['selected']];pos=[M[x] for x in g];F=FF[pos];A=AF[pos]
  for dep,z,mm in targets:
   tp=[mm[x] for x in g];y=d(z)[tp]-D0[pos];ay=b(z)[tp]-B0[pos];pi,res=solve(F,y);api,ares=solve(A,ay);sm=support(pi,thr);am=support(api,thr);score=float(np.linalg.norm(y.reshape(-1)));ps=conformal(sig,score);po=conformal(op,res) if ps<.05 else None;state='Low-Signal' if ps>=.05 else ('Bank-Insufficient' if po is not None and po<.05 else 'Decomposable')
   rows.append({'seed':a.seed,'depth':dep,'strategy':strat,'threshold':thr,'calibration_regime':'g1_transfer_diagnostic_no_finite_sample_claim','state':state,'signal_p':ps,'open_p':po,'fas_pi':pi.tolist(),'fas_relative_residual':res,**{'fas_'+k:v for k,v in sm.items()},'absolute_pi':api.tolist(),'absolute_relative_residual':ares,**{'absolute_'+k:v for k,v in am.items()}})
 out={'seed':a.seed,'protocol':'FAS_STAGE8_DEEP_ANCESTRY_MATCHEDSCALE_V1','true_support':['Math','Code','Medical'],'rows':rows,'guardrail':'Same-scale deep-chain stress only. Stage4 p-values are transfer diagnostics, not finite-sample guarantees.'};Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
