#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);ap.add_argument('--seeds',default='23,47,71');ap.add_argument('--output',required=True);a=ap.parse_args();seeds=[int(x) for x in a.seeds.split(',')];rows=[]
for s in seeds: rows += json.loads((Path(a.run_root)/f's{s}/evaluation.json').read_text())['rows']
def agg(depth,strategy,prefix='fas'):
 x=[r for r in rows if r['depth']==depth and r['strategy']==strategy];return {'n':len(x),'parent_f1':float(np.mean([r[prefix+'_parent_f1'] for r in x])),'true_parent_mass':float(np.mean([r[prefix+'_true_parent_mass'] for r in x])),'relative_residual':float(np.mean([r[prefix+'_relative_residual'] for r in x])),'top3_overlap_fraction':float(np.mean([r[prefix+'_top3_overlap_fraction'] for r in x])),'decomposable_rate_transfer_diagnostic':float(np.mean([r['state']=='Decomposable' for r in x]))}
summary={}
for d in ['D1_merge','D2_sft','D3_int4','D4_kd4b']:
 summary[d]={st:{'fas':agg(d,st,'fas'),'absolute':agg(d,st,'absolute')} for st in ['active','random']}
d1=summary['D1_merge']['active']['fas'];d4=summary['D4_kd4b']['active']['fas'];per=[]
for s in seeds:
 q=[r for r in rows if r['seed']==s and r['depth']=='D4_kd4b' and r['strategy']=='active'][0];per.append(q['fas_parent_f1'])
ret=d4['parent_f1']/(d1['parent_f1']+1e-12)
gate={'D4_active_mean_parent_f1':d4['parent_f1'],'D4_active_mean_true_parent_mass':d4['true_parent_mass'],'D4_active_mean_relative_residual':d4['relative_residual'],'D4_active_f1_retention_vs_D1':ret,'D4_active_seeds_parent_f1_ge_0p60':int(sum(x>=.60 for x in per))}
gate['pass']=bool(gate['D4_active_mean_parent_f1']>=.60 and gate['D4_active_mean_true_parent_mass']>=.65 and gate['D4_active_mean_relative_residual']<=.80 and gate['D4_active_f1_retention_vs_D1']>=.70 and gate['D4_active_seeds_parent_f1_ge_0p60']>=2)
out={'protocol':'FAS_STAGE8_DEEP_ANCESTRY_MATCHEDSCALE_V1','seeds':seeds,'summary':summary,'gate':gate,'claim_if_pass':'Deep functional ancestry remains meaningfully recoverable through a matched-scale Merge->SFT->INT4->KD chain; this is a same-family depth claim only.','claim_if_fail':'Deep ancestry degrades materially under the tested chain; report the trajectory as a boundary result and do not make a broad deep-lineage claim.','cross_scale_C3':'closed regardless of Stage8'};Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
