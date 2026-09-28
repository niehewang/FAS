#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);ap.add_argument('--seeds',default='23,47,71');ap.add_argument('--output',required=True);a=ap.parse_args();R=Path(a.run_root);seeds=[int(x) for x in a.seeds.split(',')]
rows=[];dna=[]
for s in seeds:
 p=R/f's{s}/evaluation.json'
 if p.exists(): rows+=json.loads(p.read_text())['rows']
 q=R/f's{s}/dna/decomposition.json'
 if q.exists(): dna.append({'seed':s,**json.loads(q.read_text())})
def mean(k,subset):
 v=[float(r[k]) for r in subset if r.get(k) is not None];return float(np.mean(v)) if v else None
summary={}
for st in ['active','random']:
 rr=[r for r in rows if r['strategy']==st];summary[st]={'n':len(rr),'decomposable_rate':float(np.mean([r['state']=='Decomposable' for r in rr])) if rr else None,'fas_parent_f1':mean('fas_parent_f1',rr),'absolute_parent_f1':mean('absolute_parent_f1',rr),'fas_residual':mean('fas_relative_residual',rr),'absolute_residual':mean('absolute_relative_residual',rr),'fas_exposure_l1_diagnostic':mean('fas_exposure_l1_diagnostic',rr),'absolute_exposure_l1_diagnostic':mean('absolute_exposure_l1_diagnostic',rr)}
out={'stage':'stage5_l5_mixture_kd_strong_baseline_v1','seeds':seeds,'summary':summary,'dna_decomp':{'n':len(dna),'mean_parent_f1':float(np.mean([x['parent_f1'] for x in dna])) if dna else None,'mean_exact_support':float(np.mean([x['exact_support'] for x in dna])) if dna else None,'rows':dna},'claim_gate_C3_preliminary':bool(summary.get('active',{}).get('fas_parent_f1') is not None and summary['active']['fas_parent_f1']>=0.6),'guardrail':'C3 final decision must consider FAS vs Absolute/DNA-Decomp and deep-chain follow-up; exposure weights are not functional-coordinate ground truth.'};Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
