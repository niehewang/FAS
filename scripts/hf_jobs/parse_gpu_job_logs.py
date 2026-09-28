#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path
import numpy as np
MARK='FAS_RESULT_JSON='

def load_one(path:Path):
    found=None
    for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
        if line.startswith(MARK): found=json.loads(line[len(MARK):])
    if found is None: raise ValueError(f'no {MARK} line in {path}')
    return found

def main():
    ap=argparse.ArgumentParser();ap.add_argument('logs',nargs='+');ap.add_argument('--template',default='results/templates/main_results.csv');ap.add_argument('--raw-out',default='results/raw/gpu_core_results.csv');args=ap.parse_args()
    runs=[load_one(Path(x)) for x in args.logs]
    methods=['FAS','Random-IFRF','Absolute Output + NNLS']
    raw=[]
    for r in runs:
        seed=r['seed']
        for m in methods:
            clean=float(np.mean([x[m]['parent_f1'] for x in r['clean_targets']]))
            kd=np.nan
            if 'multi_teacher_kd' in r: kd=float(r['multi_teacher_kd'][m]['parent_f1'])
            raw.append({'seed':seed,'method':m,'clean_f1':clean,'kd_f1':kd,'seconds':r.get('seconds')})
    out=Path(args.raw_out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=raw[0].keys());w.writeheader();w.writerows(raw)
    tp=Path(args.template);rows=list(csv.DictReader(tp.open(encoding='utf-8')))
    mapname={'FAS':'FAS','Random-IFRF':'Random-IFRF + NNLS','Absolute Output + NNLS':'Absolute Output + NNLS'}
    for m in methods:
        rr=[x for x in raw if x['method']==m]; clean=np.mean([x['clean_f1'] for x in rr]); kds=[x['kd_f1'] for x in rr if not np.isnan(x['kd_f1'])]
        for row in rows:
            if row['method']==mapname[m]:
                row['clean_f1']=f'{clean:.3f}'
                if kds: row['kd_f1']=f'{np.mean(kds):.3f}'
    with tp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({'n_runs':len(runs),'raw_out':str(out),'template':str(tp)},indent=2))
if __name__=='__main__':main()
