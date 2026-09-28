#!/usr/bin/env python3
"""Convert DNA-Decomp/modelDNA JSONs to decomposition_predictions-compatible rows."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def jlist(x): return json.dumps(list(x),ensure_ascii=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--truth',required=True);ap.add_argument('--input-dir',required=True);ap.add_argument('--method',required=True,choices=['DNA-Decomp','modelDNA']);ap.add_argument('--output',required=True);ap.add_argument('--strict',action='store_true');args=ap.parse_args()
    truth=list(csv.DictReader(open(args.truth,encoding='utf-8',newline='')));root=Path(args.input_dir);rows=[];missing=[]
    for t in truth:
        tid=t['target_id']; cands=[root/f'{tid}.json',root/f'{tid}.{args.method.lower().replace("-","")}.json']
        p=next((x for x in cands if x.exists()),None)
        if p is None: missing.append(tid);continue
        o=json.loads(p.read_text(encoding='utf-8'))
        row=dict(t); row['method']=args.method
        if args.method=='DNA-Decomp':
            names=o.get('names',[]); pi=o.get('coordinates',[]); support=[n for n,v in zip(names,pi) if float(v)>float(t.get('support_threshold') or 0.05)]; state='decomposable'
        else:
            # modelDNA is an open-weight baseline with a narrower applicability
            # domain than FAS.  Unsupported/failed rows are N/A, not zero-score
            # predictions; excluding them prevents an unfair penalty in the
            # aggregate summary.  The raw JSON still preserves the N/A reason.
            if o.get('status')!='ok':
                continue
            names=o.get('parents',[]); cm=o.get('coordinates',{}); pi=[float(cm.get(n,0.0)) for n in names];support=o.get('support',[]);state='decomposable'
        row.update({'parent_names':jlist(names),'pred_pi':jlist(pi),'pred_support':jlist(support),'state':state,'source_json':str(p)})
        rows.append(row)
    if args.strict and missing: raise SystemExit(f'missing {len(missing)} baseline outputs: {missing[:20]}')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f'{args.method}: {len(rows)}/{len(truth)} -> {out}; missing={len(missing)}')
if __name__=='__main__':main()
