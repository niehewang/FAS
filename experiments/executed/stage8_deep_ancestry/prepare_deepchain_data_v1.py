#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
from jsonl_utils_v1 import strict_read_jsonl, atomic_write_jsonl
DOM=['math','code','medical','science']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--aux',required=True); ap.add_argument('--kd',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--seed',type=int,default=20260927); a=ap.parse_args()
 rows=strict_read_jsonl(a.source); by={d:[] for d in DOM}
 for r in rows:
  d=str(r.get('source_domain') or r.get('domain','')).lower()
  if d in by: by[d].append(r)
 aux=[]; kd=[]; meta={'seed':a.seed,'source':a.source,'domains':{}}
 for d in DOM:
  xs=by[d]
  xs.sort(key=lambda r: hashlib.sha256(f"{a.seed}:{d}:{r.get('id','')}:{r.get('prompt','')}".encode()).hexdigest())
  if len(xs)<500: raise SystemExit(f'{d}: need >=500, got {len(xs)}')
  A=xs[:100]; K=xs[100:350]
  for r in A: aux.append({'id':'aux_'+r['id'],'domain':d.capitalize(),'prompt':r['prompt'],'source_id':r.get('source_id','')})
  for r in K: kd.append({'id':'deepkd_'+r['id'],'domain':d.capitalize(),'prompt':r['prompt'],'source_id':r.get('source_id','')})
  meta['domains'][d]={'candidate_n':len(xs),'aux_n':len(A),'kd_n':len(K),'unused_n':len(xs)-350}
 atomic_write_jsonl(a.aux,aux); atomic_write_jsonl(a.kd,kd)
 meta['aux_total']=len(aux);meta['kd_total']=len(kd);meta['aux_sha256']=hashlib.sha256(Path(a.aux).read_bytes()).hexdigest();meta['kd_sha256']=hashlib.sha256(Path(a.kd).read_bytes()).hexdigest()
 Path(a.manifest).write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
