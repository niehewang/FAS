#!/usr/bin/env python3
"""Export one audit manifest for configs, run plan, probes, and software state."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def git_info():
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
        return {'commit':commit,'dirty':dirty}
    except Exception:return None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='runs/reproducibility_manifest.json'); ap.add_argument('--probe-file',action='append',default=[]); args=ap.parse_args()
    files=[]
    for base in [ROOT/'configs',ROOT/'data/metadata']:
        if base.exists():
            for p in sorted(x for x in base.rglob('*') if x.is_file()): files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
    for x in args.probe_file:
        p=Path(x) if Path(x).is_absolute() else ROOT/x
        if p.exists(): files.append({'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    packages={}
    for name in ['numpy','scipy','pandas','sklearn','torch','transformers','peft','datasets']:
        try:
            m=__import__(name); packages[name]=getattr(m,'__version__','unknown')
        except Exception: packages[name]=None
    obj={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'git':git_info(),'packages':packages,'files':files}
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); print(out)
if __name__=='__main__':main()
