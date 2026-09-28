#!/usr/bin/env python3
"""Record reproducibility metadata for one experiment run.

The script hashes small configuration files and writes checkpoint identity
without hashing multi-GB weights by default. If the checkpoint is a directory,
it records a deterministic manifest of filenames/sizes plus selected config
files. Use --full-hash only when exact weight-byte hashing is affordable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess
from pathlib import Path
from datetime import datetime, timezone


def sha256_file(p: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def dir_identity(path: Path, full_hash: bool=False):
    items=[]
    for p in sorted(x for x in path.rglob('*') if x.is_file()):
        rel=str(p.relative_to(path)); size=p.stat().st_size
        rec={"path":rel,"size":size}
        if full_hash or p.name in {"config.json","adapter_config.json","tokenizer_config.json","generation_config.json"} or p.suffix in {'.yaml','.yml'}:
            rec['sha256']=sha256_file(p)
        items.append(rec)
    digest=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    return {"type":"directory_manifest","sha256":digest,"files":items}


def git_info(root: Path):
    try:
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True,stderr=subprocess.DEVNULL).strip())
        return {"commit":commit,"dirty":dirty}
    except Exception:
        return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--config',action='append',default=[])
    ap.add_argument('--checkpoint'); ap.add_argument('--parents',default=''); ap.add_argument('--output',required=True); ap.add_argument('--full-hash',action='store_true')
    args=ap.parse_args(); root=Path(__file__).resolve().parents[1]
    cfgs=[]
    for x in args.config:
        p=Path(x); cfgs.append({"path":str(p),"sha256":sha256_file(p)})
    ck=None
    if args.checkpoint:
        p=Path(args.checkpoint)
        if p.is_dir(): ck={"path":str(p),**dir_identity(p,args.full_hash)}
        elif p.is_file(): ck={"path":str(p),"type":"file","size":p.stat().st_size,"sha256":sha256_file(p) if args.full_hash or p.stat().st_size<50_000_000 else None}
        else: ck={"path":args.checkpoint,"type":"unresolved_or_remote"}
    meta={
        "run_id":args.run_id,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "parents":[x for x in args.parents.split(';') if x],"configs":cfgs,"checkpoint":ck,
        "environment":{"python":platform.python_version(),"platform":platform.platform(),"cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES','')},
        "repository":git_info(root),
    }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
