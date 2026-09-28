#!/usr/bin/env python3
"""Run the upstream modelDNA CLI on settings where weight access is valid.

This wrapper deliberately does not reimplement modelDNA.  It invokes the
released CLI, records the exact package version and raw stdout/stderr, and
parses only the parent-weight table needed by the FAS benchmark.

If the baseline is not applicable (e.g., cross-scale KD) do not run this file;
record N/A with the published access reason.  ``--allow-na`` is provided for
batch jobs where unsupported formats should be persisted as an explicit N/A
rather than aborting the whole sweep.
"""
from __future__ import annotations
import argparse,json,re,shutil,subprocess
from importlib.metadata import PackageNotFoundError,version
from pathlib import Path

WEIGHT_RE=re.compile(r'^\s*(\S.*?)\s+([+-]?\d+(?:\.\d+)?)\s+(?:\d+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)')

def parse_weights(text:str, candidates:list[str]):
    out={}
    for line in text.splitlines():
        m=WEIGHT_RE.match(line)
        if not m: continue
        name=m.group(1).strip(); val=float(m.group(2))
        # Prefer exact candidate/base strings; otherwise accept an unambiguous
        # suffix match because the CLI may print shortened local paths.
        hits=[c for c in candidates if name==c or name.endswith(c) or c.endswith(name)]
        if len(hits)==1: out[hits[0]]=val
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suspect',required=True)
    ap.add_argument('--parents',nargs='+',required=True)
    ap.add_argument('--base')
    ap.add_argument('--layers',action='store_true')
    ap.add_argument('--mergekit',action='store_true')
    ap.add_argument('--support-threshold',type=float,default=0.05)
    ap.add_argument('--output',required=True)
    ap.add_argument('--allow-na',action='store_true')
    args=ap.parse_args()
    exe=shutil.which('modeldna')
    if not exe:
        raise SystemExit('modeldna CLI not found; install pinned requirements-baselines.txt')
    try: ver=version('modeldna')
    except PackageNotFoundError: ver='unknown'
    cmd=[exe,'decompose',args.suspect,*args.parents]
    if args.base: cmd += ['--base',args.base]
    if args.layers: cmd += ['--layers']
    if args.mergekit: cmd += ['--mergekit']
    cp=subprocess.run(cmd,capture_output=True,text=True)
    candidates=list(args.parents)+([args.base] if args.base else [])
    weights=parse_weights(cp.stdout,candidates) if cp.returncode==0 else {}
    status='ok' if cp.returncode==0 and weights else ('parse_failed' if cp.returncode==0 else 'N/A')
    parent_weights={p:float(weights.get(p,0.0)) for p in args.parents}
    pos={p:max(0.0,v) for p,v in parent_weights.items()}; den=sum(pos.values())
    coords={p:(v/den if den>0 else 0.0) for p,v in pos.items()}
    support=[p for p,v in coords.items() if v>args.support_threshold]
    obj={
      'method':'modelDNA','status':status,'applicable':bool(cp.returncode==0),
      'package_version':ver,'command':cmd,'returncode':cp.returncode,
      'suspect':args.suspect,'parents':args.parents,'base':args.base or '',
      'raw_weights':parent_weights,'coordinates':coords,'support':support,
      'support_threshold':args.support_threshold,'stdout':cp.stdout,'stderr':cp.stderr,
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:obj[k] for k in ['status','package_version','coordinates','support']},ensure_ascii=False,indent=2))
    if cp.returncode!=0 and not args.allow_na: raise SystemExit(cp.returncode)
if __name__=='__main__':main()
