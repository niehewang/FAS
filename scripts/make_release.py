#!/usr/bin/env python3
"""Create a clean Overleaf/research release ZIP without transient build caches."""
from __future__ import annotations
import argparse, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE_DIRS={'.git','.pytest_cache','__pycache__','runs','fas_pdf_verify'}
EXCLUDE_SUFFIX={'.aux','.log','.out','.blg','.bbl','.synctex.gz','.pyc'}

def keep(p:Path,include_pdf:bool):
    rel=p.relative_to(ROOT)
    if any(x in EXCLUDE_DIRS for x in rel.parts): return False
    if p.suffix in EXCLUDE_SUFFIX or p.name.endswith('.synctex.gz'): return False
    if p.name in {'main.pdf','supplementary.pdf'} and not include_pdf:return False
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='../FAS_TPAMI_Chinese_Draft_v0.11.zip');ap.add_argument('--include-pdf',action='store_true');args=ap.parse_args()
    out=(ROOT/args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and keep(p,args.include_pdf): z.write(p,p.relative_to(ROOT))
    print(out)

if __name__=='__main__':main()
