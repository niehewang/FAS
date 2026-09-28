#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('inputs',nargs='+');ap.add_argument('--output',default='results/raw/open_set_scores.csv');args=ap.parse_args()
    files=[]
    for pat in args.inputs: files.extend(glob.glob(pat) or [pat])
    rows=[]
    for p in sorted(set(files)):
        obj=json.loads(Path(p).read_text(encoding='utf-8'));tid=obj.get('target_id') or Path(p).stem
        setting='known_only' if 'known_only' in tid else ('unknown_A' if 'unknown_A' in tid else ('unknown_B' if 'unknown_B' in tid else 'other'))
        rows.append({'target_id':tid,'setting':setting,'is_unknown':1 if setting=='unknown_A' else 0,'score':obj.get('open_score',''),'p_signal':obj.get('signal_p',''),'p_open':obj.get('open_p',''),'state':obj.get('state','')})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['target_id','setting','is_unknown','score','p_signal','p_open','state']);w.writeheader();w.writerows(rows)
    print(out)
if __name__=='__main__':main()
