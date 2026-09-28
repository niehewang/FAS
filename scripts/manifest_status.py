#!/usr/bin/env python3
"""Inspect or update the deterministic experiment run manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv')
    sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('summary')
    p=sp.add_parser('set');p.add_argument('run_id');p.add_argument('status',choices=['planned','ready','running','done','failed','skipped']);p.add_argument('--checkpoint-or-api');p.add_argument('--notes')
    p=sp.add_parser('next');p.add_argument('--status',default='planned');p.add_argument('--limit',type=int,default=10)
    args=ap.parse_args(); path=Path(args.manifest); df=pd.read_csv(path,keep_default_na=False)
    if args.cmd=='summary':
        print(df.groupby(['scenario','status']).size().unstack(fill_value=0).to_string())
        print('\nTotal:',len(df)); return
    if args.cmd=='next':
        z=df[df.status==args.status].head(args.limit)
        print(z[['run_id','scenario','parents','student','postprocess']].to_string(index=False)); return
    m=df.run_id.astype(str)==args.run_id
    if not m.any(): raise SystemExit(f'unknown run_id: {args.run_id}')
    df.loc[m,'status']=args.status
    if args.checkpoint_or_api is not None: df.loc[m,'checkpoint_or_api']=args.checkpoint_or_api
    if args.notes is not None: df.loc[m,'notes']=args.notes
    df.to_csv(path,index=False)
    print(f'{args.run_id} -> {args.status}')

if __name__=='__main__':main()
