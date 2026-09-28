#!/usr/bin/env python3
import argparse,yaml
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--parent-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
r=Path(a.parent_root)
c={'merge_method':'dare_ties','base_model':'Qwen/Qwen3-4B-Base','models':[{'model':str(r/'math'),'parameters':{'weight':0.40,'density':0.5}},{'model':str(r/'code'),'parameters':{'weight':0.35,'density':0.5}},{'model':str(r/'medical'),'parameters':{'weight':0.25,'density':0.5}}],'parameters':{'normalize':True},'tokenizer_source':'base','dtype':'bfloat16'}
Path(a.output).write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
