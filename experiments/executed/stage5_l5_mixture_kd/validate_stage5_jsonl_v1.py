#!/usr/bin/env python3
import argparse,sys
from jsonl_utils_v1 import validate_jsonl
ap=argparse.ArgumentParser();ap.add_argument('--path',required=True);ap.add_argument('--expected',type=int);a=ap.parse_args()
try:
    rows=validate_jsonl(a.path,a.expected);print(f'VALID_JSONL path={a.path} n={len(rows)}')
except Exception as e:
    print(f'INVALID_JSONL {e}',file=sys.stderr);sys.exit(1)
