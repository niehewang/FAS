#!/usr/bin/env python3
"""Static quality checks for intervention JSONL before expensive model queries."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from pathlib import Path


def tokens(s): return re.findall(r"\w+|[^\w\s]",s,flags=re.UNICODE)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('jsonl'); ap.add_argument('--max-change-ratio',type=float,default=0.45); args=ap.parse_args()
    rows=[json.loads(x) for x in Path(args.jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    errors=[]; ids=[]; pairs=[]; stats=Counter()
    for i,r in enumerate(rows,1):
        pid=str(r.get('probe_id','')); q=str(r.get('base_query','')); e=str(r.get('edited_query','')); dom=str(r.get('domain','')); typ=str(r.get('type',''))
        ids.append(pid); pairs.append((q,e)); stats[(dom,typ)]+=1
        if not all([pid,q,e,dom,typ]): errors.append((i,'missing required field'))
        if q==e: errors.append((i,'base_query equals edited_query'))
        a=tokens(q); b=tokens(e); denom=max(len(a),len(b),1)
        # SequenceMatcher ratio -> approximate changed-token fraction.
        import difflib
        changed=1-difflib.SequenceMatcher(a=a,b=b).ratio()
        if changed>args.max_change_ratio: errors.append((i,f'large edit ratio {changed:.2f}'))
    dup=[x for x,c in Counter(ids).items() if x and c>1]
    if dup: errors.append((0,f'duplicate probe_id: {dup[:10]}'))
    print('rows:',len(rows)); print('domain/type counts:')
    for k,v in sorted(stats.items()): print(' ',k,v)
    print('warnings/errors:',len(errors))
    for x in errors[:100]: print(' ',x)
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
