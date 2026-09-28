#!/usr/bin/env python3
"""Pre-submission helper: list missing result cells and draft placeholders."""
from pathlib import Path
import csv,re
ROOT=Path(__file__).resolve().parents[1]
missing=[]
for p in sorted((ROOT/'results/templates').glob('*.csv')):
    rows=list(csv.DictReader(p.open(encoding='utf-8')))
    for i,row in enumerate(rows, start=2):
        for k,v in row.items():
            if v is None or str(v).strip()=='': missing.append(f'{p.relative_to(ROOT)}:{i}:{k}')
print(f'Missing result cells: {len(missing)}')
for x in missing[:200]: print('  '+x)
if len(missing)>200: print(f'  ... {len(missing)-200} more')
patterns=[r'\\todoexp',r'\\todoresult',r'\\draftnote']
for pat in patterns:
    hits=[]
    for p in list((ROOT/'sections').glob('*.tex'))+[ROOT/'main.tex']:
        for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if re.search(pat,line): hits.append(f'{p.relative_to(ROOT)}:{n}')
    print(pat, len(hits), 'occurrences')
