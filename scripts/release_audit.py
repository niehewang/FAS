#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, sys

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXT = {'.py','.sh','.ps1','.md','.json','.yml','.yaml','.toml','.txt','.csv','.cff','.ini','.cfg','.example'}
BAD_SUFFIX = {'.pem','.key','.p12','.ckpt','.safetensors'}
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{12,}'),
    'api_secret': re.compile(r'\b(?:sk-|hf_)[A-Za-z0-9_-]{16,}'),
    'authorization_header': re.compile(r'Authorization:\s*Bearer', re.I),
    'machine_path': re.compile(r'/(?:home|data)/[^\s\"\']+|[A-Za-z]:\\\\Users\\\\[^\s\"\']+', re.I),
    'internal_workflow': re.compile(r'\b(?:LATEST_HANDOFF|CURRENT_DECISION|NEXT_ACTION|Research OS|EXPERIMENT_EXECUTION|PAPER_WRITING)\b', re.I),
    'venue_specific': re.compile(r'\bIEEE\s+T[A-Z]{2,5}\b|\bIEEE\s+Transactions\s+on\b', re.I),
}
errors=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts: continue
    rel=p.relative_to(ROOT).as_posix()
    if p.suffix.lower() in BAD_SUFFIX:
        errors.append(f'forbidden sensitive artifact: {rel}')
    if p.stat().st_size > 50*1024*1024:
        errors.append(f'large artifact >50 MiB: {rel}')
    if p.suffix.lower() in TEXT_EXT or p.name in {'.gitignore','Makefile'}:
        try: txt=p.read_text(encoding='utf-8')
        except UnicodeDecodeError: continue
        for name,pat in PATTERNS.items():
            if pat.search(txt):
                errors.append(f'{name}: {rel}')
if errors:
    print('PUBLIC_RELEASE_AUDIT_FAIL')
    for e in errors: print('-',e)
    sys.exit(2)
print('PUBLIC_RELEASE_AUDIT_PASS')
