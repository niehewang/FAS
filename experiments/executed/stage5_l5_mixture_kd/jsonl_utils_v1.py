from __future__ import annotations
import json, os, hashlib
from pathlib import Path

def atomic_write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def strict_read_jsonl(path):
    p=Path(path); out=[]
    with p.open('r',encoding='utf-8') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            try: out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f'INVALID_JSONL path={p} line={ln}: {e}') from e
    return out

def tolerant_source_read_jsonl(path, max_bad=5):
    """Read immutable raw source rows. Invalid physical records are excluded and reported.
    This is only for formal_raw holdout candidates, never for frozen expert-training rows.
    """
    p=Path(path); out=[]; bad=[]
    with p.open('r',encoding='utf-8',errors='strict') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            try: out.append(json.loads(line))
            except Exception as e:
                bad.append({'line':ln,'error':repr(e),'sha256':hashlib.sha256(line.encode('utf-8')).hexdigest()})
                if len(bad)>max_bad:
                    raise RuntimeError(f'TOO_MANY_INVALID_SOURCE_ROWS path={p} bad>{max_bad}; first={bad[:3]}') from e
    return out,bad

def recover_generated_jsonl_tail(path):
    """For resumable generated JSONL only: drop one malformed final non-empty line.
    Any malformed earlier line is a hard error.
    """
    p=Path(path)
    if not p.exists(): return [],False
    raw=p.read_text(encoding='utf-8')
    lines=raw.splitlines(keepends=True)
    nonempty=[i for i,x in enumerate(lines) if x.strip()]
    last_nonempty=nonempty[-1] if nonempty else -1
    out=[]; repaired=False
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: out.append(json.loads(line))
        except Exception as e:
            if i==last_nonempty:
                repaired=True
                break
            raise RuntimeError(f'INVALID_GENERATED_JSONL_MIDDLE path={p} line={i+1}: {e}') from e
    if repaired:
        tmp=p.with_name(p.name+'.repairtmp')
        with tmp.open('w',encoding='utf-8') as f:
            for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    return out,repaired

def validate_jsonl(path, expected_n=None):
    rows=strict_read_jsonl(path)
    if expected_n is not None and len(rows)!=expected_n:
        raise RuntimeError(f'JSONL_COUNT_MISMATCH path={path} got={len(rows)} expected={expected_n}')
    return rows
