"""Optional S3-compatible object storage (the Railway bucket) behind the local data folders.

The app keeps working with plain files: uploads/, reconstructions/ and vendor_refs/ stay the working copies. With a
bucket configured, every file written is also pushed to it under the folder's name (uploads/<key>, reconstructions/
<dir>/..., vendor_refs/...), and a file that is missing locally is downloaded on first use. That lets the Railway
server and the local GPU worker share one set of files although neither can see the other's disk.
Without bucket settings every function here is a no-op, so local development and the tests need nothing extra.

Settings (Railway bucket variables, or any S3 provider): S3_BUCKET, S3_ENDPOINT, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
S3_REGION (default auto), S3_ADDRESSING (virtual | path). The AWS_* names are accepted too.
"""
from __future__ import annotations

import os, threading, uuid
from pathlib import Path

def _env(*names, default=''):
    return next((os.environ[n] for n in names if os.environ.get(n)), default)

BUCKET = _env('S3_BUCKET', 'BUCKET', 'AWS_S3_BUCKET_NAME')
ENDPOINT = _env('S3_ENDPOINT', 'AWS_ENDPOINT_URL_S3', 'AWS_ENDPOINT_URL', 'ENDPOINT')
REGION = _env('S3_REGION', 'AWS_DEFAULT_REGION', 'AWS_REGION', 'REGION', default='auto')
ACCESS_KEY = _env('S3_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID', 'ACCESS_KEY_ID')
SECRET_KEY = _env('S3_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY', 'SECRET_ACCESS_KEY')
ADDRESSING = _env('S3_ADDRESSING', default='virtual')

_roots: dict[str, Path] = {}   # bucket prefix -> local folder
_client = None
_client_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}
_key_locks_lock = threading.Lock()

def enabled() -> bool:
    return bool(BUCKET and ACCESS_KEY and SECRET_KEY)

def register(prefix: str, folder: Path):
    """Back a local folder with the bucket prefix of the same name."""
    _roots[prefix] = Path(folder).resolve()

def client():
    global _client
    with _client_lock:
        if _client is None:
            import boto3
            from botocore.config import Config
            _client = boto3.client('s3', endpoint_url=ENDPOINT or None, region_name=REGION, aws_access_key_id=ACCESS_KEY,
                                   aws_secret_access_key=SECRET_KEY,
                                   config=Config(s3={'addressing_style': ADDRESSING}, retries={'max_attempts': 5, 'mode': 'standard'}))
        return _client

def key_for(path) -> str | None:
    """Bucket key of a local path, or None when it isn't inside a registered folder."""
    p = Path(path).resolve()
    for prefix, root in _roots.items():
        if p == root or root in p.parents:
            rel = p.relative_to(root).as_posix()
            return prefix if rel == '.' else f'{prefix}/{rel}'
    return None

def _lock(key):
    with _key_locks_lock:
        return _key_locks.setdefault(key, threading.Lock())

def fetch(path) -> Path:
    """Make sure a file is present locally (downloading it from the bucket if needed). Returns the path either way;
    a file that exists nowhere simply stays missing, so callers keep their own 'not found' handling."""
    p = Path(path)
    if p.is_file() or not enabled() or not (key := key_for(p)): return p
    with _lock(key):
        if p.is_file(): return p
        p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.part')
        try:
            client().download_file(BUCKET, key, str(tmp)); os.replace(tmp, p)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if not _is_missing(e): raise
    return p

def _is_missing(e) -> bool:
    code = str(getattr(e, 'response', {}).get('Error', {}).get('Code', ''))
    return code in ('404', 'NoSuchKey', 'NotFound')

def push(path):
    """Upload one local file (no-op without a bucket)."""
    p = Path(path)
    if not enabled() or not p.is_file() or not (key := key_for(p)): return
    client().upload_file(str(p), BUCKET, key)

def delete(path):
    if enabled() and (key := key_for(path)):
        try: client().delete_object(Bucket=BUCKET, Key=key)
        except Exception: pass

def _keys(prefix):
    pages = client().get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix.rstrip('/') + '/')
    for page in pages:
        for o in page.get('Contents', []): yield o['Key'], o['Size']

def push_tree(folder, skip=()):
    """Upload every file under a folder, except top-level entries named in skip; files already in the bucket with the
    same size are left alone. Returns the number uploaded."""
    folder = Path(folder)
    if not enabled() or not folder.is_dir() or not (prefix := key_for(folder)): return 0
    have = dict(_keys(prefix)); n = 0
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name.endswith('.part') or p.relative_to(folder).parts[0] in skip: continue
        key = key_for(p)
        if have.get(key) == p.stat().st_size: continue
        client().upload_file(str(p), BUCKET, key); n += 1
    return n

def fetch_tree(folder, skip=()):
    """Download every file of a bucket folder that is missing locally. Returns the number downloaded."""
    folder = Path(folder)
    if not enabled() or not (prefix := key_for(folder)): return 0
    root = next(r for pre, r in _roots.items() if prefix == pre or prefix.startswith(pre + '/'))
    base = prefix.split('/', 1)[0]; n = 0
    for key, size in list(_keys(prefix)):
        p = root / key[len(base) + 1:]
        if p.relative_to(folder).parts[0] in skip or (p.is_file() and p.stat().st_size == size): continue
        p.unlink(missing_ok=True); fetch(p); n += 1
    return n
