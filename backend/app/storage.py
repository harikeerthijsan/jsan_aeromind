"""Optional S3-compatible object storage (the Railway bucket) behind the local data folders.

The code works with plain files under uploads/, reconstructions/ and vendor_refs/. With a bucket configured, every
file written is also pushed to it under the folder's name (uploads/<key>, reconstructions/<dir>/..., vendor_refs/...)
and a file missing locally is downloaded on first use. That lets the Railway server, the GPU worker and a local
development copy share one set of files although none of them can see the others' disks.
Without bucket settings every function here is a no-op, so local-only use and the tests need nothing extra.

Settings (Railway bucket variables, or any S3 provider): S3_BUCKET, S3_ENDPOINT, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
S3_REGION (default auto), S3_ADDRESSING (virtual | path). The AWS_* names are accepted too.

S3_PREFIX (e.g. "dev/"): a namespace for what this installation *writes*. A local development copy has its own
database, whose inspection / job / media ids overlap production's, so its new files must not land on production's
keys (3D model folders are named <inspection>_<job>). Reads look in the namespace first and then in the shared
(production) keys, so data both databases know about is read from the one copy.

STORAGE_MODE=bucket: the bucket is the store and local folders are only a cache. Uploads are removed locally once
pushed, served files are streamed from the bucket (stream()), and trim_cache() keeps the cache under
STORAGE_CACHE_GB (default 5) by deleting the least recently used files. Folders holding a ".in-use" marker (a 3D
build in progress) are never trimmed. STORAGE_MODE=cache (default) keeps local copies as before.
"""
from __future__ import annotations

import os, threading, time, uuid
from contextlib import contextmanager
from pathlib import Path

def _env(*names, default=''):
    return next((os.environ[n] for n in names if os.environ.get(n)), default)

BUCKET = _env('S3_BUCKET', 'BUCKET', 'AWS_S3_BUCKET_NAME')
ENDPOINT = _env('S3_ENDPOINT', 'AWS_ENDPOINT_URL_S3', 'AWS_ENDPOINT_URL', 'ENDPOINT')
REGION = _env('S3_REGION', 'AWS_DEFAULT_REGION', 'AWS_REGION', 'REGION', default='auto')
ACCESS_KEY = _env('S3_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID', 'ACCESS_KEY_ID')
SECRET_KEY = _env('S3_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY', 'SECRET_ACCESS_KEY')
ADDRESSING = _env('S3_ADDRESSING', default='virtual')
PREFIX = (os.getenv('S3_PREFIX', '').strip('/') + '/') if os.getenv('S3_PREFIX', '').strip('/') else ''
MODE = os.getenv('STORAGE_MODE', 'cache').lower()
CACHE_BYTES = int(float(os.getenv('STORAGE_CACHE_GB', '5')) * 1024 ** 3)
IN_USE = '.in-use'
KEEP_RECENT_S = 20 * 60        # never trim files used in the last 20 minutes

_roots: dict[str, Path] = {}   # bucket folder -> local folder
_client = None
_client_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}
_key_locks_lock = threading.Lock()

def enabled() -> bool:
    return bool(BUCKET and ACCESS_KEY and SECRET_KEY)

def bucket_only() -> bool:
    return enabled() and MODE == 'bucket'

def register(prefix: str, folder: Path):
    """Back a local folder with the bucket folder of the same name."""
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

def _base_key(path) -> str | None:
    p = Path(path).resolve()
    for prefix, root in _roots.items():
        if p == root or root in p.parents:
            rel = p.relative_to(root).as_posix()
            return prefix if rel == '.' else f'{prefix}/{rel}'
    return None

def key_for(path) -> str | None:
    """Bucket key this installation writes a local path to (namespace + folder/relative path), or None."""
    base = _base_key(path)
    return None if base is None else PREFIX + base

def read_keys(path) -> list[str]:
    """Keys a local path may be stored under, in lookup order: this installation's namespace, then the shared one."""
    base = _base_key(path)
    if base is None: return []
    return [PREFIX + base, base] if PREFIX else [base]

def _lock(key):
    with _key_locks_lock:
        return _key_locks.setdefault(key, threading.Lock())

def _is_missing(e) -> bool:
    code = str(getattr(e, 'response', {}).get('Error', {}).get('Code', ''))
    return code in ('404', 'NoSuchKey', 'NotFound')

def head(path) -> tuple[str, int] | None:
    """(key, size) of the stored object for a local path, or None."""
    if not enabled(): return None
    for key in read_keys(path):
        try: return key, int(client().head_object(Bucket=BUCKET, Key=key)['ContentLength'])
        except Exception as e:
            if not _is_missing(e): raise
    return None

def fetch(path) -> Path:
    """Make sure a file is present locally (downloading it from the bucket if needed). Returns the path either way;
    a file that exists nowhere simply stays missing, so callers keep their own 'not found' handling."""
    p = Path(path)
    if p.is_file():
        _touch(p); return p
    if not enabled() or not read_keys(p): return p
    with _lock(str(p)):
        if p.is_file(): return p
        p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.part')
        for key in read_keys(p):
            try:
                client().download_file(BUCKET, key, str(tmp)); os.replace(tmp, p); break
            except Exception as e:
                tmp.unlink(missing_ok=True)
                if not _is_missing(e): raise
    return p

def _touch(p):
    try: os.utime(p, None)  # last use, for trim_cache's least-recently-used order (atime is often not updated)
    except OSError: pass

def push(path):
    """Upload one local file (no-op without a bucket)."""
    p = Path(path)
    if not enabled() or not p.is_file() or not (key := key_for(p)): return
    client().upload_file(str(p), BUCKET, key)

def release(path):
    """After push(): in bucket mode the local copy is not kept."""
    if bucket_only() and key_for(path) and head(path):
        Path(path).unlink(missing_ok=True)

def delete(path):
    """Remove this installation's stored copy (never a shared/production key from a namespaced installation)."""
    if enabled() and (key := key_for(path)):
        try: client().delete_object(Bucket=BUCKET, Key=key)
        except Exception: pass

def copy(src_path, dst_path) -> bool:
    """Server-side copy inside the bucket (no download); False when the source does not exist."""
    if not enabled() or not (src := head(src_path)) or not (dst := key_for(dst_path)): return False
    client().copy({'Bucket': BUCKET, 'Key': src[0]}, BUCKET, dst); return True

def stream(path, range_header: str | None = None):
    """Open a stored file for streaming to a browser without saving it locally.
    Returns (iterator of bytes, status, headers) or None when the file does not exist. Supports single byte ranges
    (videos seek with Range requests)."""
    if not enabled(): return None
    for key in read_keys(path):
        args = {'Bucket': BUCKET, 'Key': key, **({'Range': range_header} if range_header and range_header.startswith('bytes=') else {})}
        try: obj = client().get_object(**args)
        except Exception as e:
            if _is_missing(e): continue
            if 'InvalidRange' in str(e): return iter(()), 416, {}
            raise
        headers = {'Content-Length': str(obj['ContentLength']), 'Accept-Ranges': 'bytes'}
        if obj.get('ContentRange'): headers['Content-Range'] = obj['ContentRange']
        if obj.get('ETag'): headers['ETag'] = obj['ETag']
        body = obj['Body']
        def chunks():
            try:
                for c in body.iter_chunks(1024 * 1024): yield c
            finally: body.close()
        return chunks(), (206 if obj.get('ContentRange') else 200), headers
    return None

def _keys(prefix):
    pages = client().get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix.rstrip('/') + '/')
    for page in pages:
        for o in page.get('Contents', []): yield o['Key'], o['Size']

def push_tree(folder, skip=()):
    """Upload every file under a folder, except top-level entries named in skip; files already stored with the same
    size are left alone. Returns the number uploaded."""
    folder = Path(folder)
    if not enabled() or not folder.is_dir() or not (prefix := key_for(folder)): return 0
    have = dict(_keys(prefix)); n = 0
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name.endswith('.part') or p.name == IN_USE or p.relative_to(folder).parts[0] in skip: continue
        key = key_for(p)
        if have.get(key) == p.stat().st_size: continue
        client().upload_file(str(p), BUCKET, key); n += 1
    return n

def fetch_tree(folder, skip=()):
    """Download every stored file of a folder that is missing locally (namespace first, then shared). Returns the count."""
    folder = Path(folder)
    if not enabled() or not (base := _base_key(folder)): return 0
    root = next(r for pre, r in _roots.items() if base == pre or base.startswith(pre + '/'))
    top = base.split('/', 1)[0]; seen = set(); n = 0
    for key_prefix in ([PREFIX + base, base] if PREFIX else [base]):
        strip = len(key_prefix) - len(base) + len(top) + 1
        for key, size in list(_keys(key_prefix)):
            rel = key[strip:]
            if rel in seen: continue
            seen.add(rel); p = root / rel
            if p.relative_to(folder).parts[0] in skip or (p.is_file() and p.stat().st_size == size): continue
            p.unlink(missing_ok=True); fetch(p); n += 1
    return n

@contextmanager
def in_use(folder):
    """Mark a folder as being worked on (e.g. a 3D build) so trim_cache leaves it alone."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); marker = folder / IN_USE
    marker.write_text(str(os.getpid()))
    try: yield folder
    finally: marker.unlink(missing_ok=True)

def trim_cache(max_bytes: int | None = None) -> int:
    """Bucket mode: delete least recently used cached files (all stored in the bucket) until the cache is under the cap.
    Skips files used in the last 20 minutes and folders marked in use. Returns bytes freed."""
    if not bucket_only(): return 0
    cap = CACHE_BYTES if max_bytes is None else max_bytes; now = time.time(); files = []
    for root in _roots.values():
        if not root.is_dir(): continue
        busy = [m.parent for m in root.rglob(IN_USE)]
        for p in root.rglob('*'):
            try:
                if not p.is_file() or p.name.endswith('.part') or p.name == IN_USE or any(b == p.parent or b in p.parents for b in busy): continue
                st = p.stat(); files.append((max(st.st_mtime, st.st_atime), st.st_size, p))
            except OSError: continue
    total = sum(f[1] for f in files); freed = 0
    for used, size, p in sorted(files):
        if total - freed <= cap: break
        if now - used < KEEP_RECENT_S: continue
        if '.thumbs' not in p.parts and not head(p):
            push(p)  # never drop the only copy
            if not head(p): continue
        try: p.unlink(); freed += size
        except OSError: pass
    return freed
