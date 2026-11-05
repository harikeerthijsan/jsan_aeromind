"""Load backend/.env (and backend/storage.env) into the environment before any module reads its settings, so
`uvicorn app.main:app` works with or without `--env-file .env`. Variables already set in the environment win.
AEROMIND_SKIP_DOTENV=1 disables it. storage.env holds the bucket settings of a local development copy (shared bucket,
own S3_PREFIX namespace) and overrides the same names from .env or the shell; the GPU worker (AEROMIND_ROLE=worker, set by run-worker.ps1) writes production keys and skips it."""
import os
from pathlib import Path

_here = Path(__file__).resolve().parent.parent  # storage.env first: its cache folders / namespace win over .env
_files = ([] if os.getenv('AEROMIND_ROLE') == 'worker' else [_here / 'storage.env']) + [_here / '.env']
for _env in (f for f in _files if f.is_file() and not os.getenv('AEROMIND_SKIP_DOTENV')):
    for _line in _env.read_text(encoding='utf-8', errors='replace').splitlines():
        _line = _line.strip()
        if not _line or _line.startswith('#') or '=' not in _line: continue
        _k, _v = _line.split('=', 1)
        _k = _k.strip().removeprefix('export ').strip(); _v = _v.split(' #', 1)[0].strip().strip('"').strip("'")
        if _k and (_k not in os.environ or _env.name == 'storage.env'): os.environ[_k] = _v  # storage.env overrides (also uvicorn --env-file .env)
