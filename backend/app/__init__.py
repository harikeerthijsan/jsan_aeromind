"""Load backend/.env into the environment before any module reads its settings, so `uvicorn app.main:app` works
with or without `--env-file .env`. Variables already set in the environment win. AEROMIND_SKIP_DOTENV=1 disables it."""
import os
from pathlib import Path

_env = Path(__file__).resolve().parent.parent / '.env'
if _env.is_file() and not os.getenv('AEROMIND_SKIP_DOTENV'):
    for _line in _env.read_text(encoding='utf-8', errors='replace').splitlines():
        _line = _line.strip()
        if not _line or _line.startswith('#') or '=' not in _line: continue
        _k, _v = _line.split('=', 1)
        _k = _k.strip().removeprefix('export ').strip(); _v = _v.split(' #', 1)[0].strip().strip('"').strip("'")
        if _k and _k not in os.environ: os.environ[_k] = _v
