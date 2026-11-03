"""Run a backend module with backend/worker.env loaded (the shell equivalent of run-worker.ps1's settings step).

    python load_worker_env.py -m app.migrate files
"""
import os, runpy, sys
from pathlib import Path

for line in (Path(__file__).resolve().parent / 'worker.env').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1); os.environ[k.strip()] = v.strip().strip('"')
if len(sys.argv) > 2 and sys.argv[1] == '-m':
    sys.argv = [sys.argv[2], *sys.argv[3:]]; runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
