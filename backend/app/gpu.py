"""Where GPU work runs: in this process, or on a GPU worker that talks to this server over HTTPS.

Each task has a server half and a GPU half (app/tasks.py):
    prepare(db, **args) -> payload      the server reads what the job needs from the database
    compute(payload, progress) -> result   YOLO / COLMAP / Ollama, files only, no database
    apply(db, result, **args) -> value  the server writes the result to the database
    fail(db, error, **args), progress(db, stage, pct, detail, **args)   optional hooks for job rows

GPU_WORKER=local (default): all three run here, as before.
GPU_WORKER=remote (Railway, no GPU): tasks are queued in worker_tasks. The worker (`python -m app.worker` on the GPU
machine) polls /api/v1/worker/claim with WORKER_TOKEN, gets the prepared payload, computes, and posts progress and the
result back; files travel through the shared bucket. Only the server touches the database, the database stays private
and the GPU machine makes outgoing HTTPS requests only (no open port or tunnel).
"""
from __future__ import annotations

import os, secrets, time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

MODE = os.getenv('GPU_WORKER', 'local').lower()
REMOTE = MODE == 'remote'
TOKEN = os.getenv('WORKER_TOKEN', '')
STALE_AFTER = 45          # seconds without contact before the worker counts as offline
CALL_TIMEOUT = int(os.getenv('GPU_CALL_TIMEOUT', '600'))

@dataclass
class Task:
    prepare: Callable
    apply: Callable
    lane: str = 'long'                   # long (jobs) | quick (calls someone is waiting for)
    fail: Callable | None = None
    progress: Callable | None = None

TASKS: dict[str, Task] = {}

def register(name, **hooks): TASKS[name] = Task(**hooks)

def _session():
    from .database import SessionLocal
    return SessionLocal()

def _compute(name):
    from .tasks import COMPUTE
    return COMPUTE[name]

def _error(e):
    return {'status': e.status_code, 'detail': str(e.detail)} if isinstance(e, HTTPException) else {'status': 500, 'detail': f'{type(e).__name__}: {e}'[:2000]}

def run_local(name, args):
    """prepare -> compute -> apply in this process. Quick tasks raise their errors; jobs record them via fail()."""
    t = TASKS[name]
    try:
        with _session() as db: payload = t.prepare(db, **args)
        def progress(stage, pct, detail):
            if t.progress:
                with _session() as db: t.progress(db, stage, pct, detail, **args)
        result = _compute(name)(payload, progress)
        with _session() as db: return t.apply(db, result, **args)
    except Exception as e:
        if t.fail:
            with _session() as db: t.fail(db, _error(e), **args)
        if t.lane == 'quick': raise

def start(background, name, **args):
    """Fire-and-forget job: a background task here, or queued for the remote worker."""
    if REMOTE: submit(name, args)
    else: background.add_task(run_local, name, args)

def submit(name, args):
    from .models import WorkerTask
    with _session() as db:
        t = WorkerTask(kind=name, lane=TASKS[name].lane, payload=jsonable_encoder(args)); db.add(t); db.commit(); return t.id

def call(name, timeout=None, **args):
    """Run a task and return its value: here, or on the remote worker (waiting for it)."""
    if not REMOTE: return run_local(name, args)
    if not online(): raise HTTPException(503, 'The GPU worker is offline: start backend/run-worker.ps1 on the GPU machine')
    from .models import WorkerTask
    tid = submit(name, args); deadline = time.monotonic() + (timeout or CALL_TIMEOUT)
    with _session() as db:
        while time.monotonic() < deadline:
            time.sleep(.4); db.expire_all(); t = db.get(WorkerTask, tid)
            if t.status == 'done': return t.result
            if t.status == 'failed':
                e = t.error or {}; raise HTTPException(int(e.get('status') or 500), e.get('detail') or 'GPU worker task failed')
        t = db.get(WorkerTask, tid)
        if t.status in ('queued', 'running'): t.status, t.error = 'failed', {'status': 504, 'detail': 'timed out waiting for the GPU worker'}; db.commit()
    raise HTTPException(504, 'The GPU worker did not answer in time')

# ---------- worker presence ----------
def _age(seen):
    if seen.tzinfo is None: seen = seen.replace(tzinfo=timezone.utc)  # SQLite returns naive UTC
    return (datetime.now(timezone.utc) - seen).total_seconds()

def worker():
    """Latest worker contact {'name', 'info', 'seen_at', 'online'} or None."""
    from sqlalchemy import select
    from .models import WorkerState
    with _session() as db:
        w = db.scalar(select(WorkerState).order_by(WorkerState.seen_at.desc()))
        return None if w is None else {'name': w.name, 'info': w.info or {}, 'seen_at': w.seen_at, 'online': _age(w.seen_at) < STALE_AFTER}

def online():
    w = worker(); return bool(w and w['online'])

def info(section, offline):
    """One section of the worker's reported capabilities, or `offline` when no worker is connected."""
    w = worker()
    return (w['info'].get(section) or offline) if w and w['online'] else offline

# ---------- HTTPS endpoints for the worker ----------
router = APIRouter(prefix='/api/v1/worker', include_in_schema=False)

def _auth(token):
    if not TOKEN: raise HTTPException(403, 'Worker access is disabled: set WORKER_TOKEN on the server')
    if not token or not secrets.compare_digest(token, TOKEN): raise HTTPException(401, 'Bad worker token')

class Hello(BaseModel):
    name: str
    lanes: list[str] = ['long', 'quick']
    info: dict | None = None
class Progress(BaseModel):
    stage: str
    pct: int
    detail: str = ''
class Outcome(BaseModel):
    result: object = None
    error: dict | None = None

def _seen(db, hello):
    from sqlalchemy import select
    from .models import WorkerState
    w = db.scalar(select(WorkerState).where(WorkerState.name == hello.name)) or WorkerState(name=hello.name, info={})
    if hello.info is not None: w.info = hello.info
    w.seen_at = datetime.now(timezone.utc); db.add(w); db.commit()

@router.post('/claim')
def claim(hello: Hello, x_worker_token: str | None = Header(None)):
    """Heartbeat + next task. 204 when there is nothing to do."""
    _auth(x_worker_token)
    from sqlalchemy import select, update
    from .models import WorkerTask
    with _session() as db:
        _seen(db, hello)
        for t in db.scalars(select(WorkerTask).where(WorkerTask.status == 'queued', WorkerTask.lane.in_(hello.lanes)).order_by(WorkerTask.id).limit(5)).all():
            if not db.execute(update(WorkerTask).where(WorkerTask.id == t.id, WorkerTask.status == 'queued')
                              .values(status='running', worker=hello.name, started_at=datetime.now(timezone.utc))).rowcount: continue
            db.commit(); args = t.payload or {}; hooks = TASKS.get(t.kind)
            try:
                if hooks is None: raise HTTPException(400, f'Unknown task {t.kind}')
                payload = hooks.prepare(db, **args)
                return {'id': t.id, 'kind': t.kind, 'payload': jsonable_encoder(payload)}
            except Exception as e:  # e.g. the model was deleted meanwhile: the waiting request gets the error
                db.rollback(); _finish(t.id, None, _error(e))
    return Response(status_code=204)

@router.post('/tasks/{task_id}/progress')
def task_progress(task_id: int, p: Progress, x_worker_token: str | None = Header(None)):
    _auth(x_worker_token)
    from .models import WorkerTask
    with _session() as db:
        t = db.get(WorkerTask, task_id)
        if not t or t.status != 'running': raise HTTPException(409, 'Task is not running')
        if (hooks := TASKS.get(t.kind)) and hooks.progress: hooks.progress(db, p.stage, p.pct, p.detail, **(t.payload or {}))
    return {'ok': True}

def _finish(task_id, result, error):
    from .models import WorkerTask
    with _session() as db:
        t = db.get(WorkerTask, task_id); hooks = TASKS.get(t.kind); args = t.payload or {}; value = None
        if error is None:
            try: value = hooks.apply(db, result, **args)
            except Exception as e: db.rollback(); error = _error(e)
        if error is not None and hooks and hooks.fail:
            try: hooks.fail(db, error, **args)
            except Exception: db.rollback()
        t = db.get(WorkerTask, task_id)
        t.status, t.result, t.error, t.finished_at = ('failed' if error else 'done'), (jsonable_encoder(value) if error is None else None), error, datetime.now(timezone.utc)
        db.commit()

@router.post('/tasks/{task_id}/finish')
def task_finish(task_id: int, o: Outcome, x_worker_token: str | None = Header(None)):
    _auth(x_worker_token)
    from .models import WorkerTask
    with _session() as db:
        t = db.get(WorkerTask, task_id)
        if not t or t.status != 'running': raise HTTPException(409, 'Task is not running')
    _finish(task_id, o.result, o.error)
    return {'ok': True}

@router.post('/reset')
def reset(hello: Hello, x_worker_token: str | None = Header(None)):
    """A (re)started worker: whatever it was running can't resume, so fail it and let people re-run it."""
    _auth(x_worker_token)
    from sqlalchemy import select
    from .models import WorkerTask
    with _session() as db:
        ids = db.scalars(select(WorkerTask.id).where(WorkerTask.status == 'running', WorkerTask.worker == hello.name)).all()
    for i in ids: _finish(i, None, {'status': 503, 'detail': 'GPU worker restarted; run it again'})
    return {'failed': len(ids)}
