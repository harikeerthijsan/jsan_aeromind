"""GPU worker: runs the AeroMind server's GPU work on this machine (see app/gpu.py).

    python -m app.worker            (or backend/run-worker.ps1, which loads backend/worker.env first)

Settings: AEROMIND_SERVER (https://... of the deployed app), WORKER_TOKEN (same value as on the server), the bucket
(S3_*), and the local GPU tools from backend/.env (YOLO_MODEL_PATH, COLMAP_PATH, VLM_URL ...). It only makes outgoing
HTTPS requests; the database stays on the server. Two lanes run side by side so a long reconstruction doesn't hold up
the quick vision-model calls someone is waiting for in the browser.
"""
from __future__ import annotations

import os, platform, socket, sys, threading, time, traceback
from datetime import datetime

os.environ['GPU_WORKER'] = 'worker'

import httpx
from fastapi import HTTPException

from . import detector, reconstruct, storage, vendor
from .tasks import COMPUTE

SERVER = os.getenv('AEROMIND_SERVER', '').rstrip('/')
TOKEN = os.getenv('WORKER_TOKEN', '')
NAME = os.getenv('WORKER_NAME') or socket.gethostname()
POLL = {'long': 3.0, 'quick': .7}
_stop = threading.Event()
_caps = {'value': None, 'at': 0.0}

def log(*a): print(datetime.now().strftime('%H:%M:%S'), *a, flush=True)

def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device'
    except Exception: return None

def capabilities():
    """What this machine can do, refreshed every 5 minutes (Ollama models or the YOLO weights may change)."""
    if _caps['value'] is None or time.time() - _caps['at'] > 300:
        d = detector.info(load=False)
        try: d['classes'] = {str(k): v for k, v in detector.class_names().items()}
        except Exception: pass
        try:
            from fastapi.encoders import jsonable_encoder
            v = jsonable_encoder(vendor.info())
        except Exception as e: v = {'error': str(e)[:200]}
        _caps.update(value={'detector': d, 'vendor': v, 'reconstruction': reconstruct.info(), 'gpu': _gpu_name(),
                            'host': platform.node(), 'python': sys.version.split()[0], 'storage': 'bucket' if storage.enabled() else 'local'}, at=time.time())
    return _caps['value']

def client():
    return httpx.Client(base_url=SERVER, headers={'X-Worker-Token': TOKEN}, timeout=httpx.Timeout(60, connect=15),
                        auth=(os.getenv('APP_USERNAME', 'aeromind'), os.getenv('APP_PASSWORD')) if os.getenv('APP_PASSWORD') else None)

def lane_loop(lane, http=None):
    http = http or client(); last_error = None
    while not _stop.is_set():
        try:
            r = http.post('/api/v1/worker/claim', json={'name': NAME, 'lanes': [lane], 'info': capabilities()})
            if r.status_code == 204: last_error = None; _stop.wait(POLL[lane]); continue
            r.raise_for_status(); task = r.json(); last_error = None
        except Exception as e:
            if str(e) != last_error: log(f'[{lane}] cannot reach {SERVER}: {e}'); last_error = str(e)
            _stop.wait(10); continue
        tid, kind = task['id'], task['kind']; t0 = time.time(); log(f'[{lane}] task {tid} {kind}')
        last = {'t': 0.0, 'stage': None}
        def progress(stage, pct, detail):  # at most every 2 s (each report is a round trip to the server), always on a new stage
            if stage == last['stage'] and time.time() - last['t'] < 2: return
            last.update(t=time.time(), stage=stage)
            try: http.post(f'/api/v1/worker/tasks/{tid}/progress', json={'stage': stage, 'pct': int(pct), 'detail': detail})
            except Exception: pass  # progress is best effort
        try:
            if kind not in COMPUTE: raise HTTPException(400, f'Unknown task {kind}')
            outcome = {'result': COMPUTE[kind](task['payload'], progress)}
            log(f'[{lane}] task {tid} done in {time.time() - t0:.0f}s')
        except HTTPException as e:
            outcome = {'error': {'status': e.status_code, 'detail': str(e.detail)[:2000]}}; log(f'[{lane}] task {tid} -> {e.status_code} {e.detail}')
        except Exception as e:
            traceback.print_exc(); outcome = {'error': {'status': 500, 'detail': f'{type(e).__name__}: {e}'[:2000]}}
        for attempt in range(30):  # the result is the expensive part: keep trying through a short outage
            try: http.post(f'/api/v1/worker/tasks/{tid}/finish', json=outcome, timeout=300).raise_for_status(); break
            except Exception as e: log(f'[{lane}] could not report task {tid}: {e}'); _stop.wait(10)
        try:
            freed = storage.trim_cache()  # bucket mode: keep the local cache under STORAGE_CACHE_GB
            if freed: log(f'[{lane}] cache trimmed: {freed / 1e9:.2f} GB freed')
        except Exception as e: log(f'[{lane}] cache trim failed: {e}')

def main():
    if not SERVER or not TOKEN: sys.exit('Set AEROMIND_SERVER and WORKER_TOKEN (backend/worker.env)')
    caps = capabilities()
    log(f'GPU worker {NAME} -> {SERVER} · bucket {"on" if storage.enabled() else "OFF (files cannot be shared)"}')
    log(f"YOLO {caps['detector'].get('status')} ({caps['detector'].get('model')}) · COLMAP {caps['reconstruction'].get('status')} · "
        f"vision model {caps['vendor'].get('ai', {}).get('status')} ({caps['vendor'].get('ai', {}).get('model')}) · GPU {caps['gpu']}")
    with client() as http:
        try: log('reset:', http.post('/api/v1/worker/reset', json={'name': NAME}).raise_for_status().json())
        except Exception as e: log('reset failed:', e)
    threads = [threading.Thread(target=lane_loop, args=(l,), daemon=True) for l in ('long', 'quick')]
    for t in threads: t.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        _stop.set(); log('stopping')

if __name__ == '__main__':
    main()
