"""The GPU half of each job: plain functions of a JSON payload, with no database access.

The server prepares a payload from the database (which files, which boxes), one of these computes the result on the
machine with the GPU (YOLO, COLMAP, Ollama), and the server applies the result to the database (see app/gpu.py). Run
in-process when the server has the GPU; on the GPU worker when it doesn't (Railway). Files are local paths under
UPLOAD_DIR / RECON_DIR, fetched from the bucket first when they were uploaded elsewhere.
"""
from __future__ import annotations

import hashlib, os, threading
from pathlib import Path
from types import SimpleNamespace as NS
from fastapi import HTTPException

from . import detector, frames, reconstruct, storage, vendor

ROOT = Path(os.getenv('UPLOAD_DIR', './uploads')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
# with a bucket configured, these folders are working copies of it: see app/storage.py
storage.register('uploads', ROOT); storage.register('reconstructions', reconstruct.RECON_DIR); storage.register('vendor_refs', vendor.VENDOR_REFERENCE_DIR)

def _annotate():
    try:
        from . import annotate
    except ImportError: raise HTTPException(503, 'Annotated output needs Pillow: pip install -r requirements.txt')
    return annotate

# ---------- detection ----------
_frames_lock = threading.Lock()

def detect(payload, progress):
    """payload: {'model', 'items': [{id, filename, content_type, storage_key, frames: [{id, storage_key}] | None}]}
    -> {'files': [{id, ok, error, new_frames: [...], targets: [{media_id | frame_index, w, h, boxes}]}]}"""
    items, out = payload['items'], []
    for n, m in enumerate(items, 1):
        res = {'id': m['id'], 'ok': True, 'error': None, 'new_frames': [], 'targets': []}
        try:
            targets = [{'media_id': m['id'], 'storage_key': m['storage_key']}]
            if m['content_type'].startswith('video/'):
                if m.get('frames'):
                    targets = [{'media_id': f['id'], 'storage_key': f['storage_key']} for f in m['frames']]
                else:
                    progress('frame_extraction', int((n - 1) * 100 / len(items)), f"Extracting frames from {m['filename']} ({n}/{len(items)})")
                    with _frames_lock:
                        for idx, sec, p, w, h in frames.extract(storage.fetch(ROOT / m['storage_key']), ROOT / 'frames' / str(m['id'])):
                            data = p.read_bytes(); storage.push(p)
                            res['new_frames'].append({'frame_index': idx, 'frame_time': sec, 'storage_key': p.relative_to(ROOT).as_posix(),
                                                      'filename': frames.label(m['filename'], idx, sec)[0], 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)})
                    targets = [{'frame_index': f['frame_index'], 'storage_key': f['storage_key']} for f in res['new_frames']]
            for k, t in enumerate(targets, 1):
                progress('detection', int(((n - 1) + (k - 1) / len(targets)) * 100 / len(items)),
                         f"Detecting in {m['filename']}" + (f' · frame {k}/{len(targets)}' if len(targets) > 1 or 'frame_index' in t else '') + f' ({n}/{len(items)})')
                for _, w, h, boxes in detector.detect(storage.fetch(ROOT / t['storage_key']), False):
                    res['targets'].append({**{k2: t[k2] for k2 in ('media_id', 'frame_index') if k2 in t}, 'w': w, 'h': h, 'boxes': boxes})
        except Exception as e:  # one unreadable file shouldn't hide every other file's results
            res.update(ok=False, error=f"{m['filename']}: {e}"[:200], targets=[], new_frames=[])
        out.append(res)
    return {'files': out}

# ---------- 3D reconstruction ----------
_recon_lock = threading.Lock()  # one reconstruction at a time: dense stereo uses the whole GPU

def reconstruct_model(payload, progress):
    """payload: {'work', 'photos': [keys], 'videos': [keys], 'dense', 'source': {'dir', 'quality'} | None} -> result dict"""
    work = reconstruct.RECON_DIR / payload['work']
    with _recon_lock:
        try:
            if payload.get('source'):  # densify an existing sparse build
                src = reconstruct.RECON_DIR / payload['source']['dir']; storage.fetch_tree(src, skip=('images',))
                r = reconstruct.densify(src, work, progress, payload['source']['quality'] or {})
            else:
                if storage.enabled(): progress('downloading', 1, f"Fetching {len(payload['photos']) + len(payload['videos'])} file(s) from storage")
                photos = [storage.fetch(ROOT / k) for k in payload['photos']]; videos = [storage.fetch(ROOT / k) for k in payload['videos']]
                r = reconstruct.run(work, photos, videos, payload['dense'], progress)
            if storage.enabled(): progress('uploading', 99, 'Saving the model to storage'); storage.push_tree(work, skip=('images',))  # images: copies of uploads
            return r
        except Exception:
            storage.push(work / 'colmap.log'); raise

# ---------- vision-model suggestions ----------
def vendor_suggestion(payload, progress=None):
    """payload: {'storage_key', 'box', 'class_name', 'refs': [{manufacturer, storage_key, box, size}]} -> suggestion dict"""
    if (s := vendor.status()) != 'ready': raise HTTPException(503, f'Vendor model {vendor.VLM_MODEL or "(none)"} {s.replace("_", " ")} at {vendor.VLM_URL}')
    from PIL import Image
    annotate = _annotate(); box = payload['box']
    crop = annotate.crop(storage.fetch(ROOT / payload['storage_key']), box, (box['image_width'], box['image_height']))
    refs, taken = [], {}
    for r in payload['refs']:  # reviewer-confirmed drone crops first (same viewpoint and resolution as the query)
        if taken.get(r['manufacturer'], 0) >= vendor.REFS_PER_VENDOR or not storage.fetch(ROOT / r['storage_key']).is_file(): continue
        refs.append((r['manufacturer'], annotate.crop(ROOT / r['storage_key'], r['box'], tuple(r['size']), pad=0.05))); taken[r['manufacturer']] = taken.get(r['manufacturer'], 0) + 1
    for maker, p in vendor.folder_references(payload['class_name']):  # then catalog photos from the reference library
        if taken.get(maker, 0) >= vendor.REFS_PER_VENDOR: continue
        try:
            with Image.open(p) as im: refs.append((maker, im.convert('RGB'))); taken[maker] = taken.get(maker, 0) + 1
        except OSError: pass
    try: return vendor.suggest(crop, payload['class_name'], refs)
    except Exception as e: raise HTTPException(502, str(e)[:300])

def condition_assessment(payload, progress=None):
    """payload: {'storage_key', 'box'} -> condition dict"""
    from . import condition
    if (s := vendor.status()) != 'ready': raise HTTPException(503, f'Vision model {vendor.VLM_MODEL or "(none)"} {s.replace("_", " ")} at {vendor.VLM_URL}')
    box = payload['box']
    crop = _annotate().crop(storage.fetch(ROOT / payload['storage_key']), box, (box['image_width'], box['image_height']), pad=0.1)
    try: return condition.assess_verified(crop, box['class_name'], (box['x2'] - box['x1'], box['y2'] - box['y1']))
    except Exception as e: raise HTTPException(502, str(e)[:300])

# ---------- 2D -> 3D registration ----------
def register_assets(payload, progress=None):
    """payload: {'work', 'dense', 'transform', 'media': [...], 'detections': [...], 'reviewed': [...]} -> candidate dicts"""
    from . import registration
    work = reconstruct.RECON_DIR / payload['work']; dense = reconstruct.RECON_DIR / payload['dense']
    for d in {work, dense.parent.parent if dense.parent.name == 'dense' else dense.parent}: storage.fetch_tree(d, skip=('images',))  # built elsewhere
    try:
        out = registration.build_candidates(work, payload['transform'], [NS(**m) for m in payload['media']], [NS(**d) for d in payload['detections']],
                                            dense if dense.is_file() else None, set(payload['reviewed']))
    except Exception as e: raise HTTPException(500, f'2D to 3D registration failed: {e}')
    return [plain(c) for c in out]

def plain(v):
    """numpy arrays/scalars inside dicts and lists -> JSON-friendly Python values."""
    if isinstance(v, dict): return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    return v.tolist() if hasattr(v, 'tolist') else v

COMPUTE = {'run_detection': detect, 'run_reconstruction': reconstruct_model, 'vendor_suggestion': vendor_suggestion,
           'condition_assessment': condition_assessment, 'register_assets': register_assets}
