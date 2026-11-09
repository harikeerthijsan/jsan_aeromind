"""Photo 3D view: measure on the drone photos, with the 3D model's point cloud behind every pixel.

Every completed 3D model knows where each photo was taken (camera poses from drone metadata or COLMAP) and has a
dense point cloud in the same frame. The cloud is projected into a photo once (a depth map: the nearest point per
small cell, so what the camera actually sees), and a click on the photo is the 3D point in its cell (or the nearest
filled cell close by: thin steel is only a few pixels wide). Coordinates come back in metres when the model has a
real-world scale (GPS/RTK or LiDAR), with the height above the ground (the lowest 1 % of the cloud).
"""
from __future__ import annotations
import threading
from collections import OrderedDict
from pathlib import Path

CELL = 4            # depth-map cell, in full-resolution photo pixels
SNAP_CELLS = 6      # a click snaps to a filled cell within this many cells (24 px)
_models: OrderedDict = OrderedDict()   # model id -> loaded model (a couple at a time: clouds are large)
_depth: OrderedDict = OrderedDict()    # (model id, photo name) -> depth map
_lock = threading.Lock()

class Unavailable(RuntimeError): pass

def _sparse_dir(work: Path, quality: dict) -> Path | None:
    for d in ([work / quality['sparse_model']] if quality.get('sparse_model') else []) + [work / 'sparse' / '0'] + sorted((work / 'sparse').glob('*')):
        if d.is_dir() and any((d / n).is_file() for n in ('images.txt', 'images.bin')): return d
    return None

def load(model_id: int, work: Path, dense: Path, quality: dict, photos: list):
    """photos: the build's photos (Media rows, sorted as the build named them photos/00000 ...)."""
    import numpy as np, tempfile, shutil
    from . import registration, reconstruct
    with _lock:
        if model_id in _models: _models.move_to_end(model_id); return _models[model_id]
    sparse = _sparse_dir(work, quality)
    if sparse is None: raise Unavailable('This 3D model has no camera positions for its photos')
    if not dense.is_file(): raise Unavailable('This 3D model has no dense point cloud to measure on')
    tmp = Path(tempfile.mkdtemp(prefix='photoview-'))
    try:
        reconstruct.write_text_model(sparse, tmp)
        poses = registration._read_images(tmp / 'images.txt'); cams = registration._read_camera_models(tmp / 'cameras.txt')
    finally: shutil.rmtree(tmp, ignore_errors=True)
    P = np.asarray(reconstruct.read_ply(dense)['points'], np.float32)
    M = np.asarray(quality.get('transform') or np.eye(4), float); s = float(np.linalg.norm(M[:3, 0]))
    mpu = (quality.get('scale') or {}).get('m_per_unit')
    sample = P[::max(1, len(P) // 200000)].astype(np.float64) @ M[:3, :3].T + M[:3, 3]  # viewer frame (+Y up), for the ground level
    names = {m.id: f'photos/{i:05d}{Path(m.storage_key).suffix.lower()}' for i, m in enumerate(photos)}
    model = {'P': P, 'M': M, 's': s, 'mpu': float(mpu) if mpu else None, 'ground': float(np.percentile(sample[:, 1], 1)),
             'poses': poses, 'cams': cams, 'names': names, 'photos': {m.id: m for m in photos}}
    with _lock:
        _models[model_id] = model
        while len(_models) > 2: _models.popitem(last=False)
    return model

def _intrinsics(cam):
    """fx, fy, cx, cy, k1, k2, p1, p2 of a COLMAP camera (OpenCV lens model; the simpler models map onto it)."""
    model, p = cam[0], list(cam[1])
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        return [p[0], p[0], p[1], p[2], p[3] if len(p) > 3 else 0.0, p[4] if len(p) > 4 else 0.0, 0.0, 0.0]
    if model == 'PINHOLE': return [*p[:4], 0.0, 0.0, 0.0, 0.0]
    return [*p[:4], *(list(p[4:8]) + [0.0] * 4)[:4]]

def _viewer_pose(model, pose, cam):
    """Where the photo was taken in the 3D viewer's frame (the frame of web/points.ply): centre and the camera's right,
    down and forward axes, plus its lens, so the viewer can stand exactly where the drone was (photo walk)."""
    import numpy as np
    M = model['M']; Rm = M[:3, :3] / model['s']; R = np.asarray(pose[4], float)
    centre = M[:3, :3] @ np.asarray(pose[2], float) + M[:3, 3]
    r = lambda v: [round(float(x), 6) for x in v]
    return {'centre': r(centre), 'right': r(Rm @ R[0]), 'down': r(Rm @ R[1]), 'forward': r(Rm @ R[2]),
            'lens': [round(float(x), 6) for x in _intrinsics(cam)]}

def cameras(model):
    """Photos that have a camera pose, with their position for ordering around the structure."""
    import numpy as np
    out = []
    for mid, name in model['names'].items():
        pose = model['poses'].get(name)
        if not pose: continue
        cam = model['cams'][pose[3]]; m = model['photos'][mid]
        out.append({'media_id': mid, 'filename': m.filename, 'url': f'/uploads/{m.storage_key}', 'width': cam[2], 'height': cam[3], 'centre': np.asarray(pose[2], float),
                    'pose': _viewer_pose(model, pose, cam)})
    if out:  # around the structure: by the angle of each camera seen from the middle of all of them
        C = np.array([o['centre'] for o in out]); mid = C.mean(0)
        for o in out: o['angle'] = float(np.degrees(np.arctan2(o['centre'][1] - mid[1], o['centre'][0] - mid[0])) % 360)
        for o in out: del o['centre']
    return sorted(out, key=lambda o: (round(o.get('angle', 0) / 5), o['filename']))

def depth_map(model_id, model, media_id):
    import numpy as np
    from . import registration
    key = (model_id, media_id)
    with _lock:
        if key in _depth: _depth.move_to_end(key); return _depth[key]
    name = model['names'].get(media_id); pose = model['poses'].get(name) if name else None
    if not pose: raise Unavailable('This photo has no camera position in the 3D model')
    cam = model['cams'][pose[3]]; W, H = cam[2], cam[3]; cols, rows = -(-W // CELL), -(-H // CELL)
    best_z = np.full(rows * cols, np.inf, np.float32); grid = np.full(rows * cols, -1, np.int64)
    P = model['P']
    for a in range(0, len(P), 1_000_000):  # in chunks: the server's memory stays small whatever the cloud size
        uv, z = registration.project(P[a:a + 1_000_000].astype(np.float64), pose[4], pose[5], cam)
        ok = np.flatnonzero((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        if not len(ok): continue
        cell = (uv[ok, 1] // CELL).astype(np.int64) * cols + (uv[ok, 0] // CELL).astype(np.int64); zz = z[ok].astype(np.float32)
        order = np.argsort(-zz); cell, zz, ids = cell[order], zz[order], ok[order] + a
        closer = zz < best_z[cell]; c2 = cell[closer]; best_z[c2] = zz[closer]; grid[c2] = ids[closer]  # nearest point wins its cell
    dm = {'grid': grid.reshape(rows, cols), 'cols': cols, 'rows': rows, 'zgrid': best_z.reshape(rows, cols)}
    with _lock:
        _depth[key] = dm
        while len(_depth) > 8: _depth.popitem(last=False)  # ~12 MB each: the server's memory stays small
    return dm

def locate(model_id, model, media_id, pixels):
    """[[u, v], ...] in full-resolution photo pixels -> 3D points (metres when scaled) or None per pixel."""
    import numpy as np
    dm = depth_map(model_id, model, media_id); grid, zg = dm['grid'], dm['zgrid']; out = []
    mpu = model['mpu'] or 1.0; M = model['M']; centre = np.asarray(model['poses'][model['names'][media_id]][2], float)
    for u, v in pixels:
        c0, r0 = int(u // CELL), int(v // CELL); best = None
        for rad in range(SNAP_CELLS + 1):  # nearest filled cell; within a ring, the closest surface
            r1, r2, c1, c2 = max(0, r0 - rad), min(dm['rows'], r0 + rad + 1), max(0, c0 - rad), min(dm['cols'], c0 + rad + 1)
            ids, zs = grid[r1:r2, c1:c2].ravel(), zg[r1:r2, c1:c2].ravel(); keep = ids >= 0
            if keep.any(): k = int(np.argmin(np.where(keep, zs, np.inf))); best = int(ids[k]); break
        if best is None: out.append(None); continue
        V = M[:3, :3] @ model['P'][best].astype(np.float64) + M[:3, 3]
        out.append({'x': round(float(V[0]) * mpu, 3), 'y': round(float(-V[2]) * mpu, 3), 'z': round(float(V[1]) * mpu, 3),  # east, north, up
                    'height_m': round(float(V[1] - model['ground']) * mpu, 3), 'distance_from_camera': round(float(np.linalg.norm(model['P'][best] - centre)) * model['s'] * mpu, 3),
                    'snapped_px': rad * CELL,
                    # in the 3D viewer's frame (model units), with the ground point below it: the photo walk draws the measurement in 3D
                    'viewer': [round(float(x), 5) for x in V], 'foot': [round(float(V[0]), 5), round(model['ground'], 5), round(float(V[2]), 5)]})
    return out
