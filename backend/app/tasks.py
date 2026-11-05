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

from . import detector, frames, reconstruct, storage, vendor, media_paths

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
                        frame_root=ROOT/media_paths.frame_dir(m['storage_key'],m['id'])
                        for idx, sec, p, w, h in frames.extract(storage.fetch(ROOT / m['storage_key']),frame_root):
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
            src_scale = (payload.get('source') or {}).get('quality', {}).get('scale')
            if src_scale: r['scale'] = src_scale  # a dense model keeps its sparse model's viewer frame, so the same scale
            else:
                try: r['scale'] = scale_of(work, payload.get('scale_photos') or [])
                except Exception as e: r['scale_error'] = str(e)[:300]  # no GPS etc.: the model is still fine, only unscaled
            if storage.enabled(): progress('uploading', 99, 'Saving the model to storage'); storage.push_tree(work, skip=('images',))  # images: copies of uploads
            return r
        except Exception:
            storage.push(work / 'colmap.log'); raise

def pointcloud_model(payload, progress):
    """payload: {'work', 'clouds': [keys incl. .lax/.prj sidecars], 'photos': [keys in build order]} -> model result.
    The uploaded cloud is the 3D model (no dense photogrammetry). With photos, a COLMAP sparse camera solve is aligned to
    it (GPS/RTK + height search + ICP, app/pointcloud.py) so 3D equipment boxes can be placed; without photos, or
    when the cloud has no coordinate system, the cloud is shown on its own and boxes are not available."""
    import json, time
    import numpy as np
    from . import pointcloud as pc, scale as sc
    work = reconstruct.RECON_DIR / payload['work']; work.mkdir(parents=True, exist_ok=True); log = work / 'colmap.log'; t0 = time.time()
    with _recon_lock:
        try:
            progress('downloading', 1, f"Fetching {len(payload['clouds'])} point cloud file(s) and {len(payload['photos'])} photo(s)")
            clouds = [storage.fetch(ROOT / k) for k in payload['clouds']]; photos = [storage.fetch(ROOT / k) for k in payload['photos']]
            crs = pc.peek_crs(clouds)
            names = [f'photos/{i:05d}{Path(p).suffix.lower()}' for i, p in enumerate(photos[:reconstruct.MAX_IMAGES])]
            gps = {n: sc.read_position(p) for n, p in zip(names, photos)}; gps = {n: g for n, g in gps.items() if g and g.get('alt') is not None}
            use_photos = len(photos) >= 3 and crs is not None and len(gps) >= 4
            roi = None
            if use_photos:  # survey tiles are large: keep the area around the flight
                G = pc.gps_to_cloud(list(gps.values()), crs); r = max(float(np.ptp(G[:, :2], axis=0).max()) / 2, 10) + pc.ROI_MARGIN_M; c = G[:, :2].mean(0)
                roi = (c[0] - r, c[0] + r, c[1] - r, c[1] + r)
            progress('reading_cloud', 4, 'Reading the point cloud')
            cloud = pc.read_clouds(clouds, pc.MODEL_POINTS, roi, progress=lambda msg: progress('reading_cloud', 6, msg))
            origin = np.median(cloud['points'], 0); L = cloud['points'] - origin  # local metres (float precision)
            info = {'files': [Path(k).name for k in payload['clouds']], 'points_total': cloud['total'], 'points_used': cloud['kept'],
                    'crs': cloud['crs'].name if cloud['crs'] is not None else None, 'units': cloud['units'], 'origin': [round(float(v), 3) for v in origin],
                    'cropped_to_flight': roi is not None}
            if use_photos:
                span = lambda a, b: (lambda stage, pct, detail: progress(stage, int(a + (b - a) * pct / 100), detail))
                stats, img, best, cams, sparse = reconstruct.sparse_stage(work, photos, [], span(15, 70), log)  # cameras only: no dense stereo
                progress('aligning', 72, 'Aligning the photos to the point cloud (GPS + ICP)')
                Gl = {n: v - origin for n, v in zip(gps, pc.gps_to_cloud(list(gps.values()), crs))}
                pairs = [(np.asarray(c['center'], float), Gl[c['name']]) for c in cams if c['name'] in Gl]
                if len(pairs) < 4: raise pc.CloudError('too few solved photos have GPS positions to align them with the point cloud')
                a = pc.align(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), np.asarray(sparse['points'], float), L,
                             progress=lambda msg: progress('aligning', 78, msg))
                model_pts = ((L - a['t']) @ a['R']) / a['s']  # cloud in the camera model's frame: registration works unchanged
                (work / 'dense').mkdir(exist_ok=True); reconstruct.write_ply(work / 'dense' / 'fused.ply', model_pts.astype(np.float32), cloud['colors'])
                M = reconstruct.normalising_transform(sparse['points'], cams)
                progress('exporting', 92, 'Preparing the model for the viewer')
                web_files, n_web = reconstruct.export_web(work, M, {'points': model_pts, 'colors': cloud['colors']}, None, cams)
                m_per_unit = a['s'] / float(np.linalg.norm(M[:3, 0]))  # viewer = |M| R (L - t) / s  ->  metres per viewer unit
                info['alignment'] = {k: a[k] for k in ('gps_residual_m', 'vertical_shift_m', 'icp_rms_m', 'inliers_pct')}
                ok = (a['inliers_pct'] or 0) >= 30
                result = {**stats, 'dense': True, 'dense_points': int(len(model_pts)), 'mesh_faces': 0,
                          'files': {'sparse_ply': 'sparse.ply', 'log': 'colmap.log', 'dense_ply': 'dense/fused.ply', **web_files},
                          'web_points': n_web, 'transform': M.round(8).tolist(), 'boxes_available': True}
                if not ok: result['warning'] = f"Only {a['inliers_pct']}% of the photo points touch the point cloud: the cloud may not show this tower, and 3D boxes may be misplaced"
                result['scale'] = {'m_per_unit': round(m_per_unit, 6), 'method': 'pointcloud', 'rtk': all(g.get('rtk') for g in gps.values()), 'cameras': len(pairs),
                                   'residual_median_m': a['gps_residual_m'], 'note': f"from the uploaded point cloud ({cloud['units']})"}
            else:
                M, s = pc.viewer_transform(L)
                web = work / 'web'; web.mkdir(exist_ok=True); k = min(len(L), reconstruct.WEB_POINTS)
                sel = np.random.default_rng(0).choice(len(L), k, replace=False) if len(L) > k else slice(None)
                reconstruct.write_ply(web / 'points.ply', reconstruct._apply(M, L[sel]).astype(np.float32), cloud['colors'][sel])
                (work / 'dense').mkdir(exist_ok=True); reconstruct.write_ply(work / 'dense' / 'fused.ply', L.astype(np.float32), cloud['colors'])
                (web / 'cameras.json').write_text('[]')
                why = ('no photos to place the 3D boxes' if len(photos) < 3 else 'the point cloud has no coordinate system' if crs is None
                       else 'the photos have no GPS positions')
                result = {'registered_images': 0, 'images_total': len(photos), 'sparse_points': 0, 'dense': True, 'dense_points': int(len(L)), 'mesh_faces': 0,
                          'files': {'web_points': 'web/points.ply', 'web_cameras': 'web/cameras.json', 'dense_ply': 'dense/fused.ply', 'log': 'colmap.log'},
                          'web_points': int(k), 'transform': M.round(8).tolist(), 'boxes_available': False, 'boxes_note': f'3D boxes are not available: {why}',
                          'scale': {'m_per_unit': round(1 / s, 6), 'method': 'pointcloud', 'note': f"from the uploaded point cloud ({cloud['units']})"}}
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, 'a', encoding='utf-8') as f: f.write('point cloud import: ' + json.dumps(info) + chr(10))
            result.update(source='pointcloud', pointcloud=info, seconds=round(time.time() - t0))
            if storage.enabled(): progress('uploading', 99, 'Saving the model to storage'); storage.push_tree(work, skip=('images',))
            return result
        except pc.CloudError as e:
            storage.push(log); raise HTTPException(422, str(e))
        except Exception:
            storage.push(log); raise
        finally:
            reconstruct.cleanup(work)

def scale_of(work, photo_keys):
    """Metres per viewer unit of a built model: its cameras (web/cameras.json) fitted to the photos' GPS positions."""
    import json
    from . import scale
    cams = json.loads(storage.fetch(Path(work) / 'web' / 'cameras.json').read_text())
    # the build named its photos photos/00000.jpg ... in this order (reconstruct.prepare_images)
    names = {f'photos/{i:05d}{Path(k).suffix.lower()}': k for i, k in enumerate(photo_keys[:reconstruct.MAX_IMAGES])}
    positions = {n: scale.read_position(storage.fetch(ROOT / k)) for n, k in names.items() if any(c['name'] == n for c in cams)}
    return scale.estimate(cams, positions)

def model_scale(payload, progress=None):
    """payload: {'work', 'photos': [keys in build order]} -> scale dict (see app/scale.py)"""
    try: return scale_of(reconstruct.RECON_DIR / payload['work'], payload['photos'])
    except (ValueError, FileNotFoundError) as e: raise HTTPException(422, f'Cannot detect the scale: {e}')

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

COMPUTE = {'run_detection': detect, 'run_reconstruction': reconstruct_model, 'run_pointcloud_model': pointcloud_model, 'model_scale': model_scale, 'vendor_suggestion': vendor_suggestion,
           'condition_assessment': condition_assessment, 'register_assets': register_assets}
