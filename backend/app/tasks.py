"""The GPU half of each job: plain functions of a JSON payload, with no database access.

The server prepares a payload from the database (which files, which boxes), one of these computes the result on the
machine with the GPU (YOLO, COLMAP, Ollama), and the server applies the result to the database (see app/gpu.py). Run
in-process when the server has the GPU; on the GPU worker when it doesn't (Railway). Files are local paths under
UPLOAD_DIR / RECON_DIR, fetched from the bucket first when they were uploaded elsewhere.
"""
from __future__ import annotations

import hashlib, os, shutil, threading
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

PREFETCH = max(1, int(os.getenv('DETECT_PREFETCH', '8')))  # photos downloaded in parallel ahead of the detector

def fetch_all(keys):
    """Local paths of stored files, in order, downloaded PREFETCH at a time (one download ~1.8 s; a build needs hundreds)."""
    if not storage.enabled() or len(keys) < 2: return [storage.fetch(ROOT / k) for k in keys]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(PREFETCH, thread_name_prefix='fetch') as pool: return list(pool.map(lambda k: storage.fetch(ROOT / k), keys))

class _Prefetch:
    """Reads the next photos from the bucket into memory while YOLO works on the current one (YOLO needs ~0.04 s per
    photo on the GPU, one download ~1.8 s), at most `window` photos ahead (~12 MB each). Nothing is written to the
    local disk."""
    def __init__(self, keys, workers=PREFETCH, window=PREFETCH * 2):
        from concurrent.futures import ThreadPoolExecutor
        self.keys, self.window, self.next = list(dict.fromkeys(keys)), window, 0
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix='prefetch') if storage.enabled() else None
        self.jobs = {}
        self._fill(0)
    def _fill(self, upto):
        while self.pool and self.next < len(self.keys) and self.next < upto + self.window:
            k = self.keys[self.next]; self.next += 1
            self.jobs[k] = self.pool.submit(storage.read_bytes, ROOT / k)
    def get(self, key):
        if key in self.jobs:
            fut = self.jobs.pop(key); self._fill(self.keys.index(key) + 1)
            try: return fut.result()
            except Exception: pass  # a failed parallel download (network hiccup) is retried once below
        return storage.read_bytes(ROOT / key)
    def close(self):
        if self.pool:
            for f in self.jobs.values(): f.cancel()
            self.pool.shutdown(wait=True)

def detect(payload, progress):
    """payload: {'model', 'items': [{id, filename, content_type, storage_key, frames: [{id, storage_key}] | None}]}
    -> {'files': [{id, ok, error, new_frames: [...], targets: [{media_id | frame_index, w, h, boxes}]}]}"""
    items, out = payload['items'], []
    ahead = [m['storage_key'] for m in items if m['content_type'].startswith('image/')] + [f['storage_key'] for m in items for f in (m.get('frames') or [])]
    pre = _Prefetch(ahead); tower = payload.get('tower_type')
    try: return {**_detect(items, out, progress, pre, tower), 'model': detector.model_name(tower), 'tower_type': tower or detector.DEFAULT_TYPE}
    finally: pre.close()

def _detect(items, out, progress, pre, tower=None):
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
                        frame_root=ROOT/media_paths.frame_dir(m['storage_key'],m['id']); video=ROOT/m['storage_key']; had=video.is_file()
                        try: extracted=frames.extract(storage.fetch(video),frame_root)
                        finally:
                            if not had and storage.bucket_only(): video.unlink(missing_ok=True)  # only needed for extraction
                        for idx, sec, p, w, h in extracted:
                            data = p.read_bytes(); storage.push(p); storage.release(p)  # bucket mode: the frame is read back from the bucket
                            res['new_frames'].append({'frame_index': idx, 'frame_time': sec, 'storage_key': p.relative_to(ROOT).as_posix(),
                                                      'filename': frames.label(m['filename'], idx, sec)[0], 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)})
                    targets = [{'frame_index': f['frame_index'], 'storage_key': f['storage_key']} for f in res['new_frames']]
            for k, t in enumerate(targets, 1):
                progress('detection', int(((n - 1) + (k - 1) / len(targets)) * 100 / len(items)),
                         f"Detecting in {m['filename']}" + (f' · frame {k}/{len(targets)}' if len(targets) > 1 or 'frame_index' in t else '') + f' ({n}/{len(items)})')
                for _, w, h, boxes in detector.detect(pre.get(t['storage_key']), False, tower):  # the lattice or the monopole model
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
    with _recon_lock, storage.in_use(work):
        try:
            if payload.get('source'):  # densify an existing sparse build
                src = reconstruct.RECON_DIR / payload['source']['dir']; storage.fetch_tree(src, skip=('images',))
                r = reconstruct.densify(src, work, progress, payload['source']['quality'] or {})
            else:
                if storage.enabled():
                    sizes = [(storage.head(ROOT / k) or (None, 0))[1] for k in payload['photos'] + payload['videos'] if not (ROOT / k).is_file()]
                    need_disk(sum(sizes) + (6 if payload['dense'] else 2) * 1024 ** 3, f"for {len(sizes)} photo(s)/video(s) and COLMAP's working files")
                    progress('downloading', 1, f"Fetching {len(payload['photos']) + len(payload['videos'])} file(s) from storage")
                photos = fetch_all(payload['photos']); videos = [storage.fetch(ROOT / k) for k in payload['videos']]
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

def _cloud_sources(keys):
    """(name, source) per uploaded cloud file. LAS/LAZ are read straight from the bucket (no local copy); formats
    whose readers need a file on disk (E57, PLY, text) are fetched and deleted again afterwards in bucket mode."""
    sources, temp = [], []
    for k in keys:
        p = ROOT / k; ext = p.suffix.lower()
        if ext == '.lax': continue  # spatial index of a LAZ file: not needed to read it in full
        if p.is_file(): sources.append((p.name, p))
        elif ext in ('.las', '.laz') and (f := storage.open_remote(p)) is not None: sources.append((p.name, f))
        elif ext == '.prj': sources.append((p.name, storage.read_bytes(p)))
        else:
            need_disk((storage.head(p) or (None, 0))[1] + 512 * 1024 ** 2, f'to read {p.name}')
            sources.append((p.name, storage.fetch(p))); temp.append(p)
    return sources, temp

FOCUS_RADIUS_M = 12.0              # around the structure the photos look at: every point kept (up to pointcloud.FOCUS_POINTS)
SURROUNDING_POINTS = 4_000_000     # the rest of the cropped area (ground, road, trees) for context
WEB_FOCUS_POINTS = 2_500_000       # viewer: the structure ...
WEB_CONTEXT_POINTS = 1_500_000     # ... and its surroundings (trees, ground, road)

def need_disk(nbytes, why):
    """Stop with a clear message instead of failing half way when the GPU machine's disk is nearly full."""
    free = shutil.disk_usage(ROOT if ROOT.exists() else ROOT.anchor).free
    if free < nbytes:
        raise HTTPException(507, f"Only {free / 1e9:.1f} GB free on the GPU machine's disk; {nbytes / 1e9:.1f} GB needed {why}. Free some space and run it again.")

def pointcloud_model(payload, progress):
    """payload: {'work', 'clouds': [keys incl. .lax/.prj sidecars], 'photos': [keys in build order]} -> model result.
    The uploaded cloud is the 3D model; nothing is reconstructed. Each photo's camera position and direction come
    from its own metadata (RTK position, gimbal angles, factory lens calibration; app/photo_pose.py), read with a
    64 KB range request, so no photo is downloaded and COLMAP is not used. The 3D boxes are then placed exactly as
    for a photo-built model. Without photo poses, or when the cloud has no coordinate system, the cloud is shown on
    its own and boxes are not available."""
    work = reconstruct.RECON_DIR / payload['work']
    try:
        with _recon_lock, storage.in_use(work): return _pointcloud_model(payload, progress, work)
    finally:
        if storage.bucket_only(): shutil.rmtree(work, ignore_errors=True)  # stored in the bucket (or failed): nothing kept locally

def even_sample(idx, P, n, rng, cell=1.0):
    """Up to n of the points idx, as evenly spread over the ground as the cloud allows: the areas round the masts are
    kept in full detail and the rest thinned, so a plain random sample showed denser circles in the viewer."""
    import numpy as np
    if len(idx) <= n: return idx
    idx = idx[rng.permutation(len(idx))]; _, key, cnt = np.unique(np.floor(P[idx, :2] / cell).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    lo, hi = 1, int(cnt.max())
    while lo < hi:  # the largest per-cell cap that keeps the total within n
        k = (lo + hi + 1) // 2
        if np.minimum(cnt, k).sum() <= n: lo = k
        else: hi = k - 1
    key = key.ravel(); order = np.argsort(key, kind='stable'); start = np.r_[0, np.cumsum(cnt)[:-1]]
    rank = np.empty(len(idx), np.int64); rank[order] = np.arange(len(idx)) - start[key[order]]
    return idx[rank < lo]

def _pointcloud_model(payload, progress, work):
    import json, time
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from . import pointcloud as pc, photo_pose as pp
    work.mkdir(parents=True, exist_ok=True); log = work / 'colmap.log'; t0 = time.time(); sources, temp = [], []
    try:
        need_disk(1024 ** 3, 'for the 3D model files')
        photos = payload['photos']
        progress('reading_photos', 2, f'Reading camera positions from {len(photos)} photo(s)')
        with ThreadPoolExecutor(16, thread_name_prefix='meta') as pool:
            metas = list(pool.map(lambda k: pp.read(storage.read_head(ROOT / k, pp.HEAD_BYTES)), photos))
        names = [f'photos/{i:05d}{Path(k).suffix.lower()}' for i, k in enumerate(photos)]  # the 3D-box step names photos the same way
        posed = [(n, m) for n, m in zip(names, metas) if m]
        sources, temp = _cloud_sources(payload['clouds'])
        crs = pc.peek_crs(sources)
        use_photos = len(posed) >= 3 and crs is not None
        cams, roi, pinfo, r = [], None, {}, 0.0
        if use_photos:
            cams, pinfo = pp.poses([m for _, m in posed], crs, np.zeros(3))
            targets = pp.targets(cams)  # a site may have several masts, each flown around: every one keeps its full detail
            pinfo['structures'] = [{'x': round(float(t['T'][0]), 2), 'y': round(float(t['T'][1]), 2), 'photos': t['photos']} for t in targets]
            T = np.mean([t['T'] for t in targets], 0); C = np.array([c['C'] for c in cams])
            r = float(np.percentile(np.linalg.norm(C[:, :2] - T[:2], axis=1), 95)) + pc.ROI_MARGIN_M / 4  # the structures and their surroundings
            roi = (T[0] - r, T[0] + r, T[1] - r, T[1] + r)
        progress('reading_cloud', 5, 'Reading the point cloud')
        focus = [(t['T'][0], t['T'][1], FOCUS_RADIUS_M) for t in targets] if use_photos else None  # the structures keep their full detail; surroundings are thinned
        cloud = pc.read_clouds(sources, SURROUNDING_POINTS if focus else pc.MODEL_POINTS, roi, progress=lambda msg: progress('reading_cloud', 10, msg), focus=focus)
        origin = np.median(cloud['points'], 0); L = cloud['points'] - origin  # local metres (float precision)
        info = {'files': [Path(k).name for k in payload['clouds']], 'points_total': cloud['total'], 'points_used': cloud['kept'],
                'crs': cloud['crs'].name if cloud['crs'] is not None else None, 'units': cloud['units'], 'origin': [round(float(v), 3) for v in origin],
                'cropped_to_flight': roi is not None}
        F = cloud['focus'] if focus else np.ones(len(L), bool)  # the structure: what the 3D-box step works on
        (work / 'dense').mkdir(exist_ok=True); reconstruct.write_ply(work / 'dense' / 'fused.ply', L[F].astype(np.float32), cloud['colors'][F])
        if use_photos:
            progress('cameras', 80, f'Placing {len(cams)} photos on the point cloud')
            for c in cams: c['C'] = c['C'] - origin
            T = T - origin; near = np.linalg.norm(L[:, :2] - T[:2], axis=1) <= max(r - pc.ROI_MARGIN_M / 4, 10)
            M, s = pc.viewer_transform(L[near] if near.sum() > 1000 else L)  # framed on the structure the photos look at
            Z = L[np.linalg.norm(L[:, :2] - T[:2], axis=1) <= 30, 2]; ground = float(np.percentile(Z, 1)) if len(Z) else None
            rel = [c['C'][2] - m['rel_alt'] for c, (_, m) in zip(cams, posed) if m.get('rel_alt') is not None]
            takeoff = float(np.median(rel)) - ground if rel and ground is not None else None  # take-off spot vs ground at the structure
            if takeoff is not None and (not pp.has_vertical(crs) or abs(takeoff) > 15):  # unknown or wrong height system
                for c in cams: c['C'][2] -= takeoff
                pinfo['height_corrected_m'] = round(-takeoff, 2); takeoff = 0.0
            # the photos' directions against the masts in the cloud (positions are RTK; gimbal / compass angles are off ~1-2 deg)
            S, sinfo = pc.split_structure(L, cloud['colors'], F, [t['T'][:2] - origin[:2] for t in targets], FOCUS_RADIUS_M); info['structure'] = sinfo
            ids = payload.get('photo_ids') or []
            if len(ids) == len(photos):
                pid = [i for i, m in zip(ids, metas) if m]
                def small_copy(i):
                    for k in (f'.thumbs/{pid[i]}_480.jpg', None):
                        try:
                            if k: return storage.read_bytes(ROOT / k)
                        except Exception: continue
                    return None
                progress('cameras', 85, 'Correcting photo directions against the point cloud')
                pinfo['directions'] = pp.refine_directions(cams, L[S], small_copy, lambda msg: progress('cameras', 85, msg))
            pp.write_text_model(work / 'sparse' / '0', cams, [n for n, _ in posed])
            on_cloud = pp.check(cams, L)
            progress('exporting', 92, 'Preparing the model for the viewer')
            # viewer: the structure in web/points.ply, trees and the rest in web/context.ply (drawn faint, so they do not hide it)
            rng = np.random.default_rng(0); idx_s, idx_c = np.flatnonzero(S), np.flatnonzero(~S)
            web = rng.choice(idx_s, min(len(idx_s), WEB_FOCUS_POINTS), replace=False); ctx = even_sample(idx_c, L, WEB_CONTEXT_POINTS, rng)
            web_files, n_web = reconstruct.export_web(work, M, {'points': L[web], 'colors': cloud['colors'][web]}, None,
                                                      [{'name': n, 'center': c['C'].tolist(), 'dir': c['R'][2].tolist()} for (n, _), c in zip(posed, cams)], max_points=len(web))
            reconstruct.write_ply(work / 'web' / 'context.ply', reconstruct._apply(M, L[ctx]).astype(np.float32), cloud['colors'][ctx]); web_files['web_context'] = 'web/context.ply'
            view = pc.clear_view(L, S, sinfo, M)
            info['cameras'] = {'source': 'photo metadata (RTK position, gimbal angles, lens calibration)', **pinfo, 'photos': len(cams),
                               'rtk_fixed': sum(m['rtk'] for _, m in posed), 'on_cloud_pct': on_cloud,
                               'takeoff_above_ground_m': round(takeoff, 2) if takeoff is not None else None}
            result = {'registered_images': len(cams), 'images_total': len(photos), 'sparse_points': 0, 'mean_reprojection_error_px': None,
                      'dense': True, 'dense_points': int(F.sum()), 'mesh_faces': 0, 'sparse_model': 'sparse/0', 'camera_source': 'metadata',
                      'files': {'dense_ply': 'dense/fused.ply', 'log': 'colmap.log', **web_files}, 'web_points': n_web,
                      'transform': M.round(8).tolist(), 'boxes_available': True, 'view': view,
                      'scale': {'m_per_unit': round(1 / s, 6), 'method': 'pointcloud', 'rtk': all(m['rtk'] for _, m in posed), 'cameras': len(cams),
                                'note': f"from the uploaded point cloud ({cloud['units']})"}}
            if on_cloud < 50: result['warning'] = (f'Only {on_cloud}% of the photos look at the point cloud: it may show a different place, '
                                                   "or the photos' positions do not match its coordinate system; 3D boxes may be misplaced")
        else:
            M, s = pc.viewer_transform(L)
            web = work / 'web'; web.mkdir(exist_ok=True); k = min(len(L), reconstruct.WEB_POINTS)
            sel = np.random.default_rng(0).choice(len(L), k, replace=False) if len(L) > k else slice(None)
            reconstruct.write_ply(web / 'points.ply', reconstruct._apply(M, L[sel]).astype(np.float32), cloud['colors'][sel])
            (web / 'cameras.json').write_text('[]')
            why = ('no photos to place the 3D boxes' if not photos else 'the point cloud has no coordinate system' if crs is None
                   else 'the photos have no drone position and camera angles in their metadata')
            result = {'registered_images': 0, 'images_total': len(photos), 'sparse_points': 0, 'dense': True, 'dense_points': int(len(L)), 'mesh_faces': 0,
                      'files': {'web_points': 'web/points.ply', 'web_cameras': 'web/cameras.json', 'dense_ply': 'dense/fused.ply', 'log': 'colmap.log'},
                      'web_points': int(k), 'transform': M.round(8).tolist(), 'boxes_available': False, 'boxes_note': f'3D boxes are not available: {why}',
                      'scale': {'m_per_unit': round(1 / s, 6), 'method': 'pointcloud', 'note': f"from the uploaded point cloud ({cloud['units']})"}}
        with open(log, 'a', encoding='utf-8') as f: f.write('point cloud import: ' + json.dumps(info) + chr(10))
        result.update(source='pointcloud', pointcloud=info, seconds=round(time.time() - t0))
        if storage.enabled(): progress('uploading', 99, 'Saving the model to storage'); storage.push_tree(work, skip=('images',))
        return result
    except pc.CloudError as e:
        storage.push(log); raise HTTPException(422, str(e))
    finally:
        for _, src in sources:
            if hasattr(src, 'close'): src.close()
        for p in temp:
            if storage.bucket_only(): p.unlink(missing_ok=True)

def convert_pointcloud(payload, progress):
    """payload: {'key': uploaded LAS/LAZ, 'out': COPC key} -> {'points', 'bytes'}. The browser shows the COPC with Potree.
    Untwine needs the file on disk: it is fetched, converted, the COPC stored, and (bucket mode) both are removed again."""
    from . import copc
    src, out = ROOT / payload['key'], ROOT / payload['out']
    if not copc.untwine_exe(): raise HTTPException(503, 'Untwine is not installed on the GPU machine: conda install -c conda-forge untwine, or set UNTWINE_BIN')
    had = src.is_file(); found = None if had else storage.head(src)
    size = src.stat().st_size if had else (found or (None, 0))[1]
    stream = None if had else storage.open_remote(src)
    try: points = copc.point_count(src if had else stream)  # from the header: no download yet
    finally:
        if stream is not None: stream.close()
    need_disk((0 if had else size) + points * copc.SCRATCH_PER_POINT + points * 8 + 512 * 1024 ** 2, f'to convert {src.name} ({points / 1e6:.0f} M points) to COPC')
    try:
        if not had: progress('downloading', 5, f'Fetching {src.name} ({size / 1e9:.1f} GB)'); storage.fetch(src)
        progress('converting', 20, f'Converting {points / 1e6:.0f} M points to COPC')
        n = copc.convert(src, out, lambda msg: progress('converting', 50, msg))
        progress('uploading', 90, 'Saving the COPC to storage'); storage.push(out)
        return {'points': n, 'bytes': out.stat().st_size}
    finally:
        if storage.bucket_only():
            if not had: src.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

def scale_of(work, photo_keys):
    """Metres per viewer unit of a built model: its cameras (web/cameras.json) fitted to the photos' GPS positions."""
    import json
    from . import scale
    cams = json.loads(storage.fetch(Path(work) / 'web' / 'cameras.json').read_text())
    # the build named its photos photos/00000.jpg ... in this order (reconstruct.prepare_images)
    names = {f'photos/{i:05d}{Path(k).suffix.lower()}': k for i, k in enumerate(photo_keys[:reconstruct.MAX_IMAGES])}
    positions = {n: scale.read_position(storage.read_head(ROOT / k)) for n, k in names.items() if any(c['name'] == n for c in cams)}  # metadata only: no photo download
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
    folders = {work, dense.parent.parent if dense.parent.name == 'dense' else dense.parent}
    for d in folders: storage.fetch_tree(d, skip=('images',))  # built elsewhere
    try:
        out = registration.build_candidates(work, payload['transform'], [NS(**m) for m in payload['media']], [NS(**d) for d in payload['detections']],
                                            dense if dense.is_file() else None, set(payload['reviewed']),
                                            metadata_cameras=payload.get('metadata_cameras', False), progress=lambda msg: progress and progress('registration', 50, msg))
    except Exception as e: raise HTTPException(500, f'2D to 3D registration failed: {e}')
    finally:
        if storage.bucket_only():  # the model stays in the bucket; the copy fetched for this step is not kept
            for d in folders:
                if not (d / storage.IN_USE).exists(): shutil.rmtree(d, ignore_errors=True)
    return [plain(c) for c in out]

def plain(v):
    """numpy arrays/scalars inside dicts and lists -> JSON-friendly Python values."""
    if isinstance(v, dict): return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    return v.tolist() if hasattr(v, 'tolist') else v

COMPUTE = {'run_detection': detect, 'run_reconstruction': reconstruct_model, 'run_pointcloud_model': pointcloud_model, 'convert_pointcloud': convert_pointcloud, 'model_scale': model_scale, 'vendor_suggestion': vendor_suggestion,
           'condition_assessment': condition_assessment, 'register_assets': register_assets}
