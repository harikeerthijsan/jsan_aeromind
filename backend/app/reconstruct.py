"""Photogrammetry: build a 3D model of the tower from an inspection's photos and videos with COLMAP.

COLMAP (BSD licence, https://colmap.github.io) is run as a command-line program. It is found via COLMAP_PATH, then PATH, then
the usual unzip locations (e.g. D:/tools/colmap/bin/colmap.exe); COLMAP_PATH=off disables it. Pipeline, per the COLMAP CLI docs:
  feature_extractor -> sequential_matcher (ordered flights/video) or exhaustive_matcher (small sets) -> mapper   = sparse
  image_undistorter -> patch_match_stereo (CUDA) -> stereo_fusion -> poisson_mesher (-> mesh_simplifier)        = dense
The result is then levelled (up = away from where the drone cameras looked), centred, scaled and thinned so a browser
can draw it, and the bulky depth maps are deleted. Without photo GPS/known distances the model has no real-world scale.
"""
import json, os, re, shutil, subprocess, time
from pathlib import Path

# where the official Windows zip usually ends up when unpacked
_CANDIDATES = [Path(d) / 'colmap' / 'bin' / 'colmap.exe' for d in ('D:/tools', 'C:/tools', 'D:/', 'C:/')] +               [Path(os.path.expandvars(p)) for p in (r'%ProgramFiles%\COLMAPin\colmap.exe', r'%LOCALAPPDATA%\COLMAPin\colmap.exe', r'%USERPROFILE%\colmapin\colmap.exe')]

def colmap_exe():
    """Path to colmap(.exe): COLMAP_PATH if set (even if wrong, so the message can say so), else PATH, else a known location."""
    p = os.getenv('COLMAP_PATH', '').strip().strip('"')
    if p.lower() == 'off': return ''
    if p: return p
    return shutil.which('colmap') or next((str(c) for c in _CANDIDATES if c.is_file()), '')
RECON_DIR = Path(os.getenv('RECON_DIR', str(Path(__file__).resolve().parent.parent / 'reconstructions')))
VIDEO_STRIDE = max(1, int(os.getenv('RECON_VIDEO_STRIDE', '10')))  # denser than review frames: 3D needs overlap
MAX_IMAGES = max(3, int(os.getenv('RECON_MAX_IMAGES', '300')))
DENSE_SIZE = int(os.getenv('RECON_DENSE_MAX_SIZE', '1600'))  # px; dense stereo time and memory grow with the square
WEB_POINTS, WEB_FACES = 1_500_000, 1_200_000  # what a browser draws smoothly
EXHAUSTIVE_UP_TO = 80  # small sets: match every pair; larger ordered sets: neighbours in capture order

def status():
    exe = colmap_exe()
    if not exe: return 'not_found'
    return 'ready' if Path(exe).is_file() or shutil.which(exe) else 'missing'  # missing = COLMAP_PATH points nowhere

def info(): return {'status': status(), 'colmap': colmap_exe() or None, 'colmap_path_env': os.getenv('COLMAP_PATH') or None, 'video_stride': VIDEO_STRIDE, 'max_images': MAX_IMAGES, 'dense_max_size': DENSE_SIZE}

# ---------- PLY (binary little-endian or ascii) ----------
_T = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
      'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4', 'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}

def read_ply(path):
    """{'points': (N,3) float32, 'colors': (N,3) uint8 or None, 'faces': (M,3) int32 or None}. Triangle faces only."""
    import numpy as np
    with open(path, 'rb') as f:
        fmt, elements = None, []
        while (line := f.readline().decode('ascii', 'replace').strip()) != 'end_header':
            p = line.split()
            if not p: continue
            if p[0] == 'format': fmt = p[1]
            elif p[0] == 'element': elements.append([p[1], int(p[2]), []])
            elif p[0] == 'property': elements[-1][2].append(('list', p[2], p[3], p[4]) if p[1] == 'list' else (p[2], p[1]))
        data = {}
        if fmt == 'ascii':
            rows = f.read().decode('ascii').split('\n'); i = 0
            for name, count, props in elements:
                block = [r.split() for r in rows[i:i + count]]; i += count
                if name == 'vertex': data[name] = {pn: np.array([float(r[k]) for r in block]) for k, (pn, _) in enumerate(props)}
                elif name == 'face': data[name] = np.array([[int(x) for x in r[1:4]] for r in block], dtype=np.int32)
        else:
            end = '<' if 'little' in (fmt or '') else '>'
            for name, count, props in elements:
                if props and props[0][0] == 'list':
                    _, ct, it, _ = props[0]
                    dt = np.dtype([('n', end + _T[ct]), ('v', end + _T[it], 3)])
                    arr = np.frombuffer(f.read(count * dt.itemsize), dt, count)
                    if count and (arr['n'] != 3).any(): raise ValueError('only triangle meshes are supported')
                    data[name] = arr['v'].astype(np.int32)
                else:
                    dt = np.dtype([(pn, end + _T[t]) for pn, t in props])
                    data[name] = np.frombuffer(f.read(count * dt.itemsize), dt, count)
    v = data.get('vertex')
    if v is None: raise ValueError(f'{Path(path).name}: no vertices')
    pts = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], 1).astype(np.float32)
    names = v.dtype.names if hasattr(v, 'dtype') else tuple(v)
    cols = np.stack([np.asarray(v[c]) for c in ('red', 'green', 'blue')], 1).astype(np.uint8) if names and 'red' in names else None
    return {'points': pts, 'colors': cols, 'faces': data.get('face')}

def write_ply(path, points, colors=None, faces=None):
    import numpy as np
    n = len(points); fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')] + ([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')] if colors is not None else [])
    v = np.empty(n, fields); v['x'], v['y'], v['z'] = points[:, 0], points[:, 1], points[:, 2]
    if colors is not None: v['red'], v['green'], v['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    head = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}', 'property float x', 'property float y', 'property float z']
    if colors is not None: head += ['property uchar red', 'property uchar green', 'property uchar blue']
    if faces is not None: head += [f'element face {len(faces)}', 'property list uchar int vertex_indices']
    with open(path, 'wb') as f:
        f.write(('\n'.join(head + ['end_header']) + '\n').encode('ascii')); f.write(v.tobytes())
        if faces is not None:
            fa = np.empty(len(faces), [('n', 'u1'), ('v', '<i4', 3)]); fa['n'] = 3; fa['v'] = faces; f.write(fa.tobytes())

# ---------- cameras and orientation ----------
def read_cameras_txt(images_txt):
    """Camera centres and viewing directions from a COLMAP TXT model's images.txt."""
    import numpy as np
    out, lines = [], [l for l in Path(images_txt).read_text().splitlines() if l and not l.startswith('#')]
    for l in lines[0::2]:  # each image takes two lines; the second lists its 2D points
        p = l.split(); qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        out.append({'name': ' '.join(p[9:]), 'center': (-R.T @ np.array([tx, ty, tz])).tolist(), 'dir': (R.T @ np.array([0, 0, 1.0])).tolist()})
    return out

def normalising_transform(points, cameras):
    """4x4 matrix that puts the scene upright (+Y up), centred at the origin, ~10 units across.
    Up = opposite of the mean viewing direction (drones look down/inwards), refined by the plane of the camera path."""
    import numpy as np
    C = np.array([c['center'] for c in cameras]); D = np.array([c['dir'] for c in cameras])
    up = -D.mean(0); up = up / (np.linalg.norm(up) or 1)
    if len(C) >= 4:
        _, s, vt = np.linalg.svd(C - C.mean(0)); n = vt[2]
        if s[1] > 3 * s[2] and abs(n @ up) > 0.5: up = n if n @ up > 0 else -n  # a flat orbit's normal is a better "up"
    y = np.array([0, 1.0, 0]); v = np.cross(up, y); c = float(up @ y)
    R = np.eye(3) if np.linalg.norm(v) < 1e-9 else (np.eye(3) + _skew(v) + _skew(v) @ _skew(v) / (1 + c) if c > -1 + 1e-9 else np.diag([1.0, -1, -1]))
    P = points @ R.T; centre = np.median(P, 0); r = np.percentile(np.linalg.norm(P - centre, axis=1), 95) or 1
    M = np.eye(4); M[:3, :3] = R * (5.0 / r); M[:3, 3] = -centre * (5.0 / r)
    return M

def _skew(v):
    import numpy as np
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def _apply(M, P): return P @ M[:3, :3].T + M[:3, 3]

# ---------- running COLMAP ----------
class Failed(RuntimeError): pass

def colmap(args, log, cwd=None):
    """Run one COLMAP command; its output is appended to the job log. Raises Failed on a non-zero exit."""
    with open(log, 'a', encoding='utf-8', errors='replace') as lf:
        lf.write(f'\n$ colmap {" ".join(map(str, args))}\n'); lf.flush(); t = time.time()
        r = subprocess.run([colmap_exe(), *map(str, args)], stdout=lf, stderr=subprocess.STDOUT, cwd=cwd,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        lf.write(f'[exit {r.returncode} after {time.time() - t:.0f}s]\n')
    if r.returncode != 0: raise Failed(f'colmap {args[0]} failed (exit {r.returncode}); see the log')

def _link_or_copy(src, dst):
    """Hard-link when possible (same drive): the photo appears in the workspace without using disk space again."""
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

def prepare_images(work, photos, videos, log):
    """images/photos/* and images/<video>/* - one camera per folder, so photos and each video get their own intrinsics."""
    from . import frames
    img = work / 'images'; n = 0
    if photos:
        (img / 'photos').mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(photos[:MAX_IMAGES]): _link_or_copy(p, img / 'photos' / f'{i:05d}{Path(p).suffix.lower()}'); n += 1
    for v in videos:
        if n >= MAX_IMAGES: break
        out = frames.extract(v, img / Path(v).stem, stride=VIDEO_STRIDE, max_frames=MAX_IMAGES - n); n += len(out)
        with open(log, 'a') as lf: lf.write(f'{Path(v).name}: {len(out)} frames (every {VIDEO_STRIDE}th)\n')
    return img, n

def analyse(model_dir, log):
    """Registered images, points, reprojection error from `colmap model_analyzer` (logged, then parsed)."""
    before = Path(log).read_text(encoding='utf-8', errors='replace') if Path(log).is_file() else ''
    colmap(['model_analyzer', '--path', model_dir], log)
    text = Path(log).read_text(encoding='utf-8', errors='replace')[len(before):]
    g = lambda k: (m := re.search(rf'{k}:\s*([\d.]+)', text)) and float(m.group(1))
    return {'registered_images': int(g('Registered images') or g('Images') or 0), 'sparse_points': int(g('Points') or 0),
            'mean_reprojection_error_px': g('Mean reprojection error'), 'mean_track_length': g('Mean track length')}

def cameras_of(model_dir, work, log):
    """Camera centres/directions of a binary COLMAP model (converted to TXT in a scratch folder that is removed again)."""
    txt = work / 'sparse_txt'; txt.mkdir(parents=True, exist_ok=True)
    try:
        write_text_model(model_dir, txt)  # read the binary model directly; no colmap.exe call needed
        return read_cameras_txt(txt / 'images.txt')
    finally:
        shutil.rmtree(txt, ignore_errors=True)  # tens of MB of 2D observations, not needed after this

def cleanup(work):
    """Drop what is only needed while building: the feature database and dense depth maps / undistorted copies.
    Kept: images (hard links, needed to densify later), the sparse model, outputs and the log."""
    work = Path(work)
    for p in (work / 'database.db', work / 'database.db-shm', work / 'database.db-wal'): p.unlink(missing_ok=True)
    for d in (work / 'sparse_txt', work / 'dense' / 'stereo', work / 'dense' / 'images'): shutil.rmtree(d, ignore_errors=True)

def sparse_stage(work, photos, videos, progress, log):
    progress('preparing', 2, 'Collecting photos and video frames')
    img, n = prepare_images(work, photos, videos, log)
    if n < 3: raise Failed(f'Need at least 3 overlapping images, found {n}')
    db = work / 'database.db'
    progress('features', 8, f'Finding features in {n} images')
    colmap(['feature_extractor', '--database_path', db, '--image_path', img, '--ImageReader.single_camera_per_folder', 1,
            '--FeatureExtraction.use_gpu', 1, '--FeatureExtraction.max_image_size', 3200], log)
    progress('matching', 22, 'Matching overlapping images')
    if n <= EXHAUSTIVE_UP_TO: colmap(['exhaustive_matcher', '--database_path', db, '--FeatureMatching.use_gpu', 1], log)
    else: colmap(['sequential_matcher', '--database_path', db, '--FeatureMatching.use_gpu', 1, '--SequentialMatching.overlap', 15,
                  '--SequentialMatching.quadratic_overlap', 1], log)
    progress('mapping', 40, 'Solving camera positions and sparse points')
    (work / 'sparse').mkdir(exist_ok=True)
    colmap(['mapper', '--database_path', db, '--image_path', img, '--output_path', work / 'sparse'], log)
    models = [d for d in (work / 'sparse').iterdir() if (d / 'images.bin').is_file() or (d / 'images.txt').is_file()]
    if not models: raise Failed('COLMAP could not connect the images into one model: they need more overlap')
    sizes = {d: analyse(d, log) for d in models}  # the mapper can split poorly connected sets into several models; keep the largest
    best = max(models, key=lambda d: sizes[d]['registered_images'])
    colmap(['model_converter', '--input_path', best, '--output_path', work / 'sparse.ply', '--output_type', 'PLY'], log)
    stats = {**sizes[best], 'images_total': n, 'models_found': len(models), 'sparse_model': best.relative_to(work).as_posix()}
    return stats, img, best, cameras_of(best, work, log), read_ply(work / 'sparse.ply')

def dense_stage(work, img, best, progress, log, base=50):
    """GPU multi-view stereo on an existing sparse model -> dense cloud + Poisson mesh (simplified for the browser)."""
    d = work / 'dense'; span = 100 - base - 6; at = lambda f: int(base + span * f)
    progress('undistorting', at(0), 'Preparing images for dense matching')
    colmap(['image_undistorter', '--image_path', img, '--input_path', best, '--output_path', d, '--output_type', 'COLMAP', '--max_image_size', DENSE_SIZE], log)
    progress('dense_stereo', at(.08), 'Computing depth maps on the GPU (the slow step)')
    colmap(['patch_match_stereo', '--workspace_path', d, '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', 'true'], log)
    progress('fusion', at(.8), 'Fusing depth maps into a dense point cloud')
    colmap(['stereo_fusion', '--workspace_path', d, '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', d / 'fused.ply'], log)
    progress('meshing', at(.9), 'Building a surface mesh')
    colmap(['poisson_mesher', '--input_path', d / 'fused.ply', '--output_path', d / 'mesh.ply'], log)
    cloud, mesh = read_ply(d / 'fused.ply'), read_ply(d / 'mesh.ply')
    if mesh['faces'] is not None and len(mesh['faces']) > WEB_FACES:
        colmap(['mesh_simplifier', '--input_path', d / 'mesh.ply', '--output_path', d / 'mesh_web.ply',
                '--MeshSimplification.target_face_ratio', round(WEB_FACES / len(mesh['faces']), 4)], log)
        mesh = read_ply(d / 'mesh_web.ply')
    shutil.rmtree(d / 'stereo', ignore_errors=True); shutil.rmtree(d / 'images', ignore_errors=True)  # several GB, done with
    extra = {'dense': True, 'dense_points': int(len(cloud['points'])), 'mesh_faces': int(len(mesh['faces'])) if mesh['faces'] is not None else 0}
    return cloud, mesh, extra, {'dense_ply': 'dense/fused.ply', 'mesh_ply': 'dense/mesh.ply'}

def export_web(work, M, cloud, mesh, cams):
    """Levelled, centred, thinned copies for the viewer, plus the camera path."""
    import numpy as np
    web = work / 'web'; web.mkdir(exist_ok=True); P, Cc = cloud['points'], cloud['colors']
    if len(P) > WEB_POINTS:
        keep = np.random.default_rng(0).choice(len(P), WEB_POINTS, replace=False); P, Cc = P[keep], (Cc[keep] if Cc is not None else None)
    write_ply(web / 'points.ply', _apply(M, P).astype(np.float32), Cc)
    if mesh is not None: write_ply(web / 'mesh.ply', _apply(M, mesh['points']).astype(np.float32), mesh['colors'], mesh['faces'])
    (web / 'cameras.json').write_text(json.dumps([{'name': c['name'], 'center': _apply(M, np.array([c['center']]))[0].round(4).tolist(),
                                                    'dir': (np.array(c['dir']) @ M[:3, :3].T / np.linalg.norm(M[:3, 0])).round(4).tolist()} for c in cams]))
    return {'web_points': 'web/points.ply', 'web_cameras': 'web/cameras.json', **({'web_mesh': 'web/mesh.ply'} if mesh is not None else {})}, int(min(len(cloud['points']), WEB_POINTS))

def run(work, photos, videos, dense, progress):
    """Full pipeline in `work`. progress(stage, percent, detail). Returns a result dict (files are relative to work)."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True); log = work / 'colmap.log'; t0 = time.time()
    try:
        stats, img, best, cams, sparse = sparse_stage(work, photos, videos, progress, log)
        files = {'sparse_ply': 'sparse.ply', 'log': 'colmap.log'}; cloud, mesh = sparse, None; result = {**stats, 'dense': False}
        if dense:
            cloud, mesh, extra, f = dense_stage(work, img, best, progress, log); result.update(extra); files.update(f)
        progress('exporting', 96, 'Preparing the model for the viewer')
        M = normalising_transform(sparse['points'], cams)
        web_files, n_web = export_web(work, M, cloud, mesh, cams)
        return {**result, 'files': {**files, **web_files}, 'web_points': n_web, 'transform': M.round(8).tolist(), 'seconds': round(time.time() - t0)}
    finally:
        cleanup(work)

def densify(src, work, progress, src_quality):
    """Dense point cloud + mesh from an existing sparse build in `src`, without redoing features, matching and mapping."""
    import numpy as np
    src, work = Path(src), Path(work); work.mkdir(parents=True, exist_ok=True); log = work / 'colmap.log'; t0 = time.time()
    try:
        best = src / src_quality['sparse_model'] if src_quality.get('sparse_model') else \
            max((d for d in (src / 'sparse').iterdir() if (d / 'images.bin').is_file()), key=lambda d: (d / 'images.bin').stat().st_size, default=None)
        img = src / 'images'
        if best is None or not any((best / f).is_file() for f in ('images.bin', 'images.txt')): raise Failed('The sparse model files of that build are missing; build a new model instead')
        if not img.is_dir() or not any(img.rglob('*.*')): raise Failed('The images of that build were removed; build a new model instead')
        progress('preparing', 3, 'Reusing the sparse model and its camera positions')
        cams = cameras_of(best, work, log); sparse = read_ply(src / 'sparse.ply'); shutil.copy2(src / 'sparse.ply', work / 'sparse.ply')
        M = np.array(src_quality['transform']) if src_quality.get('transform') else normalising_transform(sparse['points'], cams)  # same pose as the sparse view
        cloud, mesh, extra, f = dense_stage(work, img, best, progress, log, base=5)
        progress('exporting', 96, 'Preparing the model for the viewer')
        web_files, n_web = export_web(work, M, cloud, mesh, cams)
        keep = {k: src_quality.get(k) for k in ('registered_images', 'images_total', 'sparse_points', 'mean_reprojection_error_px', 'mean_track_length', 'models_found')}
        return {**keep, **extra, 'source_model_dir': src.name, 'files': {'sparse_ply': 'sparse.ply', 'log': 'colmap.log', **f, **web_files},
                'web_points': n_web, 'transform': M.round(8).tolist(), 'seconds': round(time.time() - t0)}
    finally:
        cleanup(work)

# ---------- reading COLMAP binary models without colmap.exe ----------
# COLMAP 3.12+/4.x still write cameras.bin/images.bin/points3D.bin in the original layout (rigs/frames are extra files),
# see https://colmap.github.io/format.html. Converting to the TXT layout here keeps model reading independent of the binary.
CAMERA_MODELS = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4), 3: ('RADIAL', 5), 4: ('OPENCV', 8), 5: ('OPENCV_FISHEYE', 8),
                 6: ('FULL_OPENCV', 12), 7: ('FOV', 5), 8: ('SIMPLE_RADIAL_FISHEYE', 4), 9: ('RADIAL_FISHEYE', 5), 10: ('THIN_PRISM_FISHEYE', 12)}

def write_text_model(model_dir, out_dir):
    """cameras.txt / images.txt / points3D.txt from a binary COLMAP model (same text layout as `colmap model_converter`)."""
    import struct, numpy as np
    model_dir, out_dir = Path(model_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / 'images.txt').is_file() and not (model_dir / 'images.bin').is_file():  # already text
        for n in ('cameras.txt', 'images.txt', 'points3D.txt'): shutil.copy2(model_dir / n, out_dir / n)
        return out_dir
    with open(model_dir / 'cameras.bin', 'rb') as f, open(out_dir / 'cameras.txt', 'w') as o:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            cid, mid, w, h = struct.unpack('<iiQQ', f.read(24))
            if mid not in CAMERA_MODELS: raise Failed(f'Unsupported COLMAP camera model id {mid}')
            name, n = CAMERA_MODELS[mid]; params = struct.unpack(f'<{n}d', f.read(8 * n))
            o.write(f'{cid} {name} {w} {h} ' + ' '.join(repr(p) for p in params) + '\n')
    obs_dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
    with open(model_dir / 'images.bin', 'rb') as f, open(out_dir / 'images.txt', 'w') as o:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack('<i7di', f.read(64))
            name = b''
            while (ch := f.read(1)) not in (b'\x00', b''): name += ch
            n = struct.unpack('<Q', f.read(8))[0]; pts = np.frombuffer(f.read(24 * n), obs_dt, n)
            o.write(f'{iid} {qw!r} {qx!r} {qy!r} {qz!r} {tx!r} {ty!r} {tz!r} {cid} {name.decode()}\n')
            o.write(' '.join(f'{x:.6f} {y:.6f} {i}' for x, y, i in zip(pts['x'], pts['y'], pts['id'])) + '\n')
    with open(model_dir / 'points3D.bin', 'rb') as f, open(out_dir / 'points3D.txt', 'w') as o:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            pid, x, y, z, r, g, b, err, tl = struct.unpack('<Q3d3BdQ', f.read(51))
            f.seek(8 * tl, 1)  # track (image id, point2D index) pairs - not needed here
            o.write(f'{pid} {x!r} {y!r} {z!r} {r} {g} {b} {err!r}\n')
    return out_dir
