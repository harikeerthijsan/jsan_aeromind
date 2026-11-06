"""Uploaded point clouds (LiDAR or photogrammetry exports) as an inspection's 3D model.

Formats: LAS / LAZ (laspy + lazrs, read in chunks), E57 (pye57: every scan, in the file's global frame, metres),
PLY, and text clouds (XYZ / PTS / TXT / CSV: x y z [r g b]).
Sidecars are kept with the cloud: .lax (LAStools spatial index, not needed here) and .prj (coordinate system for
clouds whose format cannot carry one).

The cloud replaces COLMAP's dense reconstruction (the slow part). For the 3D equipment boxes the photos still have
to be placed relative to it, so with photos a COLMAP *sparse* camera solve runs (minutes) and is aligned to the cloud:
  1. the photos' GPS / RTK positions, converted into the cloud's coordinate system, give scale, rotation and position
     (similarity fit of the solved camera centres);
  2. the vertical datum difference (GPS ellipsoidal heights vs. the cloud's height system, often 30-50 m) is found by
     testing which height shift makes the most sparse points touch the cloud;
  3. ICP (trimmed, rigid) refines the fit against the cloud's surfaces.
The cloud is then written in the camera model's frame, so 2D->3D registration works exactly as for COLMAP models.
"""
from __future__ import annotations

import math
from pathlib import Path

CLOUD_EXT = {'.las', '.laz', '.e57', '.ply', '.xyz', '.pts', '.txt', '.csv'}
SIDECAR_EXT = {'.lax', '.prj'}
MODEL_POINTS = 4_000_000      # cloud points kept for registration (written as dense/fused.ply)
ROI_MARGIN_M = 60.0           # around the drone's flight: towers sit in large survey tiles
CHUNK = 2_000_000

class CloudError(ValueError): pass

# ---------- coordinate systems ----------
def _crs_parts(crs):
    """(horizontal CRS, metres per horizontal unit, metres per vertical unit)."""
    if crs is None: return None, 1.0, 1.0
    subs = crs.sub_crs_list or [crs]; horiz = subs[0]
    fh = horiz.axis_info[0].unit_conversion_factor if horiz.axis_info else 1.0
    fv = (subs[1].axis_info[0].unit_conversion_factor if len(subs) > 1 and subs[1].axis_info else
          crs.axis_info[2].unit_conversion_factor if len(crs.axis_info) > 2 else fh)
    return horiz, float(fh or 1.0), float(fv or 1.0)

def _source(p):
    """(file name, source): a cloud is a path, or a (name, source) pair where source is an open seekable stream (LAS/LAZ
    read straight from the bucket) or, for a .prj sidecar, its bytes."""
    return (p[0], p[1]) if isinstance(p, tuple) else (Path(p).name, Path(p))

def _las(src):
    import laspy
    if isinstance(src, Path): return laspy.open(src)
    src.seek(0); return laspy.open(src, closefd=False)

def _prj_crs(paths):
    from pyproj import CRS
    for name, src in map(_source, paths):
        if Path(name).suffix.lower() == '.prj':
            try: return CRS.from_wkt(src.decode('utf-8', 'replace') if isinstance(src, bytes) else Path(src).read_text(encoding='utf-8', errors='replace'))
            except Exception: pass
    return None

def _e57_crs(path):
    """E57 may name its coordinate system in root/coordinateMetadata (WKT, EPSG code or similar)."""
    try:
        import pye57
        from pyproj import CRS
        root = pye57.E57(str(path)).root
        return CRS.from_user_input(root['coordinateMetadata'].value()) if root.isDefined('coordinateMetadata') and root['coordinateMetadata'].value().strip() else None
    except Exception: return None

def peek_crs(paths):
    """The coordinate system of the first cloud that has one (LAS header or E57 metadata, else a .prj sidecar)."""
    for name, p in map(_source, paths):
        if Path(name).suffix.lower() == '.e57' and (c := _e57_crs(p)) is not None: return c
        if Path(name).suffix.lower() in ('.las', '.laz'):
            with _las(p) as r:
                try:
                    if (c := r.header.parse_crs()) is not None: return c
                except Exception: pass
    return _prj_crs(paths)

def gps_to_cloud(positions, crs):
    """[(lat, lon, alt m)] -> (N,3) metres in the cloud's horizontal CRS (height stays GPS height; offset solved later)."""
    import numpy as np
    from pyproj import Transformer
    horiz, fh, _ = _crs_parts(crs)
    t = Transformer.from_crs('EPSG:4326', horiz, always_xy=True)
    lon, lat, alt = (np.array([p[k] for p in positions], float) for k in ('lon', 'lat', 'alt'))
    x, y = t.transform(lon, lat)
    return np.c_[np.asarray(x) * fh, np.asarray(y) * fh, alt]

# ---------- reading ----------
def _colours(chunk, names):
    import numpy as np
    if {'red', 'green', 'blue'} <= names:
        rgb = np.c_[chunk.red, chunk.green, chunk.blue].astype(np.float32)
        if rgb.max() > 0: return (rgb / (257.0 if rgb.max() > 255 else 1.0)).clip(0, 255).astype(np.uint8)
    if 'intensity' in names: return np.asarray(chunk.intensity, np.float32)  # greyscale later, once its range is known
    return None

FOCUS_POINTS = 6_000_000     # points kept around the structure itself (focus), on top of max_points for the surroundings

class _Pool:
    """Points kept from a stream, thinned evenly (progressively, memory stays bounded) to about `budget`."""
    def __init__(self, budget, rng): self.budget, self.rng, self.p, self.parts, self.cols = budget, rng, 1.0, [], []
    def add(self, P, C):
        if self.p < 1.0:
            m = self.rng.random(len(P)) < self.p; P, C = P[m], (C[m] if C is not None else None)
        self.parts.append(P); self.cols.append(C)
        while sum(len(x) for x in self.parts) > 2 * self.budget:
            for i in range(len(self.parts)):
                m = self.rng.random(len(self.parts[i])) < .5; self.parts[i] = self.parts[i][m]; self.cols[i] = self.cols[i][m] if self.cols[i] is not None else None
            self.p *= .5

def read_clouds(paths, max_points=MODEL_POINTS, roi=None, progress=None, focus=None):
    """Points in metres (float64, (N,3)) from all clouds, thinned evenly to about max_points; roi = (xmin, xmax, ymin,
    ymax) in metres of the CRS keeps only that area. focus = (x, y, radius): points that close to the structure are
    kept at up to FOCUS_POINTS on their own, so the structure keeps its full detail while the surroundings are thinned.
    Returns {'points', 'colors' (uint8 or None), 'focus' (bool per point), 'crs', 'total', 'kept', 'units'}."""
    import numpy as np
    rng = np.random.default_rng(0); total = 0
    crs = None; units = 'metre'; intensity = False
    rest, near = _Pool(max_points, rng), _Pool(FOCUS_POINTS, rng)
    def add(P, C):
        if roi is not None:
            m = (P[:, 0] >= roi[0]) & (P[:, 0] <= roi[1]) & (P[:, 1] >= roi[2]) & (P[:, 1] <= roi[3]); P, C = P[m], (C[m] if C is not None else None)
        if focus is None: rest.add(P, C); return
        m = (P[:, 0] - focus[0]) ** 2 + (P[:, 1] - focus[1]) ** 2 <= focus[2] ** 2
        near.add(P[m], C[m] if C is not None else None); rest.add(P[~m], C[~m] if C is not None else None)
    clouds = [s for s in map(_source, paths) if Path(s[0]).suffix.lower() in CLOUD_EXT]
    if not clouds: raise CloudError('no point cloud file (.las, .laz, .ply, .xyz, .pts, .txt, .csv)')
    prj = _prj_crs(paths)
    for n, (name, p) in enumerate(clouds, 1):
        ext = Path(name).suffix.lower()
        if progress: progress(f'Reading {name} ({n}/{len(clouds)})')
        if ext in ('.las', '.laz'):
            with _las(p) as r:
                try: c = r.header.parse_crs()
                except Exception: c = None
                crs = crs or c or prj; _, fh, fv = _crs_parts(c or prj)
                names = set(r.header.point_format.dimension_names)
                for chunk in r.chunk_iterator(CHUNK):
                    P = np.c_[np.asarray(chunk.x) * fh, np.asarray(chunk.y) * fh, np.asarray(chunk.z) * fv]; total += len(P)
                    C = _colours(chunk, names); intensity = intensity or (C is not None and C.ndim == 1)
                    add(P, C if C is None or C.ndim == 2 else C)
                    if progress: progress(f'Reading {name}: {total / 1e6:.0f} of {r.header.point_count / 1e6:.0f} M points')
        elif ext == '.e57':  # E57 is always metres; scans are placed in the file's global frame (transform=True)
            import pye57
            f = pye57.E57(str(p)); crs = crs or _e57_crs(p) or prj
            for i in range(f.scan_count):
                if progress and f.scan_count > 1: progress(f'Reading {name}: scan {i + 1}/{f.scan_count}')
                d = f.read_scan(i, intensity=True, colors=True, ignore_missing_fields=True)
                P = np.c_[d['cartesianX'], d['cartesianY'], d['cartesianZ']].astype(np.float64); total += len(P)
                C = None
                if 'colorRed' in d:
                    rgb = np.c_[d['colorRed'], d['colorGreen'], d['colorBlue']].astype(np.float32)
                    if rgb.max() > 0: C = (rgb / (257.0 if rgb.max() > 255 else 1.0)).clip(0, 255).astype(np.uint8)
                if C is None and 'intensity' in d: C = np.asarray(d['intensity'], np.float32); intensity = True
                for s0 in range(0, len(P), CHUNK): add(P[s0:s0 + CHUNK], None if C is None else C[s0:s0 + CHUNK])
        elif ext == '.ply':
            from . import reconstruct
            d = reconstruct.read_ply(p); P = np.asarray(d['points'], float); total += len(P); crs = crs or prj
            _, fh, fv = _crs_parts(prj); add(P * [fh, fh, fv], d['colors'])
        else:  # text: x y z [r g b] per line; header / comment lines are skipped
            arr = np.genfromtxt(p, delimiter=',' if ext == '.csv' else None, comments='#', invalid_raise=False, usecols=None)
            arr = np.atleast_2d(arr); arr = arr[np.isfinite(arr[:, :3]).all(1)] if arr.shape[1] >= 3 else np.zeros((0, 3))
            if ext == '.pts' and len(arr) and arr.shape[1] == 1: arr = arr[1:]
            _, fh, fv = _crs_parts(prj); crs = crs or prj; total += len(arr)
            C = arr[:, -3:].clip(0, 255).astype(np.uint8) if arr.shape[1] >= 6 else None
            add(arr[:, :3] * [fh, fh, fv], C)
    def finish(pool):
        P = np.concatenate(pool.parts) if pool.parts else np.zeros((0, 3))
        C = np.concatenate([c if c.ndim == 2 else c[:, None] for c in pool.cols]) if pool.cols and all(c is not None for c in pool.cols) else None
        if len(P) > pool.budget:
            k = rng.choice(len(P), pool.budget, replace=False); P = P[k]; C = C[k] if C is not None else None
        return P, C
    (Pn, Cn), (Pr, Cr) = finish(near), finish(rest)
    P = np.concatenate([Pn, Pr]); in_focus = np.r_[np.ones(len(Pn), bool), np.zeros(len(Pr), bool)]
    if len(P) == 0: raise CloudError('the point cloud has no points' + (' near the drone photos (it may cover a different area)' if roi else ''))
    filled = [(p, c) for p, c in ((Pn, Cn), (Pr, Cr)) if len(p)]
    C = np.concatenate([c for _, c in filled]) if all(c is not None for _, c in filled) else None
    if C is not None and C.shape[1] == 1:  # intensity -> grey
        lo, hi = np.percentile(C[:, 0], [2, 98]); g = ((C[:, 0] - lo) / max(hi - lo, 1e-9) * 200 + 40).clip(0, 255).astype(np.uint8); C = np.c_[g, g, g]
    if C is None: C = height_colours(P[:, 2])
    if crs is not None:
        h, _, _ = _crs_parts(crs); units = h.axis_info[0].unit_name if h and h.axis_info else 'metre'
    return {'points': P, 'colors': C, 'focus': in_focus, 'crs': crs, 'total': int(total), 'kept': int(len(P)), 'units': units}

GREEN = 25  # excess green (2G - R - B) above this: vegetation (steel, antennas and cables are grey, white or black)

def split_structure(points, colors, focus):
    """Which points are the structure (tower) and which its surroundings (trees, grass, buildings), for the viewer: trees
    around a tower hide its lower half from almost every angle. Surveys often leave points unclassified, so this uses
    colour (vegetation is green) and the structure's own reach: the horizontal distance from its axis that the
    equipment at the top extends to. points: local metres (z up); focus: points near the structure.
    Returns (bool mask of structure points, info)."""
    import numpy as np
    P = points[focus]; z0, top = np.percentile(P[:, 2], 1), P[:, 2].max()
    axis = np.median(P[P[:, 2] >= np.percentile(P[:, 2], 97)][:, :2], 0)  # the middle of the highest points
    green = (2 * colors[:, 1].astype(int) - colors[:, 0] - colors[:, 2]) > GREEN if colors is not None else np.zeros(len(points), bool)
    r = np.linalg.norm(points[:, :2] - axis, axis=1)
    upper = focus & ~green & (points[:, 2] > z0 + .6 * (top - z0))
    reach = float(np.clip(np.percentile(r[upper], 95) + .5, 2.0, 8.0)) if upper.sum() > 100 else 4.0
    structure = focus & ~green & (r <= reach)
    return structure, {'axis': [round(float(v), 3) for v in axis], 'reach_m': round(reach, 2), 'structure_points': int(structure.sum())}

def clear_view(points, structure, info, M, distance=None):
    """Where the viewer's camera should start: the side of the structure with the fewest other points (trees)
    between it and the camera, far enough out to see all of it and at about its middle height, so the lines of sight
    to its foot pass over the open side rather than down through the trees next to it. Returns {'position', 'target'}
    in viewer coordinates."""
    import numpy as np
    from scipy.spatial import cKDTree
    from . import reconstruct
    S, O = points[structure], points[~structure]
    axis = np.asarray(info['axis']); z0, top = np.percentile(S[:, 2], 1), S[:, 2].max(); mid = z0 + .45 * (top - z0)
    d = distance or max(30.0, 2.3 * (top - z0))
    tree = cKDTree(O[np.random.default_rng(0).choice(len(O), min(len(O), 400_000), replace=False)]) if len(O) else None
    best, score = 0.0, None
    for a in np.radians(np.arange(0, 360, 30)):
        u = np.array([np.cos(a), np.sin(a)]); hits = 0
        for h in np.linspace(z0 + 1, top, 8):  # lines of sight from the camera to the structure, low to high
            t = np.linspace(1.5, d, 40)[:, None]; L = np.c_[axis + t * u, np.full(len(t), h)]
            if tree is not None: hits += int((tree.query(L, distance_upper_bound=1.0)[0] < 1.0).sum())
        if score is None or hits < score: best, score = a, hits
    cam = np.r_[axis + d * np.array([np.cos(best), np.sin(best)]), z0 + .6 * (top - z0)]
    return {'position': reconstruct._apply(M, cam[None])[0].round(3).tolist(), 'target': reconstruct._apply(M, np.r_[axis, mid][None])[0].round(3).tolist()}

def height_colours(z):
    """Blue -> green -> yellow by height, for clouds without colour or intensity."""
    import numpy as np
    lo, hi = np.percentile(z, [2, 98]); t = ((z - lo) / max(hi - lo, 1e-9)).clip(0, 1)
    stops = np.array([[38, 70, 160], [40, 170, 150], [140, 210, 80], [250, 225, 70]], float)
    i = np.minimum((t * 3).astype(int), 2); f = (t * 3 - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)

# ---------- alignment ----------
def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t (least squares)."""
    import numpy as np
    ms, md = src.mean(0), dst.mean(0); S, Dd = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(Dd.T @ S / len(S)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = float(np.trace(np.diag(sig) @ D) / ((S ** 2).sum() / len(S)))
    return s, R, md - s * R @ ms

def align(cam_model, cam_cloud, sparse_model, cloud, progress=None):
    """Camera model frame -> cloud frame (metres, local). cam_*: matched (N,3) camera centres; returns
    {'s', 'R', 't', 'gps_residual_m', 'vertical_shift_m', 'icp_rms_m', 'inliers_pct'}."""
    import numpy as np
    from scipy.spatial import cKDTree
    s, R, t = umeyama(cam_model, cam_cloud)
    gps_res = float(np.median(np.linalg.norm(cam_cloud - (s * cam_model @ R.T + t), axis=1)))
    rng = np.random.default_rng(1)
    C = cloud if len(cloud) <= 3_000_000 else cloud[rng.choice(len(cloud), 3_000_000, replace=False)]
    tree = cKDTree(C)
    Sm = sparse_model if len(sparse_model) <= 20000 else sparse_model[rng.choice(len(sparse_model), 20000, replace=False)]
    S0 = s * Sm @ R.T + t
    # vertical datum: GPS heights are ellipsoidal or take-off relative, the cloud's are usually orthometric
    if progress: progress('Matching heights between the photos and the point cloud')
    probe = S0 if len(S0) <= 4000 else S0[rng.choice(len(S0), 4000, replace=False)]
    def hits(dz, r): return int(np.isfinite(tree.query(probe + [0, 0, dz], distance_upper_bound=r)[0]).sum())
    coarse = np.arange(-120, 120.01, .5); scores = [hits(dz, .6) for dz in coarse]; best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - .5, best + .51, .05); best = float(fine[int(np.argmax([hits(dz, .2) for dz in fine]))])
    t = t + [0, 0, best]
    # trimmed rigid ICP (scale stays the GPS one), shrinking the match distance
    if progress: progress('Fitting the camera model to the point cloud (ICP)')
    for thr in (1.0, .6, .4, .25, .15, .1):
        for _ in range(6):
            X = s * Sm @ R.T + t; d, i = tree.query(X, distance_upper_bound=thr); ok = np.isfinite(d)
            if ok.sum() < 30: break
            _, dR, dt = _kabsch(X[ok], C[i[ok]]); R = dR @ R; t = dR @ t + dt
    X = s * Sm @ R.T + t; d, _ = tree.query(X, distance_upper_bound=.25); ok = np.isfinite(d)
    shift = float(np.median((s * cam_model @ R.T + t)[:, 2] - cam_cloud[:, 2]))  # final datum difference: cloud heights - GPS heights
    return {'s': float(s), 'R': R, 't': t, 'gps_residual_m': round(gps_res, 3), 'vertical_shift_m': round(shift, 2),
            'icp_rms_m': round(float(np.sqrt((d[ok] ** 2).mean())), 3) if ok.any() else None, 'inliers_pct': round(float(ok.mean() * 100), 1)}

def _kabsch(src, dst):
    import numpy as np
    ms, md = src.mean(0), dst.mean(0); U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    return 1.0, R, md - R @ ms

def viewer_transform(points):
    """Cloud-only model (no photos): metres, Z up -> viewer Y up, centred on the structure, ~10 viewer units across."""
    import numpy as np
    c = np.median(points, 0); lo, hi = np.percentile(points, [1, 99], axis=0)
    s = 10.0 / max(float(np.max(hi - lo)), 1e-6)
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # (x, y, z) -> (x, z, -y)
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = -s * R @ c; M[1, 3] = -5.0 - s * lo[2]  # lowest points (ground) at the viewer's grid, y = -5
    return M, s
