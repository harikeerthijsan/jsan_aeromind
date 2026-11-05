"""Uploaded point clouds (LiDAR or photogrammetry exports) as an inspection's 3D model.

Formats: LAS / LAZ (laspy + lazrs, read in chunks), PLY, and text clouds (XYZ / PTS / TXT / CSV: x y z [r g b]).
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

CLOUD_EXT = {'.las', '.laz', '.ply', '.xyz', '.pts', '.txt', '.csv'}
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

def _prj_crs(paths):
    from pyproj import CRS
    for p in paths:
        if Path(p).suffix.lower() == '.prj':
            try: return CRS.from_wkt(Path(p).read_text(encoding='utf-8', errors='replace'))
            except Exception: pass
    return None

def peek_crs(paths):
    """The coordinate system of the first cloud that has one (LAS header, else a .prj sidecar)."""
    import laspy
    for p in paths:
        if Path(p).suffix.lower() in ('.las', '.laz'):
            with laspy.open(p) as r:
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

def read_clouds(paths, max_points=MODEL_POINTS, roi=None, progress=None):
    """Points in metres (float64, (N,3)) from all clouds, thinned evenly to about max_points; roi = (xmin, xmax, ymin,
    ymax) in metres of the CRS keeps only that area. Returns {'points', 'colors' (uint8 or None), 'crs', 'total',
    'kept', 'units'}."""
    import numpy as np
    rng = np.random.default_rng(0); keep_p = 1.0; parts, cols, total = [], [], 0
    crs = None; units = 'metre'; intensity = False
    def add(P, C):
        nonlocal keep_p
        if roi is not None:
            m = (P[:, 0] >= roi[0]) & (P[:, 0] <= roi[1]) & (P[:, 1] >= roi[2]) & (P[:, 1] <= roi[3]); P, C = P[m], (C[m] if C is not None else None)
        if keep_p < 1.0:
            m = rng.random(len(P)) < keep_p; P, C = P[m], (C[m] if C is not None else None)
        parts.append(P); cols.append(C)
        while sum(len(x) for x in parts) > 2 * max_points:  # progressive thinning keeps memory bounded
            for i in range(len(parts)):
                m = rng.random(len(parts[i])) < .5; parts[i] = parts[i][m]; cols[i] = cols[i][m] if cols[i] is not None else None
            keep_p *= .5
    clouds = [Path(p) for p in paths if Path(p).suffix.lower() in CLOUD_EXT]
    if not clouds: raise CloudError('no point cloud file (.las, .laz, .ply, .xyz, .pts, .txt, .csv)')
    prj = _prj_crs(paths)
    for n, p in enumerate(clouds, 1):
        ext = p.suffix.lower()
        if progress: progress(f'Reading {p.name} ({n}/{len(clouds)})')
        if ext in ('.las', '.laz'):
            import laspy
            with laspy.open(p) as r:
                try: c = r.header.parse_crs()
                except Exception: c = None
                crs = crs or c or prj; _, fh, fv = _crs_parts(c or prj)
                names = set(r.header.point_format.dimension_names)
                for chunk in r.chunk_iterator(CHUNK):
                    P = np.c_[np.asarray(chunk.x) * fh, np.asarray(chunk.y) * fh, np.asarray(chunk.z) * fv]; total += len(P)
                    C = _colours(chunk, names); intensity = intensity or (C is not None and C.ndim == 1)
                    add(P, C if C is None or C.ndim == 2 else C)
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
    P = np.concatenate(parts) if parts else np.zeros((0, 3))
    if len(P) == 0: raise CloudError('the point cloud has no points' + (' near the drone photos (it may cover a different area)' if roi else ''))
    has_c = all(c is not None for c in cols)
    C = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols]) if has_c else None
    if len(P) > max_points:
        k = rng.choice(len(P), max_points, replace=False); P = P[k]; C = C[k] if C is not None else None
    if C is not None and C.shape[1] == 1:  # intensity -> grey
        lo, hi = np.percentile(C[:, 0], [2, 98]); g = ((C[:, 0] - lo) / max(hi - lo, 1e-9) * 200 + 40).clip(0, 255).astype(np.uint8); C = np.c_[g, g, g]
    if C is None: C = height_colours(P[:, 2])
    if crs is not None:
        h, _, _ = _crs_parts(crs); units = h.axis_info[0].unit_name if h and h.axis_info else 'metre'
    return {'points': P, 'colors': C, 'crs': crs, 'total': int(total), 'kept': int(len(P)), 'units': units}

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
