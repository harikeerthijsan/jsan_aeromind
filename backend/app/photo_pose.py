"""Camera poses straight from drone photo metadata, used instead of COLMAP when a point cloud is uploaded.

DJI enterprise drones write the camera position (RTK: ~1-2 cm), the gimbal's yaw/pitch/roll and the factory lens
calibration into each photo's XMP. That is everything needed to project the uploaded cloud into a photo (and a
photo's 2D detection boxes onto the cloud), so no photo has to be downloaded or matched: only the first ~64 KB of
each file, where the metadata sits, is read.

Frames: world = the cloud's CRS (metres, local origin subtracted); camera = COLMAP/OpenCV (x right, y down, z forward).
"""
from __future__ import annotations
import io, math, re

HEAD_BYTES = 64 * 1024            # DJI XMP ends ~35 KB into the file
_XMP_ATTR = re.compile(rb'drone-dji:(\w+)="([^"]*)"')
_XMP_ELEM = re.compile(rb'<drone-dji:(\w+)>([^<]*)</drone-dji:\1>')

def _num(x, k):
    try: return float(x[k]) if k in x and x[k] not in ('', None) else None
    except ValueError: return None

def _image_size(head: bytes):
    """(width, height) from the JPEG frame header or, if that lies beyond the bytes read, the EXIF pixel dimensions."""
    i, exif_size = 2, None
    while i + 9 < len(head) and head[i] == 0xFF:
        marker, length = head[i + 1], int.from_bytes(head[i + 2:i + 4], 'big')
        if marker in (0xC0, 0xC1, 0xC2): return int.from_bytes(head[i + 7:i + 9], 'big'), int.from_bytes(head[i + 5:i + 7], 'big')
        if marker == 0xE1 and head[i + 4:i + 10] == b'Exif\x00\x00' and exif_size is None:
            try:
                from PIL import Image
                ex = Image.Exif(); ex.load(head[i + 4:i + 2 + length]); ifd = ex.get_ifd(0x8769)
                if ifd.get(0xA002) and ifd.get(0xA003): exif_size = int(ifd[0xA002]), int(ifd[0xA003])
            except Exception: pass
        i += 2 + length
    return exif_size

def read(head: bytes) -> dict | None:
    """Pose and lens of one photo from its first bytes, or None without RTK/GPS position and gimbal angles.
    {'lat','lon','alt' (ellipsoidal m),'rel_alt','rtk','yaw','pitch','roll' (deg),'w','h','fx','fy','cx','cy','k' [k1,k2,p1,p2]}"""
    x = {k.decode(): v.decode(errors='replace') for k, v in _XMP_ATTR.findall(head)}
    x.update({k.decode(): v.decode(errors='replace') for k, v in _XMP_ELEM.findall(head)})
    lat, lon, alt = _num(x, 'GpsLatitude'), _num(x, 'GpsLongitude') or _num(x, 'GpsLongtitude'), _num(x, 'AbsoluteAltitude')
    yaw, pitch, roll = _num(x, 'GimbalYawDegree'), _num(x, 'GimbalPitchDegree'), _num(x, 'GimbalRollDegree') or 0.0
    size = _image_size(head)
    if None in (lat, lon, alt, yaw, pitch) or not size or (lat, lon) == (0.0, 0.0): return None
    w, h = size; out = {'lat': lat, 'lon': lon, 'alt': alt, 'rel_alt': _num(x, 'RelativeAltitude'), 'rtk': x.get('RtkFlag') == '50',
                        'yaw': yaw, 'pitch': pitch, 'roll': roll, 'w': w, 'h': h}
    dewarp = x.get('DewarpData', '').split(';')[-1].split(',')
    dewarped = x.get('DewarpFlag') == '1'  # the camera already removed the lens distortion
    if len(dewarp) >= 8 and not dewarped:  # fx, fy, cx and cy offsets from the image centre, k1, k2, p1, p2[, k3]
        v = [float(t) for t in dewarp]; out.update(fx=v[0], fy=v[1], cx=w / 2 + v[2], cy=h / 2 + v[3], k=v[4:8])
    else:
        f = _num(x, 'CalibratedFocalLength')
        if f is None:
            f35 = _focal35(head)
            if not f35: return None
            f = f35 / 43.2666 * math.hypot(w, h)  # 35 mm equivalent focal length -> pixels (diagonal)
        out.update(fx=f, fy=f, cx=w / 2, cy=h / 2, k=[0.0] * 4)
    return out

def _focal35(head):
    try:
        from PIL import Image
        i = head.find(b'Exif\x00\x00')
        if i < 0: return None
        ex = Image.Exif(); ex.load(head[i:]); return float(ex.get_ifd(0x8769).get(0xA405) or 0) or None
    except Exception: return None

def grid_transformer(crs):
    """WGS84 (ellipsoidal height, as RTK drones record it) -> the cloud's CRS incl. its vertical datum. Uses the
    national transformation grids (e.g. OS OSTN15 / OSGM15), fetched once from the PROJ CDN and cached."""
    import pyproj
    from pyproj import Transformer
    pyproj.network.set_network_enabled(True)
    return Transformer.from_crs('EPSG:4979', authority_crs(crs), always_xy=True)

def authority_crs(crs):
    """The registered (EPSG) equivalent of a CRS read from a file. A LAS header's WKT1 is often not recognised as the
    EPSG system it is, and PROJ then falls back to a 'ballpark' transformation: metres off horizontally and the geoid
    applied wrongly (seen: 49 m in height, from a WKT1 EXTENSION[PROJ4_GRIDS] geoid bound to the wrong datum).
    Matching it, or its horizontal and vertical parts, to EPSG codes lets PROJ choose the national grids itself."""
    from pyproj import CRS
    def epsg(c):  # matched by PROJ, else the EPSG code the file itself declares for it
        code = c.to_epsg(min_confidence=70)
        if code: return code
        d = c.to_json_dict(); d = d.get('source_crs', d) if d.get('type') == 'BoundCRS' else d  # WKT1 EXTENSION grids make a BoundCRS
        ident = d.get('id') or {}
        return int(ident['code']) if ident.get('authority') == 'EPSG' and str(ident.get('code', '')).isdigit() else None
    try:
        if (code := epsg(crs)): return CRS.from_epsg(code)
        codes = [epsg(s) for s in crs.sub_crs_list or []]
        if codes and all(codes): return CRS.from_user_input('EPSG:' + '+'.join(map(str, codes)))
    except Exception: pass
    return crs

def has_vertical(crs) -> bool:
    return len(crs.sub_crs_list or []) > 1 or (crs.is_vertical if hasattr(crs, 'is_vertical') else False)

def poses(metas, crs, origin):
    """World->camera rotations and centres in the cloud's CRS minus origin. Returns (list of dicts with 'R', 'C'
    (3,), 'meta'), info). Yaw is relative to true north; the projection's grid north differs from it by the meridian
    convergence (up to ~3 degrees), measured here per photo."""
    import numpy as np
    t = grid_transformer(crs)
    lon = np.array([m['lon'] for m in metas]); lat = np.array([m['lat'] for m in metas]); alt = np.array([m['alt'] for m in metas])
    E, N, H = (np.asarray(v, float) for v in t.transform(lon, lat, alt))
    E2, N2, _ = (np.asarray(v, float) for v in t.transform(lon, lat + 1e-4, alt))
    north = np.arctan2(E2 - E, N2 - N)  # grid azimuth of true north at each photo
    op = t.get_last_used_operation() if hasattr(t, 'get_last_used_operation') else None
    out = []
    for i, m in enumerate(metas):
        yaw, pitch, roll = math.radians(m['yaw']) + north[i], math.radians(m['pitch']), math.radians(m['roll'])
        f = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        r = np.array([math.cos(yaw), -math.sin(yaw), 0.0]); d = np.cross(f, r)
        r, d = r * math.cos(roll) + d * math.sin(roll), d * math.cos(roll) - r * math.sin(roll)
        out.append({'R': np.vstack([r, d, f]), 'C': np.array([E[i], N[i], H[i]]) - origin, 'meta': m})
    info = {'transformation': (op.name if op is not None else t.description)[:160], 'convergence_deg': round(float(np.degrees(np.median(north))), 3),
            'crs_used': authority_crs(crs).name, 'grids': [g.short_name for g in getattr(op, 'grids', []) or []]}
    return out, info

def camera_model(m):
    """COLMAP OPENCV camera line parameters (width, height, [fx, fy, cx, cy, k1, k2, p1, p2])."""
    return m['w'], m['h'], [m['fx'], m['fy'], m['cx'], m['cy'], *m['k']]

def quaternion(R):
    """COLMAP qw qx qy qz of a rotation matrix."""
    import numpy as np
    q = np.empty(4); tr = np.trace(R)
    if tr > 0:
        s = 2 * math.sqrt(tr + 1); q[:] = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q

def write_text_model(folder, cams, names):
    """cameras.txt / images.txt / points3D.txt (COLMAP text layout, no 2D points) so the 3D-box step reads these
    poses exactly like a COLMAP solve."""
    from pathlib import Path
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    ids, lines = {}, []
    with open(folder / 'images.txt', 'w') as o:
        for i, (c, name) in enumerate(zip(cams, names), 1):
            key = camera_model(c['meta']); key = (key[0], key[1], tuple(round(v, 6) for v in key[2]))
            cid = ids.setdefault(key, len(ids) + 1); v = [float(x) for x in (*quaternion(c['R']), *(-c['R'] @ c['C']))]
            o.write(f'{i} ' + ' '.join(repr(x) for x in v) + f' {cid} {name}\n\n')
    with open(folder / 'cameras.txt', 'w') as o:
        for (w, h, params), cid in ids.items(): o.write(f'{cid} OPENCV {w} {h} ' + ' '.join(repr(float(p)) for p in params) + '\n')
    (folder / 'points3D.txt').write_text('')
    return folder

def project(cam, X):
    """Pixels (N,2) and depth (N,) of world points in one photo (OpenCV lens model)."""
    import numpy as np
    m = cam['meta']; Xc = (X - cam['C']) @ cam['R'].T; z = Xc[:, 2]; zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs; k1, k2, p1, p2 = m['k']; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([m['fx'] * xd + m['cx'], m['fy'] * yd + m['cy']], 1), z

REFINE_SPAN_DEG = 3.0     # drone gimbal / compass direction errors are a degree or two; corrections beyond this are not trusted
REFINE_MIN_GAIN = .85     # accepted only when the structure's outline lies this much closer to the photo's edges

def _small_rotations(angles):
    """(n, 3) yaw, pitch, roll in degrees (about the camera's down, right and view axes) -> (n, 3, 3)."""
    import numpy as np
    y, p, r = np.radians(np.asarray(angles, float)).T; o, z = np.ones_like(y), np.zeros_like(y)
    Ry = np.stack([np.stack([np.cos(y), z, np.sin(y)], -1), np.stack([z, o, z], -1), np.stack([-np.sin(y), z, np.cos(y)], -1)], 1)
    Rx = np.stack([np.stack([o, z, z], -1), np.stack([z, np.cos(p), -np.sin(p)], -1), np.stack([z, np.sin(p), np.cos(p)], -1)], 1)
    Rz = np.stack([np.stack([np.cos(r), -np.sin(r), z], -1), np.stack([np.sin(r), np.cos(r), z], -1), np.stack([z, z, o], -1)], 1)
    return Rz @ Rx @ Ry

def refine_direction(cam, structure, jpeg):
    """Correct one photo's direction (small yaw / pitch / roll) so the structure's outline in the cloud lies on the
    photo's edges. Positions (RTK) are kept: the gimbal and compass angles are what is off, by a degree or two, which
    puts a mast a few decimetres beside itself. cam: {'R', 'C', 'meta'} (cloud frame); structure: (N,3) points of the
    masts in the same frame; jpeg: a small copy of the photo (bytes). Returns (yaw, pitch, roll, gain) or None."""
    import io, numpy as np, cv2
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert('L')); H, W = img.shape; m = cam['meta']; sc = W / m['w']
    Xc = (structure - cam['C']) @ cam['R'].T
    def pix(X):  # (..., k, 3) camera-frame points -> pixels at the copy's scale, and in-view mask
        z = X[..., 2]; zs = np.where(z > .05, z, 1.0); x, y = X[..., 0] / zs, X[..., 1] / zs; k1, k2, p1, p2 = m['k']; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        u = (m['fx'] * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + m['cx']) * sc; v = (m['fy'] * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + m['cy']) * sc
        return u, v, (z > .05) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, ok = pix(Xc)
    if ok.sum() < 2000: return None  # the structure is hardly in this photo
    ui, vi = u[ok].astype(int), v[ok].astype(int); mask = np.zeros((H, W), np.uint8); mask[vi, ui] = 255
    mask = cv2.morphologyEx(cv2.dilate(mask, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edge = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))
    on = cv2.dilate(edge, np.ones((5, 5), np.uint8))[vi, ui] > 0; Q = Xc[ok][on]  # the structure points on (within 2 px of) its outline
    if len(Q) < 300: return None
    if len(Q) > 4000: Q = Q[np.random.default_rng(0).choice(len(Q), 4000, replace=False)]
    dt = np.minimum(cv2.distanceTransform(255 - cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 50, 150), cv2.DIST_L2, 3), 8.0)
    def score(angles):
        R = _small_rotations(angles); u, v, ok = pix(np.einsum('nij,kj->nki', R, Q))
        d = dt[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)]; return np.where(ok, d, 8.0).mean(1)
    s0 = float(score([[0, 0, 0]])[0])
    g = np.arange(-REFINE_SPAN_DEG, REFINE_SPAN_DEG + 1e-9, .25); A = np.array([[a, b, 0] for a in g for b in g]); sc_ = score(A); best = A[sc_.argmin()]
    f = np.arange(-.3, .3001, .05); A = np.array([[best[0] + a, best[1] + b, r] for a in f for b in f for r in np.arange(-.75, .7501, .25)]); sc_ = score(A)
    best, s1 = A[sc_.argmin()], float(sc_.min())
    if s1 > REFINE_MIN_GAIN * s0 or np.abs(best).max() > REFINE_SPAN_DEG + .3: return None
    return float(best[0]), float(best[1]), float(best[2]), round(s1 / max(s0, 1e-9), 3)

def refine_directions(cams, structure, photo, progress=None):
    """refine_direction for every photo (photo(i) -> small copy bytes or None), applied to cams in place.
    Returns {'refined', 'looked_at', 'median_deg'}."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    P = np.asarray(structure, float)
    if len(P) > 200_000: P = P[np.random.default_rng(0).choice(len(P), 200_000, replace=False)]
    done, seen, size = 0, 0, []
    def one(i):
        try:
            b = photo(i)
            return refine_direction(cams[i], P, b) if b else None
        except Exception: return None
    with ThreadPoolExecutor(8) as pool:
        for i, r in enumerate(pool.map(one, range(len(cams)))):
            if progress and i % 100 == 0: progress(f'Correcting photo directions against the point cloud: {i}/{len(cams)}')
            if r is None: continue
            seen += 1; cams[i]['R'] = _small_rotations([r[:3]])[0] @ cams[i]['R']; done += 1; size.append(float(np.linalg.norm(r[:3])))
    return {'refined': done, 'photos': len(cams), 'median_deg': round(float(np.median(size)), 2) if size else 0.0}

def targets(cams, min_share=.08, most=4, miss_m=4.0):
    """Every structure the photos were taken around: a site often has several masts, each flown around in turn. The
    vertical lines that the cameras' viewing directions pass within miss_m metres of, each seen by at least min_share
    of the photos, most-photographed first: [{'T': (x, y, z), 'photos': n}]. Photos looking straight down say nothing
    about this; with fewer than 10 photos looking sideways the single meeting point (target) is returned."""
    import numpy as np
    C = np.array([c['C'] for c in cams], float); D = np.array([c['R'][2] for c in cams], float)
    h = D[:, :2]; n = np.linalg.norm(h, axis=1); side = np.flatnonzero(n > .25)
    if len(side) < 10: return [{'T': target(cams), 'photos': len(cams)}]
    u = h / np.maximum(n, 1e-9)[:, None]; c2 = C[:, :2]; rng = np.random.default_rng(0)
    def miss(T, idx):  # horizontal distance at which each view line passes the vertical line through T (behind: never)
        d = T - c2[idx]; t = (d * u[idx]).sum(1); off = np.linalg.norm(d - t[:, None] * u[idx], axis=1)
        return np.where(t > 0, off, np.inf)
    def solve(idx):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for i in idx: Pm = np.eye(2) - np.outer(u[i], u[i]); A += Pm; b += Pm @ c2[i]
        return np.linalg.solve(A, b) if np.linalg.cond(A) < 1e8 else None
    left, out, need = side.copy(), [], max(min_share * len(side), 10)
    while len(out) < most and len(left) >= need:
        best, inl = None, np.zeros(0, int)
        for _ in range(400):  # candidates where two view lines cross; the one most lines pass close to wins
            i, j = rng.choice(left, 2, replace=False); M = np.c_[u[i], -u[j]]
            if abs(np.linalg.det(M)) < .1: continue
            t = np.linalg.solve(M, c2[j] - c2[i])
            if (t <= 0).any(): continue
            T = c2[i] + t[0] * u[i]; k = left[miss(T, left) <= miss_m]
            if len(k) > len(inl): best, inl = T, k
        if best is None or len(inl) < need: break
        for _ in range(3):  # refine on its own photos
            T = solve(inl)
            if T is None: break
            best = T; inl = left[miss(best, left) <= miss_m]
        if any(np.linalg.norm(best - o['T'][:2]) < miss_m for o in out): left = np.setdiff1d(left, inl); continue
        s = ((best - c2[inl]) * u[inl]).sum(1) / n[inl]  # along each view line to where it passes the structure
        z = float(np.median(C[inl, 2] + s * D[inl, 2]))
        out.append({'T': np.r_[best, z], 'photos': int(len(inl))}); left = np.setdiff1d(left, inl)
    return out or [{'T': target(cams), 'photos': len(cams)}]

def target(cams):
    """The point the cameras look at (least squares meeting point of their viewing rays; rays far off it, e.g. photos
    of the horizon, are dropped and it is solved again). Inspection orbits converge on the structure."""
    import numpy as np
    C = np.array([c['C'] for c in cams]); D = np.array([c['R'][2] for c in cams]); use = np.ones(len(C), bool); T = C.mean(0)
    for _ in range(3):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for c, d in zip(C[use], D[use]): Pm = np.eye(3) - np.outer(d, d); A += Pm; b += Pm @ c
        if np.linalg.cond(A) > 1e8: return C.mean(0)  # parallel rays (a mapping flight looking straight down)
        T = np.linalg.solve(A, b)
        off = np.linalg.norm(np.cross(T - C, D), axis=1)  # distance of T from each ray
        use = off <= max(3 * np.median(off), 2.0)
    return T

def check(cams, points, max_range=150.0, tol=0.5):
    """Share of photos whose centre ray meets the cloud (within tol m): a camera pose that is badly off looks past it."""
    import numpy as np
    from scipy.spatial import cKDTree
    if not len(points) or not cams: return 0.0
    P = points if len(points) <= 2_000_000 else points[np.random.default_rng(0).choice(len(points), 2_000_000, replace=False)]
    tree = cKDTree(P); steps = np.arange(1.0, max_range, tol)
    hits = [bool((tree.query(c['C'] + steps[:, None] * c['R'][2], distance_upper_bound=tol)[0] < tol).any()) for c in cams]
    return round(100.0 * sum(hits) / len(hits), 1)
