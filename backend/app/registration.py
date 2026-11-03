"""Conservative 2D detection -> reconstructed 3D candidate registration.

COLMAP already knows which sparse 3D feature point produced each 2D keypoint.  For
each high-confidence equipment box we take the robust centre of the reconstructed
points inside the box, transform it into the browser model's levelled coordinate
frame, then cluster same-class estimates seen in multiple photos.  The result is a
review candidate, never an automatically confirmed physical asset.

Each candidate also gets an oriented 3D box: width/height from the 2D box size x camera distance / focal length
(median over the photos that saw it; height also covers the vertical spread of its 3D points, since oblique drone
views foreshorten it), depth as a per-class share of the width, and a yaw that faces away from the tower axis
(equipment on a tower faces outward). Sizes are in the viewer's units: the model has no real-world scale yet.
"""
from __future__ import annotations

import math, shutil, subprocess, tempfile
from collections import defaultdict
from pathlib import Path

from . import reconstruct

CLASS_RULES = {  # detection floor, grouping radius, depth ratio, and final multi-view quality gates
    'sector_antenna': {'confidence': .55, 'radius': .18, 'depth': .35, 'candidate': .62, 'views': 4, 'fit': .42},
    'microwave_dish': {'confidence': .70, 'radius': .20, 'depth': .60, 'candidate': .62, 'views': 4, 'fit': .40},
    'rru': {'confidence': .75, 'radius': .14, 'depth': .60, 'candidate': .68, 'views': 4, 'fit': .42},
    'cabinets': {'confidence': .70, 'radius': .18, 'depth': .80, 'candidate': .62, 'views': 3, 'fit': .40},
    'mounting_brackets': {'confidence': .50, 'radius': .15, 'depth': .50, 'candidate': .50, 'views': 5, 'fit': .42},
    'bolts': {'confidence': .45, 'radius': .06, 'depth': 1.00, 'candidate': .38, 'views': 5, 'fit': .50, 'photo_boxes': True},
    'Jumper Cables / RF Cables': {'confidence': .45, 'radius': .25, 'depth': .30, 'candidate': .46, 'views': 5, 'fit': .40},
    'Cable Management System': {'confidence': .50, 'radius': .18, 'depth': .35, 'candidate': .48, 'views': 4, 'fit': .40},
    'Cable Clamps': {'confidence': .45, 'radius': .08, 'depth': .80, 'candidate': .40, 'views': 5, 'fit': .48, 'photo_boxes': True},
    'Foundation': {'confidence': .50, 'radius': .30, 'depth': .70, 'candidate': .45, 'views': 3, 'fit': .40},
    'ladder': {'confidence': .20, 'radius': .22, 'depth': .25, 'candidate': .38, 'views': 3, 'fit': .28},
}
DEFAULT_RULE = {'confidence': .55, 'radius': .15, 'depth': .50, 'candidate': .50, 'views': 4, 'fit': .42}
FACING_CLASSES = {'sector_antenna', 'microwave_dish', 'rru'}  # mounted around the tower, so they define its axis
# photo_boxes: parts smaller than the dense cloud resolves (a bolt covers a few dense points) may keep the box
# triangulated from the photos; the views/fit gates above then decide alone whether it is one real object.
SURFACE_ANCHORED_CLASSES = {'mounting_brackets'}  # irregular steelwork: dense geometry is safer than triangulated 2D box centres
def rule_for(class_name): return CLASS_RULES.get(class_name, DEFAULT_RULE)

def _confidence(d):
    """Human-drawn detections have no model score and count as fully trusted evidence."""
    return float(d.confidence) if d.confidence is not None else 1.0

def _det_iou(a, b):
    """2D IoU for detector records in their source image coordinate system."""
    left, top = max(a.x1, b.x1), max(a.y1, b.y1)
    right, bottom = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _nms(boxes, threshold=.55):
    """Suppress duplicate same-class detector boxes within one photo."""
    ordered = sorted(boxes, key=_confidence, reverse=True)
    kept = []
    for box in ordered:
        if any(box.class_name == other.class_name and _det_iou(box, other) >= threshold for other in kept):
            continue
        kept.append(box)
    return kept

def _convert_model(model_dir: Path):
    tmp = Path(tempfile.mkdtemp(prefix='tower-registration-'))
    try:
        reconstruct.write_text_model(model_dir, tmp)  # read the binary model in Python; colmap.exe isn't needed for this
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def _read_points(path: Path):
    import numpy as np
    ids, xyz = [], []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line or line.startswith('#'): continue
        p = line.split(); ids.append(int(p[0])); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return {pid: point for pid, point in zip(ids, np.asarray(xyz, dtype=float))}

def _read_observations(path: Path):
    import numpy as np
    lines = [line for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if not line.startswith('#')]  # keep blank lines: an image with no points has an empty second line
    out = {}
    for header, points in zip(lines[0::2], lines[1::2]):
        name = ' '.join(header.split()[9:])
        p = points.split(); triples = len(p)//3
        xy = np.empty((triples, 2), dtype=float); ids = np.empty(triples, dtype='int64')
        if triples:
            xy[:, 0] = p[0::3]; xy[:, 1] = p[1::3]; ids[:] = p[2::3]
            valid = ids >= 0; xy, ids = xy[valid], ids[valid]
        out[name] = (xy, ids)
    return out

def _read_cameras(path: Path):
    """camera_id -> focal length in pixels (fx/fy averaged for models that have both)."""
    out = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line or line.startswith('#'): continue
        p = line.split(); model, params = p[1], [float(v) for v in p[4:]]
        two = model in ('PINHOLE', 'OPENCV', 'FULL_OPENCV', 'OPENCV_FISHEYE', 'FOV', 'THIN_PRISM_FISHEYE')
        out[int(p[0])] = (params[0] + params[1]) / 2 if two and len(params) > 1 else params[0]
    return out

def _read_images(path: Path):
    """name -> (2D xy, 3D point ids, camera centre in model coords, camera id)."""
    import numpy as np
    lines = [line for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if not line.startswith('#')]  # keep blank lines: an image with no points has an empty second line
    out = {}
    for header, points in zip(lines[0::2], lines[1::2]):
        h = header.split(); qw, qx, qy, qz, tx, ty, tz = map(float, h[1:8])
        R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)], [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                      [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
        xy, ids = _read_observations_line(points)
        out[' '.join(h[9:])] = (xy, ids, -R.T @ np.array([tx, ty, tz]), int(h[8]), R, np.array([tx, ty, tz]))
    return out

def _read_observations_line(points: str):
    import numpy as np
    p = points.split(); triples = len(p)//3
    xy = np.empty((triples, 2), dtype=float); ids = np.empty(triples, dtype='int64')
    if triples:
        xy[:, 0] = p[0::3]; xy[:, 1] = p[1::3]; ids[:] = p[2::3]
        valid = ids >= 0; xy, ids = xy[valid], ids[valid]
    return xy, ids

def _clusters(rows, radius):
    """Greedy compact clusters. A candidate needs support from >=2 source images."""
    import numpy as np
    remaining = set(range(len(rows))); result = []
    while remaining:
        seed = max(remaining, key=lambda i: rows[i]['weight']); members = {seed}
        for _ in range(4):
            centre = np.average([rows[i]['position'] for i in members], axis=0,
                                weights=[rows[i]['weight'] for i in members])
            grown = {i for i in remaining if float(np.linalg.norm(rows[i]['position']-centre)) <= radius}
            if grown == members: break
            members = grown
        remaining -= members
        media = {rows[i]['media_id'] for i in members}
        if len(media) < 2: continue
        weights = np.asarray([rows[i]['weight'] for i in members]); positions = np.asarray([rows[i]['position'] for i in members])
        centre = np.average(positions, axis=0, weights=weights)
        distances = np.linalg.norm(positions-centre, axis=1); spread = float(np.median(distances))
        support = len(media); confidence = min(.95, float(np.average([rows[i]['confidence'] for i in members], weights=weights)) + min(.12, .02*math.log2(support+1)))
        ys = np.concatenate([np.asarray(rows[i].get('points', np.zeros((0, 3))))[:, 1] for i in members]) if members else np.zeros(0)
        result.append({'position': centre, 'confidence': confidence, 'spread': spread, 'media_count': support,
                       'width': float(np.median([rows[i].get('width', 0.0) for i in members])), 'height': float(np.median([rows[i].get('height', 0.0) for i in members])),  # size fields are optional
                       'y_extent': float(np.percentile(ys, 90) - np.percentile(ys, 10)) if len(ys) > 2 else 0.0,
                       'evidence_count': len(members), 'point_count': sum(rows[i]['point_count'] for i in members),
                       'detection_ids': [rows[i]['detection_id'] for i in members]})
    return result

def build_candidates(work: Path, transform, media, detections, dense_path: Path | None = None, reviewed_media=()):
    """Return reviewable candidate dicts in the same coordinate frame as web/*.ply.
    Detections on photos a reviewer approved are used whatever their confidence (the class floor is for raw YOLO output)."""
    import numpy as np
    work = Path(work); sparse = work/'sparse'/'0'
    if not sparse.is_dir():
        choices = [p for p in (work/'sparse').glob('*') if p.is_dir()]
        sparse = max(choices, key=lambda p:(p/'images.bin').stat().st_size, default=None)
    if not sparse: raise RuntimeError('The reconstruction has no sparse camera model')
    M = np.asarray(transform, dtype=float)
    if M.shape != (4, 4): raise RuntimeError('The reconstruction has no valid viewer coordinate transform')
    photos = sorted([m for m in media if m.parent_media_id is None and m.content_type.startswith('image/')], key=lambda m:(m.filename.lower(),m.id))
    photo_names = {m.id:f'photos/{i:05d}{Path(m.storage_key).suffix.lower()}' for i,m in enumerate(photos)}
    by_media = defaultdict(list)
    for d in detections:
        if d.media_id in photo_names and (d.media_id in reviewed_media or (d.confidence or 0) >= rule_for(d.class_name)['confidence']): by_media[d.media_id].append(d)
    by_media = {media_id: _nms(boxes) for media_id, boxes in by_media.items()}
    if not by_media: return []

    converter = _convert_model(sparse); txt = next(converter)
    try:
        points, observations, focal = _read_points(txt/'points3D.txt'), _read_images(txt/'images.txt'), _read_cameras(txt/'cameras.txt')
        estimates = defaultdict(list)
        for m in photos:
            boxes = by_media.get(m.id); observed = observations.get(photo_names[m.id])
            if not boxes or observed is None: continue
            xy, pids, cam_centre, cam_id = observed[:4]
            if not len(pids) or cam_id not in focal: continue
            cam = M[:3,:3] @ cam_centre + M[:3,3]; f = focal[cam_id]  # camera position in viewer coordinates
            xyz = np.asarray([points.get(int(pid), [np.nan]*3) for pid in pids]); valid = np.isfinite(xyz).all(1)
            xy, pids, xyz = xy[valid], pids[valid], xyz[valid]
            xyz = xyz @ M[:3,:3].T + M[:3,3]
            for d in boxes:
                # Slightly inset the box to reduce tower/background points at its edge.
                mx=.06*(d.x2-d.x1); my=.06*(d.y2-d.y1)
                inside=(xy[:,0]>=d.x1+mx)&(xy[:,0]<=d.x2-mx)&(xy[:,1]>=d.y1+my)&(xy[:,1]<=d.y2-my)
                sample=xyz[inside]
                if len(sample)<3: continue
                centre=np.median(sample,axis=0); distance=np.linalg.norm(sample-centre,axis=1)
                sample=sample[distance<=np.percentile(distance,70)]
                if len(sample)<3: continue
                centre=np.median(sample,axis=0); spread=float(np.median(np.linalg.norm(sample-centre,axis=1)))
                if spread>.45: continue
                point_factor=min(1.,math.log2(len(sample)+1)/4); depth=float(np.linalg.norm(centre-cam))
                confidence = _confidence(d)
                estimates[d.class_name].append({'position':centre,'confidence':confidence,'weight':confidence*point_factor/(1+spread),
                                                'spread':spread,'point_count':len(sample),'media_id':m.id,'detection_id':d.id,'points':sample,
                                                'width':(d.x2-d.x1)*depth/f,'height':(d.y2-d.y1)*depth/f})  # pinhole: size = pixels x distance / focal
        out=[]
        for class_name, rows in estimates.items():
            for c in _clusters(rows, rule_for(class_name)['radius']): out.append({'class_name':class_name,**c})
        out = add_boxes(out)
        if dense_path and Path(dense_path).is_file():  # place, group and size everything from the dense cloud instead
            out = dense_candidates(reconstruct.read_ply(dense_path)['points'], M, photos, photo_names, by_media, observations, _read_camera_models(txt/'cameras.txt'))
        return sorted(out,key=lambda c:(c['class_name'],-c['confidence'],-c['media_count']))
    finally:
        try: converter.close()
        except Exception: pass

def _tower_axis(candidates):
    import numpy as np
    facing = [c['position'] for c in candidates if c['class_name'] in FACING_CLASSES] or [c['position'] for c in candidates]
    return np.median(np.asarray(facing)[:, [0, 2]], axis=0)  # the tower's vertical axis (x, z in viewer coordinates)

def add_boxes(candidates):
    """Oriented box per candidate: size_w along the tower's tangent, size_d facing outward from the tower axis, size_h vertical."""
    import numpy as np
    if not candidates: return candidates
    axis = _tower_axis(candidates)
    for c in candidates:
        w = max(c.pop('width'), 1e-3); h = max(c.pop('height'), c.pop('y_extent'), 1e-3)
        dx, dz = c['position'][0] - axis[0], c['position'][2] - axis[1]
        c.update(size_w=float(w), size_h=float(h), size_d=float(min(w, w * rule_for(c['class_name'])['depth'])),
                 yaw=float(math.atan2(dx, dz)) if math.hypot(dx, dz) > 1e-6 else 0.0,  # rotation about +Y taking +Z to "outward"
                 box_method='estimate', box_points=None)
    return candidates

# ---------- dense point cloud ----------
# With a dense cloud, detections are placed and grouped from it instead of from the sparse feature tracks: the
# sparse points inside a box are few and often belong to the neighbouring panel, so antennas a few centimetres
# apart merged into one candidate. Here every detection is anchored on the dense surface its photo actually sees
# (a z-buffer hides points behind it), same-class anchors are grouped with at most one detection per photo, and the
# box is fitted to the dense points that fall inside the group's detection boxes in most of its photos (visual hull).
MIN_DENSE_POINTS = 40           # fewer than this and the photo-based estimate is kept
DENSE_SAMPLE = 1_000_000        # dense points used (a random sample keeps a large cloud fast)
ZBUF_CELLS = 160                # z-buffer columns across the image
VISIBLE_TOL = .05               # viewer units behind the nearest surface still counted as visible
VOTE_FRACTION = .6              # a dense point must fall inside the detection box in this share of the photos that frame it
MIN_FIT_IOU = .3                # a box whose outline overlaps its own detection boxes less than this (median IoU) is dropped:
                                # its detections were of different objects (typical for bolts: tiny, dozens per photo)

def _read_camera_models(path: Path):
    """camera_id -> (model name, params, width, height)."""
    out = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line or line.startswith('#'): continue
        p = line.split(); out[int(p[0])] = (p[1], [float(v) for v in p[4:]], int(p[2]), int(p[3]))
    return out

def project(X, R, t, cam):
    """COLMAP camera projection with lens distortion: model-space points (N,3) -> pixels (N,2) and depth (N,)."""
    import numpy as np
    model, p, _, _ = cam
    Xc = X @ R.T + t; z = Xc[:, 2]; zs = np.where(np.abs(z) < 1e-9, 1e-9, z); x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        fx = fy = p[0]; cx, cy = p[1], p[2]; k1 = p[3] if len(p) > 3 else 0.0; k2 = p[4] if len(p) > 4 else 0.0; p1 = p2 = 0.0
    elif model == 'PINHOLE': fx, fy, cx, cy = p[:4]; k1 = k2 = p1 = p2 = 0.0
    else: fx, fy, cx, cy = p[:4]; k1, k2, p1, p2 = (list(p[4:8]) + [0.0] * 4)[:4]  # OPENCV family (fisheye terms approximated)
    r2 = x * x + y * y; radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1), z

def fit_box(V, axis):
    """Oriented box around viewer-space points: vertical stays +Y; horizontal axes from the footprint's principal
    directions, the thin one facing away from the tower axis. Percentiles keep stray points from inflating it.
    Returns None for too few or degenerate points."""
    import numpy as np
    V = np.asarray(V, float)
    if len(V) < 4 or not np.isfinite(V).all(): return None
    H = V[:, [0, 2]]; mid = np.median(H, 0); Hc = H - mid
    try: _, vecs = np.linalg.eigh(np.cov(Hc.T))
    except np.linalg.LinAlgError: return None
    if not np.isfinite(vecs).all(): return None
    major, minor = vecs[:, 1], vecs[:, 0]
    if minor @ (mid - axis) < 0: minor = -minor  # face outward
    a, b, y = Hc @ major, Hc @ minor, V[:, 1]
    (a0, a1), (b0, b1), (y0, y1) = (np.percentile(v, [3, 97]) for v in (a, b, y))
    centre_h = mid + major * (a0 + a1) / 2 + minor * (b0 + b1) / 2
    return {'position': np.array([centre_h[0], (y0 + y1) / 2, centre_h[1]]), 'size_w': float(max(a1 - a0, 1e-3)),
            'size_d': float(max(b1 - b0, 1e-3)), 'size_h': float(max(y1 - y0, 1e-3)), 'yaw': float(math.atan2(minor[0], minor[1]))}

def _box_px(d, cam):
    """Detection box in the camera's pixel grid (detections may have been made on a resized copy)."""
    sx = cam[2] / d.image_width if getattr(d, 'image_width', None) else 1.0
    sy = cam[3] / d.image_height if getattr(d, 'image_height', None) else 1.0
    return d.x1 * sx, d.x2 * sx, d.y1 * sy, d.y2 * sy

def _anchor_detections(P, z_tol, photos, photo_names, by_media, poses, cameras):
    """One row per detection: the dense surface it sees (model coords), its box size in model units at that distance."""
    import numpy as np
    rows = defaultdict(list)
    for m in photos:
        boxes = by_media.get(m.id); pose = poses.get(photo_names[m.id])
        if not boxes or pose is None or pose[3] not in cameras: continue
        cam = cameras[pose[3]]; W, H = cam[2], cam[3]; f = cam[1][0]
        uv, z = project(P, pose[4], pose[5], cam)
        idx = np.flatnonzero((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        if not len(idx): continue
        cell = W / ZBUF_CELLS; cols = int(W // cell) + 1
        key = (uv[idx, 1] // cell).astype(np.int64) * cols + (uv[idx, 0] // cell).astype(np.int64)
        zmin = np.full(int(key.max()) + 1, np.inf); np.minimum.at(zmin, key, z[idx])
        vis = idx[z[idx] <= zmin[key] + z_tol]; u, v, zv = uv[vis, 0], uv[vis, 1], z[vis]
        for d in boxes:
            x1, x2, y1, y2 = _box_px(d, cam); mx, my = .15 * (x2 - x1), .15 * (y2 - y1)  # inner box: less background
            inside = (u >= x1 + mx) & (u <= x2 - mx) & (v >= y1 + my) & (v <= y2 - my)
            if inside.sum() < 4: continue  # small parts (clamps, bolts) cover only a few dense points
            zz = zv[inside]; z_ref = float(np.percentile(zz, 20))
            w3, h3 = (x2 - x1) * z_ref / f, (y2 - y1) * z_ref / f  # pinhole: size = pixels x distance / focal
            front = vis[inside][zz <= z_ref + .5 * min(w3, h3)]  # the nearest layer is the object, not the tower behind it
            if len(front) < 3: continue
            rows[d.class_name].append({'anchor': np.median(P[front], 0), 'media_id': m.id, 'name': photo_names[m.id], 'det': d,
                                       'confidence': _confidence(d), 'w': w3, 'h': h3, 'points': len(front)})
    return rows

def _group(rows, radius_of):
    """Greedy groups around the most confident anchors, at most one detection per photo (its closest)."""
    import numpy as np
    if not rows: return []
    A = np.asarray([r['anchor'] for r in rows]); media = np.asarray([r['media_id'] for r in rows])
    order = sorted(range(len(rows)), key=lambda i: -rows[i]['confidence'] * min(1, math.log2(rows[i]['points'] + 1) / 6))
    free = np.ones(len(rows), bool); groups = []
    for seed in order:
        if not free[seed]: continue
        centre = A[seed]; radius = radius_of(rows[seed]); members = np.array([seed])
        for _ in range(3):
            idx = np.flatnonzero(free); dist = np.linalg.norm(A[idx] - centre, axis=1); idx, dist = idx[dist <= radius], dist[dist <= radius]
            idx = idx[np.lexsort((dist, media[idx]))]; idx = idx[np.r_[True, media[idx][1:] != media[idx][:-1]]]  # closest per photo
            members = np.union1d(idx[media[idx] != media[seed]], [seed]); centre = np.median(A[members], 0)
        free[members] = False
        if len(set(media[members].tolist())) >= 2: groups.append(members.tolist())
    return groups

def dense_candidates(dense_model_pts, M, photos, photo_names, by_media, poses, cameras):
    """Candidates (viewer coordinates, with fitted boxes) from the dense cloud; see the comment above."""
    import numpy as np
    M = np.asarray(M, float); s = float(np.linalg.norm(M[:3, 0])); Rm, T = M[:3, :3] / s, M[:3, 3]
    to_view = lambda X: s * (np.atleast_2d(X) @ Rm.T) + T
    P = np.asarray(dense_model_pts, dtype=np.float64)
    if len(P) > DENSE_SAMPLE: P = P[np.random.default_rng(0).choice(len(P), DENSE_SAMPLE, replace=False)]
    if len(P) < MIN_DENSE_POINTS: return []
    rows = _anchor_detections(P, VISIBLE_TOL / s, photos, photo_names, by_media, poses, cameras)
    out = []
    for class_name, rs in rows.items():
        rule = rule_for(class_name)
        for members in _group(rs, lambda r: max(.5 * min(r['w'], r['h']), .5 * rule['radius'] / s)):
            ms = [rs[i] for i in members]; centre = np.median([r['anchor'] for r in ms], 0)
            w, h = float(np.median([r['w'] for r in ms])), float(np.median([r['h'] for r in ms]))
            near = P[np.sum((P - centre) ** 2, 1) < (.75 * max(w, h)) ** 2]
            votes = np.zeros(len(near), np.int32); seen = np.zeros(len(near), np.int32)
            for r in ms:
                pose = poses[r['name']]; cam = cameras[pose[3]]; x1, x2, y1, y2 = _box_px(r['det'], cam); mx, my = .03 * (x2 - x1), .03 * (y2 - y1)
                uv, z = project(near, pose[4], pose[5], cam); front = z > 0
                seen += front & (uv[:, 0] >= 0) & (uv[:, 0] < cam[2]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam[3])
                votes += front & (uv[:, 0] >= x1 - mx) & (uv[:, 0] <= x2 + mx) & (uv[:, 1] >= y1 - my) & (uv[:, 1] <= y2 + my)
            obj = near[(votes >= 2) & (votes >= np.ceil(VOTE_FRACTION * seen))]
            support = len({r['media_id'] for r in ms})
            out.append({'class_name': class_name, 'position': to_view(centre)[0], 'media_count': support, 'evidence_count': len(ms),
                        'confidence': min(.95, float(np.mean([r['confidence'] for r in ms])) + min(.12, .02 * math.log2(support + 1))),
                        'spread': float(np.median(np.linalg.norm(np.asarray([r['anchor'] for r in ms]) - centre, axis=1))) * s,
                        'point_count': int(sum(r['points'] for r in ms)), 'detection_ids': [r['det'].id for r in ms],
                        'width': w * s, 'height': h * s, 'y_extent': 0.0, '_obj': to_view(obj) if len(obj) >= MIN_DENSE_POINTS else None,
                        '_views': [(r['name'], r['det']) for r in ms]})
    add_boxes(out)  # photo estimate first (also gives the tower axis), then the dense fit where there are enough points
    axis = _tower_axis(out) if out else None
    for c in out:
        obj = c.pop('_obj'); box = fit_box(obj, axis) if obj is not None else None
        if box: c.update(box, box_method='dense', box_points=int(len(obj)))
        fit_to_detections(c, c['_views'], poses, cameras, M, move_centre=c['class_name'] not in SURFACE_ANCHORED_CLASSES)
    out = merge_duplicate_candidates(out, poses, cameras, M)
    clean = []
    for c in out:
        c.pop('_views', None)
        if candidate_is_valid(c): clean.append(c)
    return clean

def _candidate_box_iou(a, b):
    """IoU of two detection records that belong to the same source image."""
    return _det_iou(a, b)

def merge_duplicate_candidates(candidates, poses, cameras, M):
    """Merge fragments of one physical asset without merging side-by-side equipment.

    Two candidates seen in the same photos are duplicates only when their 2D evidence
    overlaps in at least two photos. Candidates with disjoint photo sets may merge by
    3D proximity. This makes the image evidence, not only a loose 3D radius, decide
    whether neighbouring antennas are one instance or two.
    """
    import numpy as np
    n = len(candidates)
    if n < 2: return candidates
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        if a != b: parent[b] = a
    for i in range(n):
        a = candidates[i]; av = {name: det for name, det in a.get('_views', [])}
        for j in range(i + 1, n):
            b = candidates[j]
            if a['class_name'] != b['class_name']: continue
            bv = {name: det for name, det in b.get('_views', [])}; common = set(av) & set(bv)
            duplicate = False
            if common:
                overlaps = [_candidate_box_iou(av[name], bv[name]) for name in common]
                duplicate = len(common) >= 2 and float(np.median(overlaps)) >= .35
            else:
                pa, pb = np.asarray(a['position']), np.asarray(b['position'])
                da = math.sqrt(sum(float(a.get(k) or 0) ** 2 for k in ('size_w', 'size_h', 'size_d')))
                db = math.sqrt(sum(float(b.get(k) or 0) ** 2 for k in ('size_w', 'size_h', 'size_d')))
                duplicate = float(np.linalg.norm(pa - pb)) <= .65 * max(da, db)
            if duplicate: union(i, j)
    groups = defaultdict(list)
    for i in range(n): groups[root(i)].append(i)
    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(candidates[members[0]]); continue
        score = lambda c: float(c.get('fit_iou') or 0) * float(c.get('confidence') or 0) * math.log2(int(c.get('media_count') or 0) + 1)
        base = dict(max((candidates[i] for i in members), key=score))
        # Keep at most one (highest confidence) detection per photo.
        by_name = {}
        for i in members:
            for name, det in candidates[i].get('_views', []):
                if name not in by_name or _confidence(det) > _confidence(by_name[name]): by_name[name] = det
        views = list(by_name.items()); base['_views'] = views
        base['media_count'] = base['evidence_count'] = len(views)
        base['detection_ids'] = [det.id for _, det in views]
        base['point_count'] = sum(int(candidates[i].get('point_count') or 0) for i in members)
        base['box_points'] = sum(int(candidates[i].get('box_points') or 0) for i in members)
        base['confidence'] = max(float(candidates[i].get('confidence') or 0) for i in members)
        base['spread'] = min(float(candidates[i].get('spread') or 0) for i in members)
        fit_to_detections(base, views, poses, cameras, M, move_centre=base['class_name'] not in SURFACE_ANCHORED_CLASSES)
        merged.append(base)
    return merged

def candidate_is_valid(c):
    """Only geometrically supported, repeatable instances are safe to draw on the twin."""
    rule = rule_for(c['class_name'])
    dims = [float(c.get(k) or 0) for k in ('size_w', 'size_h', 'size_d')]
    fit = float(c.get('fit_iou') or 0)
    if not all(math.isfinite(v) and v > 0 for v in dims): return False
    if c.get('box_method') != 'dense' and not (rule.get('photo_boxes') and c.get('box_method') == 'estimate'): return False
    if int(c.get('media_count') or 0) < rule['views']: return False
    if float(c.get('confidence') or 0) < rule['candidate'] or not math.isfinite(fit) or fit < max(MIN_FIT_IOU, rule['fit']): return False
    w, h, _ = dims
    if c['class_name'] == 'sector_antenna' and h < 1.35 * w: return False
    if c['class_name'] == 'rru' and h < .50 * w: return False
    if c['class_name'] == 'microwave_dish' and not .25 <= h / w <= 2.5: return False
    if c['class_name'] == 'ladder' and h < 2.5 * w: return False
    return True

def box_corners(c):
    """The 8 corners of a candidate's oriented box (viewer coordinates): local x = width, y = up, z = depth (outward)."""
    import numpy as np
    w, h, d = c['size_w'] / 2, c['size_h'] / 2, c['size_d'] / 2; cy, sy = math.cos(c['yaw']), math.sin(c['yaw'])
    a, b, e = (np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * w, np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * h, np.array([-1, 1, -1, 1, -1, 1, -1, 1]) * d)
    x, y, z = c['position']
    return np.stack([x + a * cy + e * sy, y + b, z - a * sy + e * cy], 1)

def fit_to_detections(c, views, poses, cameras, M, rounds=4, move_centre=True):
    """Final adjustment against the photos: optionally move the centre to where the detection-box centres
    triangulate, then scale the projected outline to the detection boxes. Irregular structural classes can keep
    the dense surface centre while still using the photos for their extent."""
    import numpy as np
    M = np.asarray(M, float); s = float(np.linalg.norm(M[:3, 0])); Rm, T = M[:3, :3] / s, M[:3, 3]
    to_model = lambda V: ((np.atleast_2d(V) - T) @ Rm) / s; to_view = lambda X: s * (np.atleast_2d(X) @ Rm.T) + T
    obs = []
    for name, d in views:
        pose = poses.get(name)
        if pose is None or pose[3] not in cameras: continue
        cam = cameras[pose[3]]; x1, x2, y1, y2 = _box_px(d, cam)
        if x2 - x1 > 1 and y2 - y1 > 1: obs.append((pose, cam, np.array([(x1 + x2) / 2, (y1 + y2) / 2]), x2 - x1, y2 - y1))
    if len(obs) < 2: return c
    def residuals(X):  # box-centre reprojection error, in box sizes so near and far photos weigh the same
        out = []
        for pose, cam, ctr, bw, bh in obs:
            uv, z = project(np.atleast_2d(X), pose[4], pose[5], cam)
            out.append((uv[0] - ctr) / np.array([bw, bh]) if z[0] > 0 else np.array([1.0, 1.0]))
        return np.concatenate(out)
    if move_centre:
        X = to_model(c['position'])[0]; start = X.copy(); eps = 1e-3 * max(c['size_w'], c['size_h']) / s
        for _ in range(10):  # Gauss-Newton, numeric Jacobian (3 unknowns)
            r = residuals(X); J = np.stack([(residuals(X + eps * e) - r) / eps for e in np.eye(3)], 1)
            try: step = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError: break
            X = X + step
            if np.linalg.norm(step) < eps: break
        moved = float(np.linalg.norm(X - start)) * s
        if np.isfinite(X).all() and moved <= max(c['size_w'], c['size_h'], c['size_d']):  # a big jump means bad triangulation: keep the fit
            c['position'] = to_view(X)[0]
    def outlines():  # projected box outline per photo
        corners = to_model(box_corners(c))
        for pose, cam, ctr, bw, bh in obs:
            uv, z = project(corners, pose[4], pose[5], cam)
            if (z > 0).all(): yield uv.min(0), uv.max(0), ctr, bw, bh
    for _ in range(rounds):
        spans = [(hi - lo) / (bw, bh) for lo, hi, _, bw, bh in outlines()]
        if not spans: break
        fw, fh = np.clip(np.median(spans, 0), .5, 2)
        c.update(size_w=c['size_w'] / fw, size_d=c['size_d'] / fw, size_h=c['size_h'] / fh)
    ious = []
    for lo, hi, ctr, bw, bh in outlines():
        a, b = ctr - (bw / 2, bh / 2), ctr + (bw / 2, bh / 2); inter = np.prod(np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None))
        ious.append(inter / (bw * bh + np.prod(hi - lo) - inter))
    c['fit_iou'] = float(np.median(ious)) if ious else 0.0
    return c
