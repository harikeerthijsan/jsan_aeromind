"""Real-world scale of a photogrammetry model from the drone's own positions.

COLMAP models have an arbitrary scale: the viewer's units are not metres. Drone photos record where they were taken
(EXIF GPS, and on DJI aircraft XMP drone-dji:GpsLatitude/... with RTK status). Fitting the reconstructed camera
centres to those positions with a similarity transform (Umeyama, least squares) gives metres per model unit; the
residuals show how well they agree. With RTK-fixed positions (drone-dji:RtkFlag=50, ~1-2 cm) the scale is typically
good to a fraction of a percent; plain GPS is metres-noisy, so a wide flight helps and the uncertainty is reported.
"""
from __future__ import annotations

import math, re
from pathlib import Path

EARTH_R = 6378137.0
MIN_CAMERAS = 4
MIN_SPREAD_M = 3.0            # the flight must span at least this much for a meaningful scale
_XMP = re.compile(rb'drone-dji:(GpsLatitude|GpsLongitude|AbsoluteAltitude|RelativeAltitude|RtkFlag)="([^"]*)"')

def _ratio(v):
    try: return float(v.numerator) / float(v.denominator)
    except AttributeError: return float(v)

def read_position(path) -> dict | None:
    """{'lat', 'lon', 'alt', 'rtk'} of one photo, or None without a position. DJI XMP first (more digits, RTK status), then EXIF GPS."""
    if isinstance(path, (bytes, bytearray)): head = bytes(path)  # the file's first bytes (storage.read_head)
    else:
        p = Path(path)
        if not p.is_file(): return None
        with open(p, 'rb') as f: head = f.read(512 * 1024)  # EXIF and XMP sit in the first APP segments
    if not head: return None
    x = {k.decode(): v.decode() for k, v in _XMP.findall(head)}
    try:
        if 'GpsLatitude' in x and 'GpsLongitude' in x:
            rtk = x.get('RtkFlag') == '50'  # 50 = RTK fixed solution
            alt = x.get('AbsoluteAltitude') if rtk else x.get('RelativeAltitude') or x.get('AbsoluteAltitude')  # barometric relative height is steadier than plain GPS height
            lat, lon = float(x['GpsLatitude']), float(x['GpsLongitude'])
            if (lat, lon) != (0.0, 0.0): return {'lat': lat, 'lon': lon, 'alt': float(alt) if alt else None, 'rtk': rtk}
    except ValueError: pass
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(head)) as im: g = im.getexif().get_ifd(0x8825)
        if 2 in g and 4 in g:
            dms = lambda v: _ratio(v[0]) + _ratio(v[1]) / 60 + _ratio(v[2]) / 3600
            lat = dms(g[2]) * (-1 if g.get(1) == 'S' else 1); lon = dms(g[4]) * (-1 if g.get(3) == 'W' else 1)
            alt = _ratio(g[6]) * (-1 if g.get(5) in (1, b'\x01') else 1) if 6 in g else None
            if (lat, lon) != (0.0, 0.0): return {'lat': lat, 'lon': lon, 'alt': alt, 'rtk': False}
    except Exception: pass
    return None

def estimate(cameras, positions) -> dict:
    """cameras: [{'name', 'center'}] in viewer units (web/cameras.json); positions: name -> read_position(...).
    Returns {'m_per_unit', 'method', 'cameras', 'rtk', 'residual_median_m', 'residual_max_m', 'uncertainty_pct', 'spread_m'}."""
    import numpy as np
    pairs = [(np.asarray(c['center'], float), positions[c['name']]) for c in cameras if positions.get(c['name'])]
    pairs = [(c, p) for c, p in pairs if p['alt'] is not None] or [(c, {**p, 'alt': 0.0}) for c, p in pairs]  # 2D fallback without heights
    if len(pairs) < MIN_CAMERAS: raise ValueError(f'only {len(pairs)} photo(s) of this model have a GPS position; need {MIN_CAMERAS}')
    lat0, lon0 = pairs[0][1]['lat'], pairs[0][1]['lon']; k = math.cos(math.radians(lat0))
    E = np.array([[math.radians(p['lon'] - lon0) * EARTH_R * k, math.radians(p['lat'] - lat0) * EARTH_R, p['alt']] for _, p in pairs])  # local east/north/up, metres
    C = np.array([c for c, _ in pairs])
    spread = float(np.linalg.norm(np.ptp(E, axis=0)))
    if spread < MIN_SPREAD_M: raise ValueError(f'the photos were taken within {spread:.1f} m of each other; too close to tell the scale')
    Cc, Ec = C - C.mean(0), E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec.T @ Cc / len(C)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    s = float(np.trace(np.diag(S) @ D) / ((Cc ** 2).sum() / len(C)))  # Umeyama: metres per model unit
    res = np.linalg.norm(Ec - s * Cc @ (U @ D @ Vt).T, axis=1)
    i, j = np.triu_indices(len(C), 1); dg, dc = np.linalg.norm(E[i] - E[j], axis=1), np.linalg.norm(C[i] - C[j], axis=1)
    ok = (dg > max(2.0, spread / 10)) & (dc > 1e-9); ratio = dg[ok] / dc[ok]
    unc = float(1.4826 * np.median(np.abs(ratio - np.median(ratio))) / np.median(ratio) * 100 / math.sqrt(max(1, len(C) - 3))) if len(ratio) else None
    if not math.isfinite(s) or s <= 0: raise ValueError('the camera positions could not be matched to the GPS track')
    rtk = all(p['rtk'] for _, p in pairs)
    return {'m_per_unit': round(s, 6), 'method': 'gps_rtk' if rtk else 'gps', 'cameras': len(C), 'rtk': rtk,
            'residual_median_m': round(float(np.median(res)), 3), 'residual_max_m': round(float(res.max()), 3),
            'uncertainty_pct': round(unc, 3) if unc is not None else None, 'spread_m': round(spread, 1)}
