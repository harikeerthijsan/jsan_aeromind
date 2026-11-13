"""Drone telemetry inside a video: where the drone was and where the gimbal pointed, for every frame.

DJI aircraft record it in the MP4 itself: newer models (Air 3/3S, Mini 4 Pro, Mavic 3 ...) as a per-frame protobuf
track ('djmd'), older ones as a subtitle text track. Each model lays the protobuf out differently, so ExifTool
(https://exiftool.org, Perl Artistic licence), which knows them all, does the decoding: `exiftool -ee3` gives one
sample per frame (SampleTime, GPSLatitude/Longitude, Absolute/RelativeAltitude, GimbalPitch/Yaw/Roll, DroneYaw).
ExifTool is found via EXIFTOOL_PATH, then PATH, then the usual Windows install folder; EXIFTOOL_PATH=off disables it.
Without it, or for a video with no telemetry, frames simply carry no position (as before).

Extracted frames get the values as DJI photo XMP (drone-dji:GpsLatitude ...), so the rest of the app reads a video
frame's position exactly like a DJI photo's (scale.read_position, photo_pose.read).
"""
from __future__ import annotations
import json, math, os, shutil, subprocess
from pathlib import Path

_CANDIDATES = [Path(os.path.expandvars(p)) for p in (r'%LOCALAPPDATA%\Programs\ExifTool\ExifTool.exe', r'%ProgramFiles%\ExifTool\ExifTool.exe')] + \
              [Path(d) / 'exiftool' / 'exiftool.exe' for d in ('D:/tools', 'C:/tools')]
_TAGS = ('SampleTime', 'GPSLatitude', 'GPSLongitude', 'AbsoluteAltitude', 'GPSAltitude', 'RelativeAltitude',
         'GimbalPitch', 'GimbalYaw', 'GimbalRoll', 'DroneYaw')
TIMEOUT = int(os.getenv('EXIFTOOL_TIMEOUT', '600'))  # s; a long 4K flight has tens of thousands of samples

def exiftool_exe():
    p = os.getenv('EXIFTOOL_PATH', '').strip().strip('"')
    if p.lower() == 'off': return ''
    if p: return p
    return shutil.which('exiftool') or next((str(c) for c in _CANDIDATES if c.is_file()), '')

def status():
    exe = exiftool_exe()
    if not exe: return 'not_found'
    return 'ready' if Path(exe).is_file() or shutil.which(exe) else 'missing'

def parse(doc: dict) -> list[dict]:
    """Samples from one file's `exiftool -j -n -ee3 -G3` record: [{'t', 'lat', 'lon', 'abs_alt', 'rel_alt', 'gimbal_pitch',
    'gimbal_yaw', 'gimbal_roll', 'drone_yaw'}] sorted by time; samples without a position are left out."""
    docs: dict[str, dict] = {}
    for key, v in doc.items():
        group, _, tag = key.rpartition(':')
        if tag in _TAGS and group.startswith('Doc') and isinstance(v, (int, float)): docs.setdefault(group, {})[tag] = float(v)
    out = []
    for d in docs.values():
        lat, lon = d.get('GPSLatitude'), d.get('GPSLongitude')
        if lat is None or lon is None or (lat, lon) == (0.0, 0.0) or 'SampleTime' not in d: continue
        out.append({'t': d['SampleTime'], 'lat': lat, 'lon': lon, 'abs_alt': d.get('AbsoluteAltitude', d.get('GPSAltitude')),
                    'rel_alt': d.get('RelativeAltitude'), 'gimbal_pitch': d.get('GimbalPitch'), 'gimbal_yaw': d.get('GimbalYaw'),
                    'gimbal_roll': d.get('GimbalRoll'), 'drone_yaw': d.get('DroneYaw')})
    return sorted(out, key=lambda s: s['t'])

def read(video_path) -> list[dict]:
    """Per-frame telemetry of a video (see parse), or [] without ExifTool / telemetry. Never raises."""
    exe = exiftool_exe()
    if not exe: return []
    try:
        r = subprocess.run([exe, '-j', '-n', '-ee3', '-G3', '-api', 'LargeFileSupport=1', *(f'-{t}' for t in _TAGS), str(video_path)],
                           capture_output=True, timeout=TIMEOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        docs = json.loads(r.stdout.decode('utf-8', 'replace') or '[]')
        return parse(docs[0]) if docs else []
    except (OSError, ValueError, subprocess.SubprocessError): return []

def _lerp_angle(a, b, f):
    if a is None or b is None: return a if b is None else b
    return (a + ((b - a + 180) % 360 - 180) * f + 180) % 360 - 180

def at(samples: list[dict], t: float) -> dict | None:
    """Telemetry at video time t (s), interpolated between the neighbouring samples; None outside the recorded span."""
    if not samples or t < samples[0]['t'] - 0.5 or t > samples[-1]['t'] + 0.5: return None
    import bisect
    i = bisect.bisect_right([s['t'] for s in samples], t)
    if i == 0: return dict(samples[0])
    if i == len(samples): return dict(samples[-1])
    a, b = samples[i - 1], samples[i]; f = (t - a['t']) / ((b['t'] - a['t']) or 1)
    lin = lambda k: a[k] + (b[k] - a[k]) * f if a[k] is not None and b[k] is not None else (a[k] if b[k] is None else b[k])
    return {'t': t, **{k: lin(k) for k in ('lat', 'lon', 'abs_alt', 'rel_alt', 'gimbal_pitch')},
            **{k: _lerp_angle(a[k], b[k], f) for k in ('gimbal_yaw', 'gimbal_roll', 'drone_yaw')}}

def position(s: dict | None) -> dict | None:
    """The sample as scale.read_position returns a photo's position (relative height: steadier than plain GPS height)."""
    if not s: return None
    alt = s['rel_alt'] if s['rel_alt'] is not None else s['abs_alt']
    return {'lat': s['lat'], 'lon': s['lon'], 'alt': alt, 'rtk': False}

_XMP_FIELDS = (('GpsLatitude', 'lat', '{:.9f}'), ('GpsLongitude', 'lon', '{:.9f}'), ('AbsoluteAltitude', 'abs_alt', '{:+.3f}'),
               ('RelativeAltitude', 'rel_alt', '{:+.3f}'), ('GimbalPitchDegree', 'gimbal_pitch', '{:+.2f}'),
               ('GimbalYawDegree', 'gimbal_yaw', '{:+.2f}'), ('GimbalRollDegree', 'gimbal_roll', '{:+.2f}'), ('FlightYawDegree', 'drone_yaw', '{:+.2f}'))

def xmp(s: dict) -> bytes:
    """DJI-style XMP packet for one frame (the same drone-dji names a DJI photo uses)."""
    attrs = ''.join(f' drone-dji:{name}="{fmt.format(s[k])}"' for name, k, fmt in _XMP_FIELDS if s.get(k) is not None)
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"{attrs}'
            f' drone-dji:VideoTime="{s["t"]:.3f}"/></rdf:RDF></x:xmpmeta>').encode('utf-8')

def with_xmp(jpeg: bytes, packet: bytes) -> bytes:
    """The JPEG with an XMP APP1 segment after its SOI/APP0 header."""
    seg = b'http://ns.adobe.com/xap/1.0/\x00' + packet
    if len(seg) + 2 > 0xFFFF or jpeg[:2] != b'\xff\xd8': return jpeg
    i = 2
    if jpeg[2:4] == b'\xff\xe0': i = 4 + int.from_bytes(jpeg[4:6], 'big')  # after JFIF
    return jpeg[:i] + b'\xff\xe1' + (len(seg) + 2).to_bytes(2, 'big') + seg + jpeg[i:]

def coverage(samples: list[dict]) -> dict:
    """Summary for logs/results: samples, time span and how far the drone moved (m)."""
    if not samples: return {'samples': 0}
    k = math.cos(math.radians(samples[0]['lat'])); R = 6378137.0
    xy = [(math.radians(s['lon']) * R * k, math.radians(s['lat']) * R) for s in samples]
    span = math.hypot(max(x for x, _ in xy) - min(x for x, _ in xy), max(y for _, y in xy) - min(y for _, y in xy))
    return {'samples': len(samples), 'seconds': round(samples[-1]['t'] - samples[0]['t'], 1), 'span_m': round(span, 1)}
