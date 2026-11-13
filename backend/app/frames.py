"""Turn an uploaded video into still frames so they go through the same review flow as photos.

Every VIDEO_FRAME_STRIDE-th frame is kept (default: YOLO_VID_STRIDE, i.e. ~1 per second of 30 fps footage); the stride
grows for long videos so no more than VIDEO_MAX_FRAMES frames are written. Frames are JPEGs beside their inspection at
<UPLOAD_DIR>/sites/<tower>/inspections/<inspection>/frames/MEDIA-<video id>/ and become Media rows pointing at the video
(parent_media_id, frame_index, frame_time). Legacy flat uploads continue using frames/<video id>/.
Within each stride-long window the sharpest of a few evenly spaced candidates is kept (VIDEO_SHARP_CANDIDATES, 1 = always
the window's first frame), so a frame caught mid-turn or mid-shake is passed over. Drone telemetry recorded in the video
(see video_meta) is written into each frame as DJI photo XMP: position, heights and gimbal angles at that moment.
"""
import math, os
from pathlib import Path

STRIDE = max(1, int(os.getenv('VIDEO_FRAME_STRIDE', os.getenv('YOLO_VID_STRIDE', '30'))))
MAX_FRAMES = max(1, int(os.getenv('VIDEO_MAX_FRAMES', '300')))
SHARP_CANDIDATES = max(1, int(os.getenv('VIDEO_SHARP_CANDIDATES', '5')))

def plan(total_frames, stride=None, max_frames=None):
    """Frame indices to keep: 0, s, 2s ... with s widened so at most max_frames are taken."""
    stride, max_frames = stride or STRIDE, max_frames or MAX_FRAMES
    if total_frames <= 0: return []
    s = max(stride, math.ceil(total_frames / max_frames))
    return list(range(0, total_frames, s))

def _offsets(window, n):
    """Candidate positions inside a window: n evenly spaced, always including its first frame."""
    if n <= 1 or window <= 1: return {0}
    return {round(j * (window - 1) / (n - 1)) for j in range(min(n, window))}

def sharpness(frame):
    """Variance of the Laplacian on a ~960 px grey copy: higher = crisper edges (less motion blur)."""
    import cv2
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); k = 960 / max(g.shape)
    if k < 1: g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

def extract(video_path, out_dir, stride=None, max_frames=None, quality=92, candidates=None, telemetry=None):
    """Write one JPEG per window (the sharpest candidate). Returns [(frame_index, seconds, path, width, height)].
    telemetry: video_meta samples to stamp on the frames; None reads them from the video, [] skips. Needs OpenCV (installed with ultralytics)."""
    import cv2
    from . import video_meta
    stride, max_frames, n = stride or STRIDE, max_frames or MAX_FRAMES, candidates or SHARP_CANDIDATES
    if telemetry is None: telemetry = video_meta.read(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video {Path(video_path).name}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # some containers report no frame count: then windows of `stride`
        s = max(stride, math.ceil(total / max_frames)) if total > 0 else stride
        offsets = _offsets(s, n); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); out, best, i = [], None, 0
        def keep(b):
            _, idx, frame = b; sec = idx / fps; p = out_dir / f'{idx:06d}.jpg'
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality]); data = buf.tobytes()
            if (t := video_meta.at(telemetry, sec)): data = video_meta.with_xmp(data, video_meta.xmp(t))
            p.write_bytes(data); out.append((idx, round(sec, 3), p, frame.shape[1], frame.shape[0]))
        while True:
            win, off = divmod(i, s)
            if best is not None and best[1] // s != win: keep(best); best = None
            if win >= max_frames: break
            if off not in offsets:
                if not cap.grab(): break  # skip without decoding to an image
                i += 1; continue
            ok, frame = cap.read()
            if not ok: break
            score = sharpness(frame) if len(offsets) > 1 else 0.0
            if best is None or score > best[0]: best = (score, i, frame)
            i += 1
        if best is not None: keep(best)
        return out
    finally:
        cap.release()

def label(video_name, frame_index, seconds):
    stem = Path(video_name).stem
    return f'{stem}_f{frame_index:06d}.jpg', f'{int(seconds // 60):02d}:{seconds % 60:04.1f}'
