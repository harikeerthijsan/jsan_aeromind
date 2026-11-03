"""Turn an uploaded video into still frames so they go through the same review flow as photos.

Every VIDEO_FRAME_STRIDE-th frame is kept (default: YOLO_VID_STRIDE, i.e. ~1 per second of 30 fps footage); the stride
grows for long videos so no more than VIDEO_MAX_FRAMES frames are written. Frames are JPEGs under
<UPLOAD_DIR>/frames/<video media id>/ and become Media rows pointing at the video (parent_media_id, frame_index, frame_time).
Drone telemetry (GPS/gimbal in DJI .SRT sidecars or video metadata) is not carried onto the frames.
"""
import math, os
from pathlib import Path

STRIDE = max(1, int(os.getenv('VIDEO_FRAME_STRIDE', os.getenv('YOLO_VID_STRIDE', '30'))))
MAX_FRAMES = max(1, int(os.getenv('VIDEO_MAX_FRAMES', '300')))

def plan(total_frames, stride=None, max_frames=None):
    """Frame indices to keep: 0, s, 2s ... with s widened so at most max_frames are taken."""
    stride, max_frames = stride or STRIDE, max_frames or MAX_FRAMES
    if total_frames <= 0: return []
    s = max(stride, math.ceil(total_frames / max_frames))
    return list(range(0, total_frames, s))

def extract(video_path, out_dir, stride=None, max_frames=None, quality=92):
    """Write the planned frames as JPEGs. Returns [(frame_index, seconds, path, width, height)]. Needs OpenCV (installed with ultralytics)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video {Path(video_path).name}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wanted = set(plan(total, stride, max_frames)) if total > 0 else None  # some containers report no frame count
        step = (stride or STRIDE) if wanted is None else None
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); out, i = [], 0
        while True:
            take = (i in wanted) if wanted is not None else (i % step == 0 and len(out) < (max_frames or MAX_FRAMES))
            if not take:
                if not cap.grab(): break  # skip without decoding
                i += 1; continue
            ok, frame = cap.read()
            if not ok: break
            p = out_dir / f'{i:06d}.jpg'
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            out.append((i, round(i / fps, 3), p, frame.shape[1], frame.shape[0])); i += 1
            if wanted is not None and i > max(wanted): break
        return out
    finally:
        cap.release()

def label(video_name, frame_index, seconds):
    stem = Path(video_name).stem
    return f'{stem}_f{frame_index:06d}.jpg', f'{int(seconds // 60):02d}:{seconds % 60:04.1f}'
