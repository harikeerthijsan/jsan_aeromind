import os, threading
from importlib.util import find_spec
from pathlib import Path
MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '')
IMGSZ = int(os.getenv('YOLO_IMGSZ', '1920'))
CONF = float(os.getenv('YOLO_CONF', '0.25'))
IOU = float(os.getenv('YOLO_IOU', '0.7'))
MAX_DET = int(os.getenv('YOLO_MAX_DET', '1000'))
DEVICE = os.getenv('YOLO_DEVICE') or None
VID_STRIDE = max(1, int(os.getenv('YOLO_VID_STRIDE', '30')))
_model = None
_lock = threading.Lock()

def model_name():
    p = Path(MODEL_PATH)
    return f'{p.parent.parent.name}/{p.name}' if p.parent.name == 'weights' else p.name

def status():
    if not MODEL_PATH: return 'not_configured'
    if not Path(MODEL_PATH).is_file(): return 'model_missing'
    if not find_spec('ultralytics'): return 'dependency_missing'
    return 'ready'

def _load():
    global _model
    if _model is None:
        if (s := status()) != 'ready': raise RuntimeError(f'YOLO detector {s}')
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH, task='detect')
    return _model

def info(load=False):
    d = {'status': status(), 'model': model_name() if MODEL_PATH else None, 'model_path': MODEL_PATH or None, 'imgsz': IMGSZ, 'conf': CONF, 'iou': IOU, 'max_det': MAX_DET, 'device': DEVICE or 'auto', 'vid_stride': VID_STRIDE, 'loaded': _model is not None}
    if load and d['status'] == 'ready':
        with _lock: d['classes'] = _load().names
        d['loaded'] = True
    return d

def class_names():
    """{class_id: name} from the model, or {} when it can't load. Once loaded, doesn't wait on a running detection."""
    try:
        if _model is None and status() == 'ready':
            with _lock: _load()
    except Exception:
        return {}
    return {int(k): v for k, v in _model.names.items()} if _model is not None else {}

def detect(path, video=False):
    """Yield (frame_index, width, height, boxes) per image or sampled video frame. Boxes are pixel xyxy.
    path may also be an encoded image in memory (bytes): photos read straight from the bucket never touch the disk."""
    if isinstance(path, (bytes, bytearray, memoryview)):
        import cv2, numpy as np
        source = cv2.imdecode(np.frombuffer(path, np.uint8), cv2.IMREAD_COLOR)
        if source is None: raise ValueError('not a readable image')
    else: source = str(path)
    with _lock:
        results = _load().predict(source=source, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET, device=DEVICE, vid_stride=VID_STRIDE, stream=True, verbose=False)
        for i, r in enumerate(results):
            h, w = r.orig_shape
            # ultralytics grabs vid_stride frames before each retrieve, so sample i is source frame (i+1)*stride-1
            frame = (i + 1) * VID_STRIDE - 1 if video else 0
            boxes = [{'class_id': int(k), 'class_name': r.names[int(k)], 'confidence': round(c, 4), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                     for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            yield frame, w, h, boxes
