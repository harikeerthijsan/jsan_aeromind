import os, threading
from importlib.util import find_spec
from pathlib import Path
# One YOLO model per tower type, in YOLO_MODELS_DIR as <type>.pt (lattice.pt, monopole.pt). YOLO_MODEL_PATH stays the
# lattice model's fallback (it was the only model before tower types).
MODELS_DIR = Path(os.getenv('YOLO_MODELS_DIR', 'D:/TowerYOLO/models'))
MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '')
TOWER_TYPES = ('lattice', 'monopole')
DEFAULT_TYPE = 'lattice'
IMGSZ = int(os.getenv('YOLO_IMGSZ', '1920'))
CONF = float(os.getenv('YOLO_CONF', '0.25'))
IOU = float(os.getenv('YOLO_IOU', '0.5'))  # NMS IoU: overlapping boxes of a class above this are one detection
MAX_DET = int(os.getenv('YOLO_MAX_DET', '1000'))
DEVICE = os.getenv('YOLO_DEVICE') or None
VID_STRIDE = max(1, int(os.getenv('YOLO_VID_STRIDE', '30')))
_models: dict = {}
_lock = threading.Lock()

def tower_model_type(tower_type: str | None) -> str:
    """The model for a site's structure type: monopoles have their own; lattice, guyed masts and the rest use lattice."""
    return 'monopole' if (tower_type or '').strip().lower().startswith('monopole') else 'lattice'

def model_path(tower_type: str | None = None) -> str:
    t = tower_type if tower_type in TOWER_TYPES else DEFAULT_TYPE
    p = MODELS_DIR / f'{t}.pt'
    if p.is_file(): return str(p)
    return MODEL_PATH if t == DEFAULT_TYPE else ''

def model_name(tower_type: str | None = None):
    p = Path(model_path(tower_type) or '')
    return f'{p.parent.parent.name}/{p.name}' if p.parent.name == 'weights' else (f'{p.stem} model' if p.parent == MODELS_DIR else p.name)

def status(tower_type: str | None = None):
    p = model_path(tower_type)
    if not p: return 'not_configured'
    if not Path(p).is_file(): return 'model_missing'
    if not find_spec('ultralytics'): return 'dependency_missing'
    return 'ready'

def _load(tower_type: str | None = None):
    p = model_path(tower_type)
    if p not in _models:
        if (s := status(tower_type)) != 'ready': raise RuntimeError(f'YOLO detector for {tower_type or DEFAULT_TYPE} towers: {s}')
        from ultralytics import YOLO
        _models[p] = YOLO(p, task='detect')
    return _models[p]

def info(load=False):
    d = {'status': status(), 'model': model_name() if model_path() else None, 'model_path': model_path() or None, 'imgsz': IMGSZ, 'conf': CONF, 'iou': IOU,
         'max_det': MAX_DET, 'device': DEVICE or 'auto', 'vid_stride': VID_STRIDE, 'loaded': bool(_models),
         'models': {t: {'status': status(t), 'model': model_name(t) if model_path(t) else None} for t in TOWER_TYPES}}
    if load and d['status'] == 'ready':
        with _lock: d['classes'] = _load().names
        d['loaded'] = True
    return d

def class_names():
    """{class_id: name} from the default model, or {} when it can't load. Once loaded, doesn't wait on a running detection."""
    try:
        if not _models and status() == 'ready':
            with _lock: _load()
    except Exception:
        return {}
    m = _models.get(model_path())
    return {int(k): v for k, v in m.names.items()} if m is not None else {}

def detect(path, video=False, tower_type: str | None = None):
    """Yield (frame_index, width, height, boxes) per image or sampled video frame. Boxes are pixel xyxy.
    path may also be an encoded image in memory (bytes): photos read straight from the bucket never touch the disk.
    tower_type picks the model (lattice / monopole)."""
    if isinstance(path, (bytes, bytearray, memoryview)):
        import cv2, numpy as np
        source = cv2.imdecode(np.frombuffer(path, np.uint8), cv2.IMREAD_COLOR)
        if source is None: raise ValueError('not a readable image')
    else: source = str(path)
    with _lock:
        results = _load(tower_type).predict(source=source, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET, device=DEVICE, vid_stride=VID_STRIDE, stream=True, verbose=False)
        for i, r in enumerate(results):
            h, w = r.orig_shape
            # ultralytics grabs vid_stride frames before each retrieve, so sample i is source frame (i+1)*stride-1
            frame = (i + 1) * VID_STRIDE - 1 if video else 0
            boxes = [{'class_id': int(k), 'class_name': r.names[int(k)], 'confidence': round(c, 4), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                     for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            yield frame, w, h, boxes
