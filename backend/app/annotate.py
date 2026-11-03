"""Draw detection boxes onto a copy of an image while keeping the source image's metadata.

YOLO's own `save=True` output (and any plain cv2/Pillow re-encode) drops EXIF, GPS, maker notes, DJI XMP
(gimbal/flight attitude), ICC and IPTC. Here JPEG metadata segments are copied byte-for-byte from the source
into the re-encoded file, so nothing is reinterpreted or lost. PNG/TIFF keep their text/XMP, EXIF and ICC via Pillow.

CLI (replacement for `model.predict(save=True)`), run from backend/:
    python -m app.annotate D:/survey/raw --out D:/survey/annotated [--model best.pt] [--conf 0.25]
"""
import argparse, io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin, PngImagePlugin

PALETTE = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000','#aaffc3']  # same as the web viewer
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
# EXIF orientation -> transpose that makes the image upright; 6 and 8 are each other's inverse, the rest self-inverse
_UPRIGHT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
_BACK = {**_UPRIGHT, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}
# descriptive TIFF tags worth carrying over (incl. XMP 700, IPTC 33723); structural ones are rewritten by Pillow
_TIFF_KEEP = {269, 270, 271, 272, 274, 282, 283, 296, 305, 306, 315, 316, 700, 33432, 33723}

def jpeg_segments(data: bytes):
    """[(marker, raw segment bytes)] for the header segments of a JPEG, and the offset where scan data starts."""
    if data[:2] != b'\xff\xd8': raise ValueError('not a JPEG file')
    i, out = 2, []
    while i + 4 <= len(data):
        if data[i] != 0xFF: raise ValueError(f'corrupt JPEG header at byte {i}')
        m = data[i + 1]
        if m == 0xFF: i += 1; continue  # fill byte
        if m in (0xDA, 0xD9): break  # start of scan / end of image
        n = int.from_bytes(data[i + 2:i + 4], 'big')
        out.append((m, data[i:i + 2 + n])); i += 2 + n
    return out, i

def _is_metadata(m: int, seg: bytes, keep_icc: bool) -> bool:
    if m == 0xFE: return True  # COM
    if not 0xE0 <= m <= 0xEF: return False  # quantisation/huffman/frame tables belong to the new encode
    body = seg[4:]
    if m == 0xE2 and body.startswith(b'MPF\0'): return False  # multi-picture index: its offsets point into the old file
    if m == 0xE2 and body.startswith(b'ICC_PROFILE\0'): return keep_icc
    if m == 0xEE and body.startswith(b'Adobe'): return False  # colour-transform flag of the old encoding, not metadata
    return True  # JFIF, EXIF (incl. GPS + MakerNote), XMP, extended XMP, IPTC, DJI/vendor APPn

def splice_jpeg_metadata(source: bytes, encoded: bytes, keep_icc=True) -> bytes:
    """`encoded` with its own APPn/COM segments replaced by the metadata segments of `source`."""
    src, _ = jpeg_segments(source)
    new, sos = jpeg_segments(encoded)
    meta = [s for m, s in src if _is_metadata(m, s, keep_icc)]
    tables = [s for m, s in new if not (0xE0 <= m <= 0xEF or m == 0xFE)]
    return b'\xff\xd8' + b''.join(meta + tables) + encoded[sos:]

def _font(size):
    try: return ImageFont.load_default(size=size)
    except (TypeError, OSError, ImportError): return ImageFont.load_default()  # Pillow without FreeType: fixed bitmap font

def _hex(c): return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))

def draw_boxes(im: Image.Image, boxes, labels=True):
    d = ImageDraw.Draw(im); W, H = im.size
    lw, fs = max(2, round(max(W, H) / 640)), max(12, round(max(W, H) / 120)); font, pad = _font(fs), max(2, fs // 6)
    boxes = sorted(boxes, key=lambda b: (b['x2'] - b['x1']) * (b['y2'] - b['y1']), reverse=True)  # small parts drawn last, on top
    for b in boxes:
        d.rectangle([b['x1'], b['y1'], b['x2'], b['y2']], outline=_hex(PALETTE[b['class_id'] % len(PALETTE)]), width=lw)
    if not labels: return im
    for b in boxes:
        c = _hex(PALETTE[b['class_id'] % len(PALETTE)]); conf = b.get('confidence')
        text = f"{b['class_name']} {conf:.2f}" if conf is not None else b['class_name']
        if b.get('manufacturer') and b['manufacturer'] != 'Unknown': text += f" · {b['manufacturer']}" + (f" {b['product_model']}" if b.get('product_model') else '')
        if b.get('condition_grade') in ('Fair', 'Poor', 'Critical'): text += f" · {b['condition_grade']}"
        l, t, r, btm = d.textbbox((0, 0), text, font=font); tw, th = r - l + 2 * pad, btm - t + 2 * pad
        x, y = min(max(b['x1'], 0), max(W - tw, 0)), b['y1'] - th if b['y1'] - th >= 0 else b['y1']
        d.rectangle([x, y, x + tw, y + th], fill=c)
        d.text((x + pad - l, y + pad - t), text, font=font, fill=(0, 0, 0) if sum(c) > 380 else (255, 255, 255))
    return im

def box_frame(src, seen_size):
    """How boxes measured on a seen_size=(w, h) image map onto `src`: (orientation to make upright or None, sx, sy).
    ultralytics applies EXIF orientation, so by default boxes are in the upright (display) frame."""
    orientation = src.getexif().get(0x0112, 1); upright = src.size[::-1] if orientation in (5, 6, 7, 8) else src.size
    if seen_size and tuple(seen_size) == src.size and upright != src.size: return None, 1.0, 1.0  # boxes already in stored orientation
    sx, sy = (upright[0] / seen_size[0], upright[1] / seen_size[1]) if seen_size else (1.0, 1.0)  # != 1 for boxes from a resized copy
    return (orientation if orientation in _UPRIGHT else None), sx, sy

def crop(path, box, seen_size=None, pad=0.15):
    """Upright RGB crop of one box from the original file, with `pad` context on each side."""
    with Image.open(path) as src:
        turn, sx, sy = box_frame(src, seen_size); im = src.convert('RGB')
        if turn: im = im.transpose(_UPRIGHT[turn])
        x1, x2, y1, y2 = box['x1'] * sx, box['x2'] * sx, box['y1'] * sy, box['y2'] * sy; px, py = (x2 - x1) * pad, (y2 - y1) * pad
        return im.crop((max(0, round(x1 - px)), max(0, round(y1 - py)), min(im.width, round(x2 + px)), min(im.height, round(y2 + py))))

def render(path, boxes, seen_size=None, labels=True):
    """Return (bytes, mime) for `path` with `boxes` drawn on it and its metadata preserved.

    boxes: dicts with class_id, class_name, x1, y1, x2, y2 and optional confidence, in pixels of the image
    as the detector saw it. seen_size=(w, h) of that image; ultralytics applies EXIF orientation, so by
    default boxes are taken to be in the upright (display) frame. Pixels are written back in the stored
    orientation so the original Orientation tag and every other metadata field stay valid unchanged.
    """
    path = Path(path); raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as src:
        src.load(); fmt = src.format; turn, sx, sy = box_frame(src, seen_size)
        boxes = [{**b, 'x1': b['x1'] * sx, 'x2': b['x2'] * sx, 'y1': b['y1'] * sy, 'y2': b['y2'] * sy} for b in boxes]
        alpha = src.mode in ('RGBA', 'LA', 'PA') or (src.mode == 'P' and 'transparency' in src.info)
        keep_icc = src.mode in ('RGB', 'RGBA', 'P')  # grey/CMYK profiles don't describe the RGB pixels written here
        im = src.convert('RGBA' if alpha and fmt != 'JPEG' else 'RGB')
        if turn: im = im.transpose(_UPRIGHT[turn])
        draw_boxes(im, boxes, labels)
        if turn: im = im.transpose(_BACK[turn])
        out = io.BytesIO(); icc = src.info.get('icc_profile') if keep_icc else None
        if fmt == 'JPEG':
            kw = {'quality': 95, 'progressive': bool(src.info.get('progressive'))}
            if getattr(src, 'quantization', None):  # re-use the camera's quantisation so size/quality stay comparable
                kw.update(qtables=src.quantization, subsampling=max(JpegImagePlugin.get_sampling(src), 0)); del kw['quality']
            try: im.save(out, 'JPEG', **kw)
            except (ValueError, TypeError): out = io.BytesIO(); im.save(out, 'JPEG', quality=95)
            return splice_jpeg_metadata(raw, out.getvalue(), keep_icc), 'image/jpeg'
        if fmt == 'PNG':
            info = PngImagePlugin.PngInfo()
            for k, v in getattr(src, 'text', {}).items():  # tEXt/iTXt incl. XMP (XML:com.adobe.xmp)
                try: v.encode('latin-1'); info.add_text(k, v)
                except UnicodeEncodeError: info.add_itxt(k, v)
            kw = {k: v for k, v in {'exif': src.info.get('exif'), 'icc_profile': icc, 'dpi': src.info.get('dpi')}.items() if v}
            im.save(out, 'PNG', pnginfo=info, **kw); return out.getvalue(), 'image/png'
        if fmt == 'TIFF':
            exif, src_exif = Image.Exif(), src.getexif()
            for k, v in src.tag_v2.items():
                if k in _TIFF_KEEP: exif[k] = v
            subs = {t: dict(src_exif.get_ifd(t)) for t in (0x8769, 0x8825)}  # Exif and GPS sub-IFDs
            for t, d in subs.items():
                if d: exif.get_ifd(t).update(d); exif[t] = 0  # Pillow only writes a sub-IFD whose tag is in the main IFD
            # libtiff (used for compressed output) can't write sub-IFDs, so metadata wins over compression
            comp = src.info.get('compression') if not any(subs.values()) and src.info.get('compression') in ('tiff_lzw', 'tiff_deflate', 'tiff_adobe_deflate', 'packbits') else None
            kw = {k: v for k, v in {'icc_profile': icc, 'compression': comp}.items() if v}
            im.save(out, 'TIFF', exif=exif, **kw); return out.getvalue(), 'image/tiff'
        # other formats: best effort with whatever Pillow can write back
        kw = {k: v for k, v in {'exif': src.info.get('exif'), 'icc_profile': icc, 'xmp': src.info.get('xmp')}.items() if v}
        im.save(out, fmt, **kw); return out.getvalue(), Image.MIME.get(fmt, 'application/octet-stream')

def main(argv=None):
    from . import detector
    ap = argparse.ArgumentParser(prog='python -m app.annotate', description='Run the YOLO detector on images and write annotated copies that keep each source image\'s metadata.')
    ap.add_argument('source', type=Path, help='image file or folder of images')
    ap.add_argument('--out', type=Path, required=True, help='folder for the annotated images (created if missing)')
    ap.add_argument('--model', default=detector.MODEL_PATH, help='YOLO weights (default: YOLO_MODEL_PATH)')
    ap.add_argument('--conf', type=float, default=detector.CONF, help=f'confidence threshold (default {detector.CONF})')
    ap.add_argument('--recursive', action='store_true', help='include sub-folders')
    ap.add_argument('--no-labels', action='store_true', help='draw boxes only')
    a = ap.parse_args(argv)
    if not a.model: ap.error('set --model or YOLO_MODEL_PATH')
    detector.MODEL_PATH, detector.CONF = a.model, a.conf
    files = [a.source] if a.source.is_file() else sorted(p for p in (a.source.rglob('*') if a.recursive else a.source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files: ap.error(f'no images found in {a.source}')
    a.out.mkdir(parents=True, exist_ok=True)
    for n, p in enumerate(files, 1):
        rel = p.relative_to(a.source) if a.source.is_dir() else Path(p.name); target = a.out / rel
        if target.resolve() == p.resolve(): ap.error(f'--out would overwrite the original {p}')
        target.parent.mkdir(parents=True, exist_ok=True)
        boxes = []; seen = None
        for _, w, h, bs in detector.detect(p): boxes, seen = bs, (w, h)
        target.write_bytes(render(p, boxes, seen, labels=not a.no_labels)[0])
        print(f'[{n}/{len(files)}] {rel}: {len(boxes)} boxes -> {target}')

if __name__ == '__main__':
    main()
