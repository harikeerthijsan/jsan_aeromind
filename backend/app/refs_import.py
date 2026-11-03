"""Import manufacturer reference photos listed in a manifest CSV into VENDOR_REFERENCE_DIR.

Manifest columns (as in the multiview audit package): category, manufacturer, model, view, url, page, status, notes.
Each image is downloaded, checked to decode, shrunk to at most 1024 px and saved as
    <VENDOR_REFERENCE_DIR>/<class>/<Manufacturer>/<model>__<view>.jpg
with every row (including failures) recorded in <VENDOR_REFERENCE_DIR>/references.csv.

Run from backend/:  python -m app.refs_import "C:/Users/me/Downloads/tower_multiview_corrected/verified_and_partial_manifest.csv"
Seller/manufacturer images may carry usage restrictions; keep the library internal (it is git-ignored).
"""
import argparse, csv, io, re, urllib.request
from pathlib import Path
from . import vendor

CLASS_OF = {'sector_antennas': 'sector_antenna', 'sector_antenna': 'sector_antenna', 'microwave_dishes': 'microwave_dish', 'microwave_dish': 'microwave_dish'}
# manifest spellings -> names used in the manufacturer picker
MAKER = {'Aviat_Networks': 'Aviat', 'Cambium_Networks': 'Cambium', 'Ericsson_Kathrein': 'Kathrein'}
FIELDS = ['class', 'manufacturer', 'model', 'view', 'file', 'match', 'status', 'source_url', 'page', 'notes']

def _safe(s): return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or 'x'

def full_size_url(url):
    # the audit's ANDREW gallery links /thumbnail previews (60x90 px); /medium on the same asset is 600x900
    return url[:-len('/thumbnail')] + '/medium' if url.endswith('/thumbnail') else url

def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AeroMind-reference-import', 'Accept': 'image/*'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

def import_manifest(manifest, dest=None, force=False, log=print):
    from PIL import Image
    dest = Path(dest or vendor.VENDOR_REFERENCE_DIR); rows, done = [], set()
    index = dest / 'references.csv'
    if index.is_file() and not force:  # keep earlier imports from other manifests
        with index.open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
    with open(manifest, encoding='utf-8-sig', newline='') as f: items = list(csv.DictReader(f))
    for r in items:
        cls = CLASS_OF.get((r.get('category') or '').strip().lower())
        maker = MAKER.get(r['manufacturer'].strip(), r['manufacturer'].strip().replace('_', ' '))
        model, view = r.get('model', '').strip(), (r.get('view') or 'unknown').strip()
        rec = {'class': cls or r.get('category', ''), 'manufacturer': maker, 'model': model, 'view': view, 'file': '', 'match': '',
               'status': '', 'source_url': r['url'], 'page': r.get('page', ''), 'notes': ' '.join(x for x in (r.get('status', ''), r.get('notes', '')) if x)}
        rows = [x for x in rows if x['source_url'] != r['url']]
        if not cls: rec['status'] = 'skipped: unknown category'; rows.append(rec); continue
        rel = Path(cls) / _safe(maker) / f'{_safe(model)}__{_safe(view)}.jpg'; target = dest / rel
        rec['file'], rec['match'] = rel.as_posix(), 'yes' if vendor.view_matchable(view) else 'no'
        if target.is_file() and not force: rec['status'] = 'ok (already imported)'; rows.append(rec); continue
        try:
            im = Image.open(io.BytesIO(fetch(full_size_url(r['url'])))); im.load(); im = im.convert('RGB'); im.thumbnail((1024, 1024))
            if min(im.size) < 80: raise ValueError(f'image too small ({im.width}x{im.height})')
            target.parent.mkdir(parents=True, exist_ok=True); im.save(target, 'JPEG', quality=92)
            rec['status'] = 'ok'; log(f'ok    {cls}/{maker} {model} [{view}] {im.width}x{im.height}' + ('' if rec['match'] == 'yes' else '  (reviewer gallery only: view not used for matching)'))
        except Exception as e:
            rec['file'] = ''; rec['status'] = f'failed: {type(e).__name__}: {str(e)[:120]}'; log(f'FAIL  {cls}/{maker} {model} [{view}] {rec["status"]}')
        rows.append(rec)
    dest.mkdir(parents=True, exist_ok=True)
    with index.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows({k: x.get(k, '') for k in FIELDS} for x in rows)
    ok = sum(x['status'].startswith('ok') for x in rows)
    log(f'{ok}/{len(rows)} references available in {dest} (index: {index.name})')
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m app.refs_import', description='Download manufacturer reference photos listed in a manifest CSV.')
    ap.add_argument('manifest', help='CSV with category, manufacturer, model, view, url, page, status, notes')
    ap.add_argument('--dest', help=f'reference library folder (default {vendor.VENDOR_REFERENCE_DIR})')
    ap.add_argument('--force', action='store_true', help='re-download images that already exist')
    a = ap.parse_args(argv); import_manifest(a.manifest, a.dest, a.force)

if __name__ == '__main__':
    main()
