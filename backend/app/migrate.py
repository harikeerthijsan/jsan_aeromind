"""Move a local AeroMind installation to the cloud deployment (run in backend/).

    python -m app.migrate db --target <postgres url> [--source sqlite:///tower_vision.db] [--replace]
        copies every table (sites, inspections, media, detections, reviews, annotations, 3D models, registrations ...)
    python -m app.migrate files
        uploads uploads/, reconstructions/ (without the images/ copies of uploaded photos) and vendor_refs/ to the
        bucket from the S3_* settings; files already there with the same size are skipped, so it can be re-run.
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from pathlib import Path

def copy_db(source, target, replace=False):
    from sqlalchemy import create_engine, inspect, select, text, DateTime
    from .database import Base, normalise_url
    from . import models  # noqa: F401  (registers the tables)
    src, dst = create_engine(normalise_url(source)), create_engine(normalise_url(target))
    Base.metadata.create_all(dst)
    have_src = inspect(src)
    with dst.begin() as conn:
        busy = [t.name for t in Base.metadata.sorted_tables if conn.execute(select(text('1')).select_from(t).limit(1)).first()]
        if busy and not replace: sys.exit(f'Target already has data in {", ".join(busy)}; use --replace to overwrite it')
        for t in reversed(Base.metadata.sorted_tables): conn.execute(t.delete())
    for t in Base.metadata.sorted_tables:
        if not have_src.has_table(t.name): print(f'{t.name:24s} (not in source)'); continue
        cols = [c for c in t.columns if c.name in {c['name'] for c in have_src.get_columns(t.name)}]
        with src.connect() as s: rows = [dict(r._mapping) for r in s.execute(select(*cols))]
        for r in rows:  # SQLite returns naive datetimes; they were stored as UTC
            for c in cols:
                if isinstance(c.type, DateTime) and r[c.name] is not None and r[c.name].tzinfo is None: r[c.name] = r[c.name].replace(tzinfo=timezone.utc)
        with dst.begin() as conn:
            for i in range(0, len(rows), 1000): conn.execute(t.insert(), rows[i:i + 1000])
            if dst.dialect.name == 'postgresql' and 'id' in t.columns:  # next ids continue after the copied ones
                conn.execute(text(f"SELECT setval(pg_get_serial_sequence('{t.name}', 'id'), COALESCE((SELECT MAX(id) FROM {t.name}), 1), (SELECT COUNT(*) FROM {t.name}) > 0)"))
        print(f'{t.name:24s} {len(rows):7d} rows')

def copy_files(workers=8):
    from . import storage, reconstruct, vendor
    upload_dir = Path(os.getenv('UPLOAD_DIR', './uploads')).resolve()
    if not storage.enabled(): sys.exit('No bucket configured: set S3_BUCKET, S3_ENDPOINT, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY')
    for prefix, folder in (('uploads', upload_dir), ('reconstructions', reconstruct.RECON_DIR), ('vendor_refs', vendor.VENDOR_REFERENCE_DIR)):
        storage.register(prefix, folder)
    jobs = []
    for prefix, folder in (('uploads', upload_dir), ('reconstructions', reconstruct.RECON_DIR), ('vendor_refs', vendor.VENDOR_REFERENCE_DIR)):
        if not folder.is_dir(): continue
        have = dict(storage._keys(prefix))
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or p.name.endswith('.part'): continue
            rel = p.relative_to(folder).parts
            if prefix == 'reconstructions' and len(rel) > 1 and rel[1] == 'images': continue  # copies of uploaded photos
            key = storage.key_for(p)
            if have.get(key) != p.stat().st_size: jobs.append((p, key))
    total = sum(p.stat().st_size for p, _ in jobs); done = 0; t0 = time.time()
    print(f'{len(jobs)} file(s), {total / 1e6:.0f} MB to upload to bucket {storage.BUCKET}')
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(storage.client().upload_file, str(p), storage.BUCKET, key): p for p, key in jobs}
        for n, f in enumerate(as_completed(futures), 1):
            f.result(); done += futures[f].stat().st_size
            if n % 25 == 0 or n == len(jobs): print(f'  {n}/{len(jobs)} files, {done / 1e6:.0f}/{total / 1e6:.0f} MB, {done / 1e6 / max(time.time() - t0, 1):.1f} MB/s', flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('db'); d.add_argument('--source', default='sqlite:///tower_vision.db'); d.add_argument('--target', required=True); d.add_argument('--replace', action='store_true')
    f = sub.add_parser('files'); f.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'db': copy_db(a.source, a.target, a.replace)
    else: copy_files(a.workers)

if __name__ == '__main__':
    main()
