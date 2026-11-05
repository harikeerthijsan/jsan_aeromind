"""Move a local AeroMind installation to the cloud deployment (run in backend/).

    python -m app.migrate db --target <postgres url> [--source sqlite:///tower_vision.db] [--replace]
        copies every table (sites, inspections, media, detections, reviews, annotations, 3D models, registrations ...)
    python -m app.migrate files
        uploads uploads/, reconstructions/ (without the images/ copies of uploaded photos) and vendor_refs/ to the
        bucket from the S3_* settings; files already there with the same size are skipped, so it can be re-run.
    python -m app.migrate media-layout [--apply] [--delete-old]
        reorganises existing media into uploads/sites/<tower>/inspections/<inspection>/{images|videos|frames}/ locally
        and in the configured bucket. The default is a dry run; --delete-old removes old objects only after every new
        object is stored and the database keys are committed.
"""
from __future__ import annotations

import argparse, os, shutil, sys, time
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

def media_layout(apply=False,delete_old=False):
    """Safely copy legacy media to the structured local/bucket keys, then atomically switch database references."""
    from sqlalchemy import select, func
    from .database import SessionLocal
    from .models import Media, Inspection, Site, Job
    from .tasks import ROOT  # also registers ROOT with the production bucket adapter
    from . import media_paths, storage
    if delete_old and not apply:sys.exit('--delete-old requires --apply')
    with SessionLocal() as db:
        active=db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(['queued','running']))) or 0
        if active:sys.exit(f'Refusing to reorganise media while {active} processing job(s) are queued or running')
        rows=db.scalars(select(Media).order_by(Media.parent_media_id.is_not(None),Media.id)).all()
        by_id={m.id:m for m in rows};inspections={};sites={};planned={};moves=[]
        for m in rows:
            inspection=inspections.setdefault(m.inspection_id,db.get(Inspection,m.inspection_id))
            if not inspection:continue
            site=sites.setdefault(inspection.site_id,db.get(Site,inspection.site_id))
            if not site:continue
            if media_paths.is_structured(m.storage_key):new=m.storage_key
            else:
                parent=by_id.get(m.parent_media_id) if m.parent_media_id else None
                parent_key=planned.get(parent.id,parent.storage_key) if parent else None
                new=media_paths.migrated_key(site.code,inspection.id,m.id,m.filename,m.content_type,m.storage_key,parent_key,parent.id if parent else None,m.frame_index)
            planned[m.id]=new
            if new!=m.storage_key:moves.append((m,m.storage_key,new))
        total=sum((ROOT/old).stat().st_size if (ROOT/old).is_file() else 0 for _,old,_ in moves)
        print(f'{len(moves)} of {len(rows)} media file(s) need the structured layout ({total/1e6:.1f} MB currently local)')
        for _,old,new in moves[:20]:print(f'  {old} -> {new}')
        if len(moves)>20:print(f'  ... and {len(moves)-20} more')
        print(f'  bucket: {storage.BUCKET if storage.enabled() else "not configured (local files only)"}')
        if not apply:return {'planned':len(moves),'total':len(rows)}
        copied=[];missing=[]
        try:
            for n,(_,old,new) in enumerate(moves,1):
                dst=ROOT/new
                if storage.enabled():  # server-side copy: nothing is downloaded (the bucket is the store)
                    if not storage.head(dst) and not storage.copy(ROOT/old,dst):
                        if not (ROOT/old).is_file():missing.append(old);continue
                        storage.push(ROOT/old);storage.copy(ROOT/old,dst)
                    if (ROOT/old).is_file() and not dst.is_file():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/old,dst);copied.append(dst)
                else:
                    if not dst.is_file():
                        if not (ROOT/old).is_file():missing.append(old);continue
                        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/old,dst);copied.append(dst)
                if n%25==0 or n==len(moves):print(f'  stored {n}/{len(moves)}',flush=True)
            if missing:raise RuntimeError(f'{len(missing)} source file(s) are missing locally and in the bucket; first: {missing[0]}')
            for m,_,new in moves:m.storage_key=new
            db.commit()
        except Exception:
            db.rollback()
            print('Database keys were not changed. Any copied destination objects are safe to reuse on the next run.',file=sys.stderr)
            raise
        deleted=0
        if delete_old:
            new_keys=set(planned.values())
            for _,old,new in moves:
                if old in new_keys or db.scalar(select(func.count()).select_from(Media).where(Media.storage_key==old)):continue
                path=ROOT/old;storage.delete(path);path.unlink(missing_ok=True);deleted+=1
                parent=path.parent
                while parent!=ROOT and ROOT in parent.parents:
                    try:parent.rmdir()
                    except OSError:break
                    parent=parent.parent
        print(f'Media layout complete: {len(moves)} database key(s) updated, {deleted} old object(s) removed.')
        return {'migrated':len(moves),'deleted':deleted,'total':len(rows)}

def free_local(apply=False):
    """Delete local files that the bucket holds with the same size (uploads/, reconstructions/; never the database).
    Without --apply only reports. Files missing in the bucket or of a different size are kept."""
    from . import storage, reconstruct, vendor
    from .tasks import ROOT
    if not storage.enabled():sys.exit('No bucket configured')
    keep=free=n=0;kept=[]
    for folder in (ROOT,reconstruct.RECON_DIR):
        for p in sorted(folder.rglob('*')) if folder.is_dir() else []:
            if not p.is_file() or p.name==storage.IN_USE or any(m.name==storage.IN_USE for m in p.parent.glob(storage.IN_USE)):continue
            size=p.stat().st_size;h=None if '.thumbs' in p.parts else storage.head(p)
            if '.thumbs' in p.parts or (h and h[1]==size):
                n+=1;free+=size
                if apply:p.unlink(missing_ok=True)
            else:keep+=size;kept.append(p)
    print(f'{"Deleted" if apply else "Would delete"} {n} local file(s), {free/1e9:.2f} GB stored in the bucket; kept {len(kept)} ({keep/1e6:.1f} MB) not in the bucket')
    for p in kept[:10]:print('   kept',p)
    return {'deleted' if apply else 'deletable':n,'bytes':free,'kept':len(kept)}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('db'); d.add_argument('--source', default='sqlite:///tower_vision.db'); d.add_argument('--target', required=True); d.add_argument('--replace', action='store_true')
    f = sub.add_parser('files'); f.add_argument('--workers', type=int, default=8)
    m = sub.add_parser('media-layout');m.add_argument('--apply',action='store_true');m.add_argument('--delete-old',action='store_true')
    fl = sub.add_parser('free-local');fl.add_argument('--apply',action='store_true')
    a = ap.parse_args()
    if a.cmd == 'db': copy_db(a.source, a.target, a.replace)
    elif a.cmd == 'files': copy_files(a.workers)
    elif a.cmd == 'free-local': free_local(a.apply)
    else: media_layout(a.apply,a.delete_old)

if __name__ == '__main__':
    main()
