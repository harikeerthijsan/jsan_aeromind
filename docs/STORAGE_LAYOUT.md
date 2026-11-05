# Inspection media storage layout

Uploaded media is grouped by tower, inspection, and media type. The local directory and production S3/Railway bucket use the same relative structure:

```text
uploads/
└── sites/
    └── TWR-002/
        └── inspections/
            └── INS-000004/
                ├── images/
                │   └── MEDIA-000005_Model_001.jpg
                ├── videos/
                │   └── a1b2c3d4e5f6_orbit-pass.mov
                └── frames/
                    └── MEDIA-000106/
                        ├── 000000.jpg
                        └── 000030.jpg
```

Bucket objects have the same path with the registered `uploads/` prefix. For example:

```text
uploads/sites/TWR-002/inspections/INS-000004/images/MEDIA-000005_Model_001.jpg
```

Names supplied by clients are sanitized, while the file extension always comes from the validated MIME type. A unique prefix prevents two uploads with the same filename from overwriting one another.

## Existing-media migration

Run a read-only preview first from `backend/`:

```powershell
python -m app.migrate media-layout
```

Apply the migration while no processing jobs are running:

```powershell
python -m app.migrate media-layout --apply --delete-old
```

The migration performs these operations in order:

1. Copies every legacy local or bucket object to its structured destination.
2. Uploads the destination to the configured S3-compatible bucket.
3. Updates database `storage_key` values only after every copy succeeds.
4. Deletes old objects only after the database commit, and only with `--delete-old`.

The command is safe to rerun. It refuses to start while detection or reconstruction jobs are queued or running.

For production, run it inside the deployed backend environment so `DATABASE_URL` and the `S3_*`/Railway bucket variables point to production. New uploads use the structured layout automatically without running the migration.

## Whole bucket

```text
<bucket>/
├── uploads/
│   └── sites/<TOWER>/inspections/INS-000004/
│       ├── images/        MEDIA-000005_Model_001.jpg …   photos
│       ├── videos/        <token>_<name>.mp4             videos
│       ├── frames/        MEDIA-<video id>/000030.jpg    frames extracted from videos
│       └── pointclouds/   <token>_<name>.laz|.e57|.las|.ply|.xyz (+ .lax/.prj sidecars)
├── reconstructions/<inspection>_<job>/   3D builds: sparse/, dense/fused.ply, web/points.ply, web/cameras.json, colmap.log
├── vendor_refs/<class>/<Manufacturer>/   manufacturer reference photos (+ references.csv)
├── backups/db/                           database snapshots
└── dev/                                  local development copy's own files (same structure below it)
```

## Where each installation reads and writes

| Installation | Writes to | Reads from | Local disk |
|---|---|---|---|
| Railway server (`STORAGE_MODE=bucket`) | `uploads/`, `reconstructions/` … | same | streams files, small cache |
| GPU worker (`run-worker.ps1`, `STORAGE_MODE=bucket`) | production keys | production keys | cache ≤ `STORAGE_CACHE_GB`, least recently used trimmed |
| Local development copy (`backend/storage.env`: `S3_PREFIX=dev/`) | `dev/…` | `dev/…`, then production keys | cache in `backend/storage-cache/dev/` |

The local copy has its own database whose inspection, job and media ids overlap production's, so its new files go to `dev/` (3D build folders are named `<inspection>_<job>`). Everything both databases know about (data that existed when production was set up) is read from the one shared copy.

`python -m app.migrate free-local [--apply]` deletes local files whose identical copy (same size) is in the bucket.
