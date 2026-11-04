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
