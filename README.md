# AeroMind — telecom tower inspection workspace

React + TypeScript + Three.js + FastAPI application starter for drone-media-based tower asset inspection and 3D digital-twin workflows.

## What is working
- Responsive SaaS UI with dashboard, project/site list, inspections, media upload, processing jobs, AI detection workspace placeholder, reconstruction studio, interactive procedural 3D tower, asset inventory, defect review, comparisons, reports, work orders and administration views.
- FastAPI OpenAPI docs, persistent SQLAlchemy data models, site/inspection creation, real file streaming uploads with basic type and size limits, asset/defect listing, asset corrections, human defect review and CSV export.
- **3D Digital Twin** automatically loads the latest completed reconstruction for the selected tower, including sparse points or a dense mesh, source-camera overlay and model-quality details. Towers without a completed model retain the procedural demonstration fallback. **Generate estimated markers** performs conservative 2D→3D registration: high-confidence equipment boxes are located with COLMAP's tracked 3D feature points and clustered across multiple images. Orange markers remain `needs_review`; they are not silently promoted to canonical assets.
- Demo seed dataset clearly flagged illustrative.

- YOLO asset detection (optional): with `YOLO_MODEL_PATH` set, `POST /api/v1/inspections/{id}/process` runs the fine-tuned detector over the inspection's images (and every `YOLO_VID_STRIDE`-th video frame) in a background job and stores boxes as `detections` observations. Read them with `GET /api/v1/inspections/{id}/detections`; `GET /api/v1/detector` shows the model's config and classes.
- Detection review and manual QC: the **AI Detection** and **QC Review** pages share a box editor (zoom/pan, full screen, move/resize, relabel, delete, draw new boxes, undo/redo, keyboard shortcuts). **Approve** saves the reviewed box set for that image to `annotations` (with an `image_reviews` sign-off record) and marks the model's detections accepted/corrected/rejected. API: `GET|PUT /api/v1/media/{id}/annotations`, `GET /api/v1/inspections/{id}/reviews`, `GET /api/v1/classes`. Images only; video frames are view-only.
- Annotated output that keeps image metadata: YOLO's `save=True` (and any plain re-encode) strips EXIF, GPS, maker notes, DJI XMP (gimbal/flight data), ICC and IPTC. `app/annotate.py` draws the boxes and copies the source's JPEG metadata segments into the output byte-for-byte (PNG/TIFF via Pillow; the stale MPF preview index is dropped). Download per image with `GET /api/v1/media/{id}/annotated?source=auto|model|approved` or per inspection with `GET /api/v1/inspections/{id}/annotated.zip` (buttons in both viewers), or run it standalone instead of `predict(save=True)`:
  ```bash
  cd backend
  <yolo-gpu python> -m app.annotate D:/survey/raw --out D:/survey/annotated --model D:/TowerYOLO/runs/tower26s_1920/weights/best.pt [--conf 0.25] [--recursive] [--no-labels]
  ```
- Manufacturer tagging: sector antennas and microwave dishes (`VENDOR_CLASSES`) get Manufacturer / Product model fields in the box editor, saved with the approved annotations and exported via `GET /api/v1/inspections/{id}/annotations.csv` and the labels of annotated images. **Suggest with AI** runs local Qwen3-VL models through Ollama on a crop from the original file, in two stages: (1) logo / printed text (`VLM_MODEL`, default `qwen3-vl:8b-instruct`); (2) if nothing is legible, the physical structure (radome profile, end caps, ODU, mounts) is compared one manufacturer at a time with reference photos of known equipment (`VLM_COMPARE_MODEL`, default the thinking `qwen3-vl:8b`). References are reviewer-confirmed tags plus any images in `VENDOR_REFERENCE_DIR/<class>/<Manufacturer>/`. Structure matches are capped at 60% confidence, two close matches give *Unknown* (ambiguous), and the model's own recognition of a design is only shown as an unverified hint. Setup: `ollama pull qwen3-vl:8b-instruct` and `ollama pull qwen3-vl:8b`.
  - Reference library: `python -m app.refs_import <manifest.csv>` (run in `backend/`) downloads catalog photos listed in a manifest (`category, manufacturer, model, view, url, page, status, notes`) into `vendor_refs/<class>/<Manufacturer>/` and indexes them in `vendor_refs/references.csv`. Connector-end views are kept for reviewers but not used for matching. Reviewer-confirmed drone crops are preferred over catalog photos. The box editor shows the library as a **Reference photos** gallery. The images come from third-party sites; keep the library internal (it is git-ignored).
- Condition assessment: every box can be graded **Good / Fair / Poor / Critical / Unknown** with a 0-100 health score and a list of visible defects (rust, corrosion, paint peeling, cracks, dents, radome damage, cable damage, missing parts, bird nests ... each with severity, approximate % of the visible surface and location). **Assess condition with AI** (per box, or for every unassessed box) uses the fast `VLM_MODEL`; results with defects are re-checked by `VLM_COMPARE_MODEL` and marked *confirmed* or *disputed* (disputed results drop to 30% confidence). Boxes smaller than `CONDITION_MIN_SIDE` px (default 48) are not sent to the model and come back *Unknown - needs a closer photo*. Reviewers can change the grade and add or remove defects. Results are saved with the approved annotations, shown on annotated images and exported in `annotations.csv`. Percentages are visual estimates from one photo, not measurements.
- Approved data drives the inventory pages (`GET /api/v1/sites/{id}/inventory`): **Asset Inventory** shows the equipment observed in QC-approved images (class counts, manufacturers, condition breakdown) and one card per image with its boxes drawn and an item list; **Condition & Defects** lists Fair/Poor/Critical items worst first with a crop of the equipment, defects, second-model status and an **Open in QC** link that opens the image with the box selected (`/review?inspection=&media=&item=`). The dashboard counts observed equipment and items with defects. Items are per image: the same physical antenna seen in two photos is counted twice until identity reconciliation exists.
- Uploads and video: **Media Upload** can start detection automatically for just the new files (`POST /api/v1/inspections/{id}/process?media_ids=..`, or `only_new=true`). Videos are split into still frames (`VIDEO_FRAME_STRIDE`, default every 30th; at most `VIDEO_MAX_FRAMES`, default 300) stored under `uploads/frames/<video id>/`. The frames are listed under their video and reviewed, approved and inventoried like photos. Each file shows its latest detection run. Drone telemetry (GPS/gimbal) is not carried onto frames.
- 3D reconstruction (photogrammetry) with [COLMAP](https://colmap.github.io) (BSD): on **3D Reconstruction**, *Build 3D model* runs COLMAP on the inspection's photos and on frames sampled from its videos (`RECON_VIDEO_STRIDE`, default every 10th). *Sparse* gives camera positions and a point cloud in minutes; *Dense + mesh* adds GPU stereo, a dense point cloud and a Poisson mesh (10-60 min). The model is levelled and centred automatically and shown in the browser with the drone camera path; full-resolution PLY files and the COLMAP log can be downloaded. Setup: download `colmap-x64-windows-cuda.zip` from the COLMAP releases, unzip, and set `COLMAP_PATH=.../bin/colmap.exe` in `backend/.env`. Best input: an orbit around the tower at 2-3 heights with 70-80% overlap. The model has no real-world scale or GPS position yet.

## Not implemented (integration boundaries)
- No YOLO, SAM 2, Qwen3-VL or COLMAP weights bundled (YOLO is configured via `YOLO_MODEL_PATH`; Qwen3-VL runs in a local Ollama). No segmentation, object tracking across video frames, confirmed identity reconciliation, or georeferencing/scale for 3D models. The 2D→3D markers are geometry-based review candidates from high-confidence detections, not confirmed physical assets. Condition grades and manufacturer suggestions are vision-model estimates for reviewers to confirm, not measurements. Approved items and registration candidates are not promoted to the canonical asset register automatically.
- No authentication/authorization, tenant isolation, signed object storage, migrations, audit log, out-of-process job queue or production security hardening. `uploads` static serving is for **local development only**; not safe for private customer imagery.
- Without a configured detector, inspection processing creates a truthful `awaiting_worker` record instead of inventing inference results.
- To deliver the full enterprise feature scope see `docs/IMPLEMENTATION_ROADMAP.md`.

## Local development
Requires Python 3.11+, Node 20+.

### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

#### With YOLO detection
Use a Python that already has CUDA torch + ultralytics (e.g. the `yolo-gpu` conda env), or `pip install -r requirements-ml.txt`. Then:
```bash
cd backend
<yolo-gpu python> -m venv --system-site-packages .venv   # reuse the env's torch/ultralytics
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env    # set YOLO_MODEL_PATH, e.g. D:/TowerYOLO/runs/tower26s_1920/weights/best.pt
.venv\Scripts\python -m uvicorn app.main:app --reload --reload-dir app --port 8000 --env-file .env
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
UI: http://localhost:5173

Use `ENABLE_DEMO_SEED=false` for an initially empty local database. In production, replace SQLite with Postgres/PostGIS and implement Alembic migrations and authentication before accepting real customer data.

## Deployment (Railway + local GPU worker)
```
 browser ──HTTPS──► Railway: aeromind (FastAPI + built frontend, Dockerfile)
                        │  ├─ PostgreSQL (private network only)
                        │  └─ bucket aeromind-data (S3): uploads/, reconstructions/, vendor_refs/
                        ▲
     HTTPS, outgoing only (WORKER_TOKEN) + bucket
                        │
 GPU PC: backend/run-worker.ps1  ── YOLO (CUDA) · COLMAP (CUDA) · Qwen3-VL in Ollama
```
- **Railway service** (`Dockerfile`, `railway.json`): serves the API and the frontend from one container with `GPU_WORKER=remote`. It has no GPU, so detection runs, 3D reconstruction, manufacturer/condition suggestions and 3D box registration are queued (`worker_tasks`) instead of run.
- **GPU worker** (`app/worker.py`): on the machine with the GPU, it polls `/api/v1/worker/claim`, receives a prepared payload, computes with the local YOLO weights, COLMAP and Ollama (`app/tasks.py`, no database access) and posts the result, which the server writes to the database (`app/gpu.py`). Only outgoing HTTPS: no open port, tunnel or public database. While it is off, the app keeps working and GPU actions say *GPU worker is offline*.
- **Files** (`app/storage.py`): every upload and 3D output is pushed to the bucket; each side downloads what it is missing on first use. Without `S3_*` settings everything stays local as before.
- **Login**: with `APP_PASSWORD` set, the browser asks for `APP_USERNAME` / `APP_PASSWORD` (HTTP Basic). Delete the variable to open the app.

Start the worker (keep it running whenever the web app should use the GPU):
```powershell
cd backend
copy worker.env.example worker.env   # AEROMIND_SERVER, WORKER_TOKEN (same as on Railway), S3_* bucket keys
.\run-worker.ps1                      # restarts itself after a crash; Ctrl+C to stop
```
Railway variables of the `aeromind` service: `DATABASE_URL=${{Postgres.DATABASE_URL}}`, `GPU_WORKER=remote`, `WORKER_TOKEN`, `S3_BUCKET`, `S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `APP_USERNAME`, `APP_PASSWORD`, `PORT=8000`.
Deploy: `railway up --service aeromind` from the repository root (or connect the GitHub repo in the Railway dashboard).

Moving a local installation (run in `backend/` with `worker.env` loaded, e.g. `python load_worker_env.py -m app.migrate files`):
- `python -m app.migrate files` uploads `uploads/`, `reconstructions/` (without the `images/` copies) and `vendor_refs/` to the bucket; re-runnable, skips files already there.
- Database: upload an SQLite snapshot to the bucket and set `IMPORT_DB_FROM_BUCKET=<key>` on the service once: an empty database is filled from it at startup (remove the variable afterwards). Or `python -m app.migrate db --target <postgres url>` where the database is reachable.

## Security notes
The deployment is protected by one shared login (`APP_PASSWORD`), not per-user accounts; uploaded files are served to anyone with that login. Add OIDC/RBAC, malware scanning, quotas, private S3 with signed URLs, and strict tenant query scoping. Commercial use of YOLO implementations may require separate licensing.

## License
No license granted by this sample package; confirm ownership and dependencies before commercial distribution.
