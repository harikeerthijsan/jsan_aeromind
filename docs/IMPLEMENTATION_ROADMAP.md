# Implementation roadmap and production gaps

## Core progression
1. Identity and permissions: OIDC, RBAC, tenant isolation, audit, CSRF/CORS restrictions, rate limits, quotas, S3 private objects.
2. Database migrations and PostGIS coordinate model with accuracy/provenance, asset identity edges, evidence and condition history.
3. Media worker: FFmpeg frame extraction with timestamps, image QC, keyframe selection and image tiling.
4. YOLO adapter: fine-tuned detector, class taxonomy, calibration, bounding boxes/masks, source observations.
5. SAM 2 adapter: prompted segmentation and tracklets; cross-frame mask propagation.
6. Multi-view reconciliation: visual embeddings, calibrated camera geometry, human merge/split review; a detection is not an asset.
7. COLMAP/OpenDroneMap adapter: camera calibration, SfM, sparse/dense point clouds, metric scale validation and GLB generation, versioned model quality reports.
8. 2D→3D registration: camera poses, feature/keypoint correspondences, triangulation, reprojection QC; immutable asset-local frame and aligned model revisions. Support unlocalized assets, unknown position uncertainty, occlusion.
9. Qwen3-VL evidence packages for visible defects; schema-constrained suggestions; deterministic grading rules; external confidence calibration; Unknown and QA gate.
10. Frontend real model loader, inspection camera overlays, review queues, changes across dates, report builder, PDF/XLSX/GeoJSON exporters.
11. Continuous evaluation: per-class precision/recall, count error, identity match, defect performance and critical defect sensitivity; split by site, not frame.

## Domain invariants
- Detection observations are many-to-one with canonical physical assets.
- Never treat missing visibility as confirmed asset removal.
- Never claim visual inspection confirms RF function, bolt torque, corrosion depth or tower safety.
- 3D anchors must carry registration provenance; invalid/unverified locations must be explicitly marked.
- Every approved defect has source evidence IDs and timestamp, reviewer and rubric version.

## Suggested production endpoints beyond starter
- POST /inspections/{id}/reconstruct-3d
- GET /models/{id}/manifest
- GET /models/{id}/camera-poses
- POST /models/{id}/register-assets
- POST /assets/{id}/identity-review
- GET /assets/{id}/observations
- POST /inspections/{id}/finalize
- POST /reports
- GET /reports/{id}/download

## Production readiness checklist
- E2E authorization, isolation, data retention & encryption tests
- Media antivirus, upload presigning, integrity checks
- PostgreSQL migrations, backups and restore drill
- GPU model version pinning, repeatable inference and failure recovery
- Geometry benchmarks with calibrated drone capture
- Human QA and critical defect escalation procedures
- Monitoring, tracing, queue and worker autoscaling
- Report template acceptance and signed export provenance
