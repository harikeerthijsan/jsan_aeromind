# UI page coverage and next-step integrations

| Screen | Status | Notes |
|---|---|---|
| Executive dashboard | Functional demo | Live counters and sample condition distribution |
| Projects and tower sites | CRUD starter | Site create and list; needs editing/delete and tenancy |
| Tower detail | Demo + live site | Site metrics, navigation |
| Inspections | Create/list | Needs campaign scope form and metadata |
| Media upload | Functional local | Real file uploads, optional auto-detection of just the new files; private S3 missing |
| Processing center | Functional | YOLO jobs (in-process) incl. video frame extraction, progress and results; awaiting_worker when no model is configured |
| AI detection | Functional | YOLO boxes with zoom/pan/full screen, edit/relabel/delete/draw, per-image approval; SAM 2 not wired |
| Reconstruction studio | Functional | COLMAP sparse/dense + mesh build jobs, point cloud/mesh viewer with drone cameras, quality metrics, downloads; no scale/georeference yet |
| 3D digital twin | Reconstruction + candidate registration | Loads the selected tower's latest sparse/dense model and quality metrics; creates multi-image, COLMAP-backed estimated equipment markers; candidates require review before canonical asset promotion |
| Asset inventory | Live API | Observed equipment from approved images (per image, classes, manufacturers, condition); demo canonical register below; needs reconciliation across images |
| Condition and defect QA | Functional | Findings from approved condition assessments, worst first, deep link into QC; demo defect records below |
| QC review | Functional | Image-by-image QC queue using the same box editor; approved boxes saved as annotations |
| Inspection comparison | Layout scaffold | Requires dated stable-ID baseline |
| Reports | CSV works | PDF/GLB/GeoJSON gated |
| Work orders | Design scaffold | Needs action model and workflow |
| Taxonomy and rules | Design scaffold | Needs versioned domain rules |
| Model settings | Design scaffold | Needs registry, GPU jobs and metrics |
| Users and roles | Design scaffold | Needs OIDC/RBAC |
| Integrations | Design scaffold | Needs provider connections |
| Settings and audit | Design scaffold | Needs actual event trail |

A production-quality 26-screen specification is described in the preceding planning discussion; this repository is an intentionally honest implementation starter, not the completed production platform.
