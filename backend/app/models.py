from datetime import datetime, timezone
from sqlalchemy import String, Integer, BigInteger, Float, DateTime, ForeignKey, Text, JSON, UniqueConstraint, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base

def now(): return datetime.now(timezone.utc)
class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    role: Mapped[str] = mapped_column(String(24), default='viewer')  # admin | manager | inspector | viewer
    password_hash: Mapped[str] = mapped_column(String(512))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class AuthSession(Base):
    __tablename__ = 'auth_sessions'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class Site(Base):
    __tablename__ = 'sites'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(80), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    region: Mapped[str] = mapped_column(String(120), default='Unknown')
    tower_type: Mapped[str] = mapped_column(String(60), default='Lattice')
    height_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default='active')  # active | maintenance | decommissioned | archived
    operator: Mapped[str | None] = mapped_column(String(120), nullable=True)  # tower company / mobile operator
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=now)
class Inspection(Base):
    __tablename__ = 'inspections'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey('sites.id'))
    label: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(48), default='draft')
    tower_type: Mapped[str | None] = mapped_column(String(60), nullable=True)  # chosen at upload (lattice / monopole); None = the site's type
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
class Media(Base):
    __tablename__ = 'media'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'))
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)  # videos and point clouds can exceed 2 GB
    # a still frame extracted from an uploaded video (see frames.py); None for files uploaded directly
    parent_media_id: Mapped[int | None] = mapped_column(ForeignKey('media.id'), nullable=True, index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # seconds into the video
class Asset(Base):
    __tablename__ = 'assets'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey('sites.id'))
    code: Mapped[str] = mapped_column(String(100), unique=True)
    category: Mapped[str] = mapped_column(String(100))
    condition: Mapped[str] = mapped_column(String(24), default='Unknown')
    identity_status: Mapped[str] = mapped_column(String(32), default='provisional')
    localization_status: Mapped[str] = mapped_column(String(32), default='unlocalized')
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    z: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str] = mapped_column(Text, default='')
class Defect(Base):
    __tablename__ = 'defects'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey('assets.id'))
    defect_type: Mapped[str] = mapped_column(String(100))
    proposed_grade: Mapped[str] = mapped_column(String(24), default='Unknown')
    review_status: Mapped[str] = mapped_column(String(32), default='pending')
    notes: Mapped[str] = mapped_column(Text, default='')
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'))
    kind: Mapped[str] = mapped_column(String(50), default='full_inspection')
    status: Mapped[str] = mapped_column(String(40), default='queued')
    stage: Mapped[str] = mapped_column(String(80), default='awaiting_worker')
    progress: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[str] = mapped_column(Text, default='')
    media_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)  # uploaded files this run covers; None = the whole inspection
class Detection(Base):
    # A detection is an observation, not an asset; asset_id is set later by reconciliation/review.
    __tablename__ = 'detections'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey('jobs.id'), index=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'), index=True)
    media_id: Mapped[int] = mapped_column(ForeignKey('media.id'))
    asset_id: Mapped[int | None] = mapped_column(ForeignKey('assets.id'), nullable=True)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(255))
    review_status: Mapped[str] = mapped_column(String(32), default='unreviewed')
class ImageReview(Base):
    # Human sign-off for one image (or video frame); the approved boxes are its annotations.
    __tablename__ = 'image_reviews'
    __table_args__ = (UniqueConstraint('media_id', 'frame_index'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'), index=True)
    media_id: Mapped[int] = mapped_column(ForeignKey('media.id'))
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default='approved')
    workflow: Mapped[str] = mapped_column(String(32))  # ai_detection | manual_qc
    job_id: Mapped[int | None] = mapped_column(ForeignKey('jobs.id'), nullable=True)  # detection run the review started from
    box_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default='')
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
class Annotation(Base):
    # Human-approved box. Raw model detections stay untouched in `detections` for model evaluation.
    __tablename__ = 'annotations'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey('image_reviews.id'), index=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'), index=True)
    media_id: Mapped[int] = mapped_column(ForeignKey('media.id'))
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    detection_id: Mapped[int | None] = mapped_column(ForeignKey('detections.id'), nullable=True)
    source: Mapped[str] = mapped_column(String(24))  # model | model_edited | manual
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # model score, kept for reference
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    # equipment manufacturer, for vendor classes (sector antennas, microwave dishes)
    manufacturer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    product_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    vendor_source: Mapped[str | None] = mapped_column(String(24), nullable=True)  # reviewer | ai (suggestion left as-is)
    vendor_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # AI confidence when vendor_source == 'ai'
    vendor_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    vendor_basis: Mapped[str | None] = mapped_column(String(24), nullable=True)  # AI evidence: logo_text | reference_match | structure | none
    # condition assessment (any class): Good | Fair | Poor | Critical | Unknown, 0-100 score, [{type, severity, extent_percent, location, evidence, source}]
    condition_grade: Mapped[str | None] = mapped_column(String(16), nullable=True)
    condition_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    condition_defects: Mapped[list | None] = mapped_column(JSON, nullable=True)
    condition_source: Mapped[str | None] = mapped_column(String(24), nullable=True)  # ai | reviewer
    condition_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    condition_verified: Mapped[str | None] = mapped_column(String(16), nullable=True)  # confirmed | disputed | not_run (second-model check)
    condition_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
class Model3D(Base):
    __tablename__ = 'models_3d'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'))
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default='not_available')
    quality: Mapped[dict] = mapped_column(JSON, default=dict)
    job_id: Mapped[int | None] = mapped_column(ForeignKey('jobs.id'), nullable=True)  # reconstruction job that built it
    dense: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=now)
class RegistrationCandidate(Base):
    __tablename__ = 'registration_candidates'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('models_3d.id'), index=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'), index=True)
    site_id: Mapped[int] = mapped_column(ForeignKey('sites.id'), index=True)
    class_name: Mapped[str] = mapped_column(String(100))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)  # horizontal model axis (Three.js Z)
    z: Mapped[float] = mapped_column(Float)  # model height (Three.js Y)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default='needs_review')
    evidence_count: Mapped[int] = mapped_column(Integer)
    media_count: Mapped[int] = mapped_column(Integer)
    point_count: Mapped[int] = mapped_column(Integer)
    spread: Mapped[float] = mapped_column(Float)
    detection_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # oriented 3D box (viewer units): width along the tower tangent, height vertical, depth facing outward; yaw about +Y
    size_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    yaw: Mapped[float | None] = mapped_column(Float, nullable=True)
    box_method: Mapped[str | None] = mapped_column(String(16), nullable=True)  # dense (fitted to dense points) | estimate (photo-based)
    box_points: Mapped[int | None] = mapped_column(Integer, nullable=True)  # dense points the box was fitted to
    fit_iou: Mapped[float | None] = mapped_column(Float, nullable=True)  # median agreement between projected 3D box and source 2D boxes

class WorkerTask(Base):
    # Work for the GPU worker when the server has no GPU (GPU_WORKER=remote, e.g. on Railway): detection runs,
    # 3D reconstruction and the vision-model calls. The worker claims queued rows from this table (see app/gpu.py).
    __tablename__ = 'worker_tasks'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    lane: Mapped[str] = mapped_column(String(8), default='long')  # long (jobs) | quick (interactive calls someone waits on)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default='queued', index=True)  # queued | running | done | failed
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {'status': http status, 'detail': message}
    worker: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class WorkerState(Base):
    # Heartbeat of the GPU worker: what it can do (YOLO model, COLMAP, vision models) and when it was last seen.
    __tablename__ = 'worker_state'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    info: Mapped[dict] = mapped_column(JSON, default=dict)
    seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

class PointCloudFile(Base):
    # An uploaded point cloud (LAS/LAZ/PLY/XYZ...) or its sidecar (.lax index, .prj coordinate system) for an inspection.
    # Building from it replaces COLMAP's dense reconstruction (see app/pointcloud.py).
    __tablename__ = 'point_cloud_files'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspections.id'), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(500))
    file_format: Mapped[str] = mapped_column(String(16))  # extension without the dot: las, laz, ply, xyz, pts, txt, csv, lax, prj
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # COPC copy for the browser (every point, streamed level of detail by Potree): converted with Untwine on the GPU worker
    copc_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    copc_status: Mapped[str | None] = mapped_column(String(20), nullable=True)  # queued / converting / ready / failed
    copc_points: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    copc_error: Mapped[str | None] = mapped_column(Text, nullable=True)
