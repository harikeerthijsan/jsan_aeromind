import os, uuid, hashlib, shutil, csv, io, tempfile, zipfile, threading, math, secrets, base64
from urllib.parse import quote
from collections import Counter
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.encoders import jsonable_encoder
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, update, delete, inspect, text, and_, or_
from .database import Base, engine, SessionLocal
from .models import Site, Inspection, Media, Asset, Defect, Job, Model3D, RegistrationCandidate, Detection, ImageReview, Annotation, now
from . import detector, vendor, condition, frames, reconstruct, registration, storage, gpu
from .tasks import ROOT  # uploads folder; it and the 3D/reference folders are backed by the bucket when one is configured
MAX_SIZE = int(os.getenv('MAX_UPLOAD_BYTES', str(500 * 1024 * 1024)))
ALLOWED = {'image/jpeg':'.jpg','image/png':'.png','image/tiff':'.tif','video/mp4':'.mp4','video/quicktime':'.mov'}  # stored extension comes from the checked type, never the client's filename
WORKFLOWS = {'ai_detection','manual_qc'}
STAGES = ['validation','frame_extraction','detection','segmentation','identity_reconciliation','reconstruction','registration','vlm_condition','qa_pending']

def seed(db):
    if db.scalar(select(func.count()).select_from(Site)): return
    for code,name,region,height in [('TWR-001','North Ridge','Hyderabad',45),('TWR-002','West Link','Bengaluru',38),('TWR-003','Coastal Point','Chennai',52)]:
        db.add(Site(code=code,name=name,region=region,height_m=height,tower_type='Lattice'))
    db.flush()
    site=db.scalar(select(Site).where(Site.code=='TWR-001'))
    db.add(Inspection(site_id=site.id,label='Reference survey · Demo',status='demo'))
    rows=[('ANT-009','Panel Antenna','Fair',2.2,0.2,33,'Suspected radome wear'),('ANT-010','Panel Antenna','Good',-1.8,0.4,34,'No visible defect in reference imagery'),('RRU-003','Remote Radio Unit','Good',1.2,-0.7,29,'Reference equipment'),('BRK-004','Mounting Bracket','Poor',-1.5,-0.2,31,'Visible corrosion'),('MW-002','Microwave Dish','Unknown',1.8,0.1,24,'Insufficient inspection evidence'),('CBL-008','Cable Run','Fair',0.3,0.6,21,'Potential missing support')]
    for code,cat,grade,x,y,z,desc in rows:
        db.add(Asset(site_id=site.id,code=code,category=cat,condition=grade,identity_status='demo',localization_status='demo_approximate',x=x,y=y,z=z,description=desc))
    db.flush()
    for code,kind,grade in [('ANT-009','Surface wear','Fair'),('BRK-004','Corrosion','Poor'),('CBL-008','Unsupported cable','Fair')]:
        a=db.scalar(select(Asset).where(Asset.code==code))
        db.add(Defect(asset_id=a.id,defect_type=kind,proposed_grade=grade,review_status='demo_unverified',notes='Illustrative finding — not AI verified',evidence={}))
    db.commit()

def add_missing_columns():
    # create_all doesn't alter existing tables; add new nullable columns so older databases keep working
    insp=inspect(engine)
    with engine.begin() as conn:
        for t in Base.metadata.sorted_tables:
            if not insp.has_table(t.name):continue
            have={c['name'] for c in insp.get_columns(t.name)}
            for c in t.columns:
                if c.name not in have and c.nullable:conn.execute(text(f'ALTER TABLE {t.name} ADD COLUMN {c.name} {c.type.compile(engine.dialect)}'))
def import_snapshot():
    """One-time move of a local installation: IMPORT_DB_FROM_BUCKET=<key of an SQLite file in the bucket> fills an empty
    database from it at startup (python -m app.migrate db, run where the database is reachable). Ignored once there is data."""
    key=os.getenv('IMPORT_DB_FROM_BUCKET')
    if not key or not storage.enabled():return
    with SessionLocal() as db:
        if db.scalar(select(func.count()).select_from(Site)) or db.scalar(select(func.count()).select_from(Media)):return
    from . import migrate
    tmp=Path(tempfile.gettempdir())/'aeromind-import.db'
    storage.client().download_file(storage.BUCKET,key,str(tmp))
    try:migrate.copy_db(f'sqlite:///{tmp.as_posix()}',engine.url.render_as_string(hide_password=False),replace=True)
    finally:tmp.unlink(missing_ok=True)
@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine);add_missing_columns();import_snapshot()
    with SessionLocal() as db:
        if not gpu.REMOTE:  # jobs run in-process, so anything still active at startup was cut off by a restart/reload
            db.execute(update(Job).where(Job.status.in_(['queued','running'])).values(status='failed',detail='Interrupted by server restart; re-run processing.'))
            db.execute(update(Model3D).where(Model3D.status.in_(['queued','running'])).values(status='failed'));db.commit()
        if os.getenv('ENABLE_DEMO_SEED','true').lower()=='true': seed(db)
    if storage.enabled():  # the reference library is small and listed (not just opened), so keep a full copy
        threading.Thread(target=lambda:storage.fetch_tree(vendor.VENDOR_REFERENCE_DIR),daemon=True).start()
    yield

app=FastAPI(title='AeroMind API',version='0.1.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173').split(','),allow_methods=['*'],allow_headers=['*'])
@app.middleware('http')
async def no_sniff(request,call_next):
    r=await call_next(request);r.headers['X-Content-Type-Options']='nosniff';return r  # browsers must not reinterpret uploads as HTML
USER,PASSWORD=os.getenv('APP_USERNAME','aeromind'),os.getenv('APP_PASSWORD','')
@app.middleware('http')
async def password_gate(request,call_next):
    # Optional shared login for a public deployment (APP_PASSWORD set): the browser asks once and sends it with every request.
    if PASSWORD and request.url.path!='/health' and not request.url.path.startswith('/api/v1/worker/'):  # the worker has its own token
        try:user,_,pw=base64.b64decode(request.headers.get('authorization','').removeprefix('Basic ')).decode('utf-8','replace').partition(':')
        except ValueError:user=pw=''
        if not (secrets.compare_digest(user,USER) and secrets.compare_digest(pw,PASSWORD)):
            return Response('Login required',401,{'WWW-Authenticate':'Basic realm="AeroMind", charset="UTF-8"'})
    return await call_next(request)
class SyncedFiles(StaticFiles):
    """Static files that are downloaded from the bucket on first request when they aren't on this machine yet."""
    async def get_response(self,path,scope):
        full,_=self.lookup_path(path)
        if not full and storage.enabled() and '..' not in path:await run_in_threadpool(storage.fetch,Path(self.directory)/path)
        return await super().get_response(path,scope)
app.include_router(gpu.router)
app.mount('/uploads',SyncedFiles(directory=ROOT),name='uploads')
reconstruct.RECON_DIR.mkdir(parents=True,exist_ok=True)
app.mount('/recon',SyncedFiles(directory=reconstruct.RECON_DIR),name='recon')  # 3D outputs
vendor.VENDOR_REFERENCE_DIR.mkdir(parents=True,exist_ok=True)
app.mount('/vendor-refs',SyncedFiles(directory=vendor.VENDOR_REFERENCE_DIR),name='vendor-refs')  # reference library

def get_db():
    with SessionLocal() as db: yield db

def to_dict(x): return {c.name:getattr(x,c.name) for c in x.__table__.columns}
class SiteIn(BaseModel):
    code:str=Field(min_length=2,max_length=80)
    name:str=Field(min_length=2,max_length=200)
    region:str='Unknown'
    tower_type:str='Lattice'
    height_m:float|None=None
    latitude:float|None=None
    longitude:float|None=None
class InspectionIn(BaseModel):
    site_id:int
    label:str=Field(min_length=2,max_length=200)
class DefectDecision(BaseModel):
    status:str
    notes:str=''
class BoxIn(BaseModel):
    detection_id:int|None=None  # model detection this box was kept/edited from; None for hand-drawn boxes
    class_name:str=Field(min_length=1,max_length=100)
    x1:float
    y1:float
    x2:float
    y2:float
    manufacturer:str|None=Field(None,max_length=100)
    product_model:str|None=Field(None,max_length=100)
    vendor_source:str|None=None
    vendor_confidence:float|None=Field(None,ge=0,le=1)
    vendor_evidence:str|None=Field(None,max_length=500)
    vendor_basis:str|None=Field(None,max_length=24)
    condition_grade:str|None=None
    condition_score:int|None=Field(None,ge=0,le=100)
    condition_defects:list[dict]|None=None
    condition_source:str|None=None
    condition_confidence:float|None=Field(None,ge=0,le=1)
    condition_verified:str|None=None
    condition_summary:str|None=Field(None,max_length=600)
class VendorBoxIn(BaseModel):
    class_name:str
    x1:float
    y1:float
    x2:float
    y2:float
    image_width:int=Field(gt=0)
    image_height:int=Field(gt=0)
class ReviewIn(BaseModel):
    workflow:str='manual_qc'
    job_id:int|None=None  # detections of this run that aren't kept are marked rejected
    frame_index:int=0
    image_width:int=Field(gt=0)
    image_height:int=Field(gt=0)
    notes:str=''
    boxes:list[BoxIn]=[]
class AssetUpdate(BaseModel):
    condition:str|None=None
    identity_status:str|None=None
    localization_status:str|None=None
    x:float|None=None
    y:float|None=None
    z:float|None=None
class RegistrationReviewIn(BaseModel):
    status:str
class RegistrationBulkReviewIn(BaseModel):
    candidate_ids:list[int]=Field(min_length=1,max_length=10000)
    status:str
class RegistrationGeometryIn(BaseModel):
    x:float
    y:float
    z:float
    yaw:float|None=None

def detector_state():
    """YOLO status/model of whoever runs detection: this process, or the remote GPU worker."""
    if gpu.REMOTE:return gpu.info('detector',{'status':'worker_offline','model':None})
    return {'status':detector.status(),'model':detector.model_name() if detector.MODEL_PATH else None}
@app.get('/health')
def health():
    d=detector_state();w=gpu.worker() if gpu.REMOTE else None
    return {'status':'ok','mode':'starter','ai_inference':d['status'],'model':d.get('model'),'gpu_worker':gpu.MODE,
            **({'worker':{'name':w['name'],'online':w['online'],'seen_at':w['seen_at'],'gpu':w['info'].get('gpu')} if w else None} if gpu.REMOTE else {}),
            'storage':'bucket' if storage.enabled() else 'local'}
@app.get('/api/v1/detector')
def detector_info():
    if gpu.REMOTE:d=dict(detector_state());d.setdefault('loaded',False);return d
    return detector.info(load=True)
@app.get('/api/v1/dashboard')
def dashboard(db:Session=Depends(get_db)):
    issues=db.scalar(select(func.count()).select_from(Annotation).where(Annotation.condition_grade.in_(['Fair','Poor','Critical'])))
    return {'approved_images':db.scalar(select(func.count()).select_from(ImageReview)),'observed_items':db.scalar(select(func.count()).select_from(Annotation)),'items_with_defects':issues,'sites':db.scalar(select(func.count()).select_from(Site)), 'inspections':db.scalar(select(func.count()).select_from(Inspection)), 'assets':db.scalar(select(func.count()).select_from(Asset)), 'defects':db.scalar(select(func.count()).select_from(Defect)), 'qa_pending':db.scalar(select(func.count()).select_from(Defect).where(Defect.review_status=='pending'))}
@app.get('/api/v1/sites')
def sites(db:Session=Depends(get_db)): return [to_dict(x) for x in db.scalars(select(Site).order_by(Site.id)).all()]
@app.post('/api/v1/sites',status_code=201)
def create_site(data:SiteIn,db:Session=Depends(get_db)):
    if db.scalar(select(Site).where(Site.code==data.code)): raise HTTPException(409,'Site code already exists')
    x=Site(**data.model_dump());db.add(x);db.commit();db.refresh(x);return to_dict(x)
@app.get('/api/v1/sites/{site_id}')
def site_detail(site_id:int,db:Session=Depends(get_db)):
    x=db.get(Site,site_id)
    if not x:raise HTTPException(404,'Site not found')
    return to_dict(x)
@app.get('/api/v1/inspections')
def inspections(site_id:int|None=None,db:Session=Depends(get_db)):
    q=select(Inspection)
    if site_id:q=q.where(Inspection.site_id==site_id)
    return [to_dict(x) for x in db.scalars(q.order_by(Inspection.id.desc())).all()]
@app.post('/api/v1/inspections',status_code=201)
def create_inspection(data:InspectionIn,db:Session=Depends(get_db)):
    if not db.get(Site,data.site_id):raise HTTPException(404,'Site not found')
    x=Inspection(**data.model_dump());db.add(x);db.commit();db.refresh(x);return to_dict(x)
@app.post('/api/v1/inspections/{inspection_id}/media',status_code=201)
async def upload(inspection_id:int, file:UploadFile=File(...),db:Session=Depends(get_db)):
    if not db.get(Inspection,inspection_id):raise HTTPException(404,'Inspection not found')
    if file.content_type not in ALLOWED:raise HTTPException(415,'Unsupported image/video type')
    key=uuid.uuid4().hex+ALLOWED[file.content_type]
    target=ROOT/key;size=0;sha=hashlib.sha256()
    try:
        with target.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_SIZE:raise HTTPException(413,'File too large')
                sha.update(chunk);out.write(chunk)
        await run_in_threadpool(storage.push,target)
        x=Media(inspection_id=inspection_id,filename=Path(file.filename or 'upload').name[:255],storage_key=key,content_type=file.content_type,sha256=sha.hexdigest(),size_bytes=size)
        db.add(x);db.commit();db.refresh(x);return to_dict(x)
    except Exception:
        target.unlink(missing_ok=True)
        raise
@app.get('/api/v1/inspections/{inspection_id}/media')
def media(inspection_id:int,db:Session=Depends(get_db)):
    return [to_dict(m)|{'url':f'/uploads/{m.storage_key}'} for m in db.scalars(select(Media).where(Media.inspection_id==inspection_id)).all()]
@app.post('/api/v1/inspections/{inspection_id}/process',status_code=202)
def process(inspection_id:int,background:BackgroundTasks,media_ids:list[int]|None=Query(None),only_new:bool=False,db:Session=Depends(get_db)):
    """Run YOLO over the inspection's uploaded files (videos are split into frames first).
    media_ids limits the run to those uploads; only_new skips uploads already covered by a completed run."""
    if not db.get(Inspection,inspection_id):raise HTTPException(404,'Inspection not found')
    if (state:=detector_state()['status'])!='ready':
        why='the GPU worker is offline: start backend/run-worker.ps1 on the GPU machine' if state=='worker_offline' else f'YOLO detector {state}'
        x=Job(inspection_id=inspection_id,status='awaiting_worker',stage='not_configured',detail=f'{why[0].upper()+why[1:]}. This starter does not fabricate detections or model reconstructions.')
        db.add(x);db.commit();db.refresh(x);return to_dict(x)
    top=db.scalars(select(Media.id).where(Media.inspection_id==inspection_id,Media.parent_media_id.is_(None))).all()
    if not top:raise HTTPException(409,'Upload media before processing')
    scope=None
    if media_ids:
        scope=[i for i in media_ids if i in top]
        if not scope:raise HTTPException(422,'None of these media are uploads of this inspection')
    if only_new:
        done=completed_runs(db,inspection_id);scope=[i for i in (scope or top) if not any(covers(j,i) for j in done)]
        if not scope:raise HTTPException(409,'Nothing new to process: every upload already has detections')
    if scope is None:scope=list(top)  # a full run still records its files (None only on runs made before this change)
    x=Job(inspection_id=inspection_id,kind='detection',status='queued',stage='detection',media_ids=scope,
          detail=f'Queued for YOLO detection ({detector_state().get("model")}) · '+(f'{len(scope)} file(s)' if scope is not None else 'all files'))
    db.add(x);db.commit();db.refresh(x);gpu.start(background,'run_detection',job_id=x.id);return to_dict(x)
def completed_runs(db,inspection_id):
    return db.scalars(select(Job).where(Job.inspection_id==inspection_id,Job.kind=='detection',Job.status=='completed').order_by(Job.id.desc())).all()
def covers(job,upload_id):return job.media_ids is None or upload_id in job.media_ids
# Detection job: prepare (files and existing video frames) -> tasks.detect on the GPU -> apply (frames + boxes)
def _job_progress(db,stage,pct,detail,job_id,**_):
    job=db.get(Job,job_id);job.stage=stage;job.progress=max(0,min(100,int(pct)));job.detail=detail;db.commit()
def _detection_prepare(db,job_id):
    job=db.get(Job,job_id);q=select(Media).where(Media.inspection_id==job.inspection_id,Media.parent_media_id.is_(None))
    items=db.scalars((q if job.media_ids is None else q.where(Media.id.in_(job.media_ids))).order_by(Media.id)).all()
    job.status='running';db.commit()
    def frames_of(m):  # frames extracted by an earlier run are reused
        kids=db.scalars(select(Media).where(Media.parent_media_id==m.id).order_by(Media.frame_index)).all()
        return [{'id':k.id,'storage_key':k.storage_key} for k in kids] or None
    return {'model':detector_state().get('model'),'items':[{'id':m.id,'filename':m.filename,'content_type':m.content_type,'storage_key':m.storage_key,
            'frames':frames_of(m) if m.content_type.startswith('video/') else None} for m in items]}
def _detection_apply(db,result,job_id):
    job=db.get(Job,job_id);name=detector_state().get('model') or 'yolo';counts=Counter();nframes=0;failed,ok_ids=[],[]
    for f in result['files']:
        if not f['ok']:failed.append(f['error']);continue
        video=db.get(Media,f['id']);by_index={}
        if f['new_frames']:
            have={k.frame_index:k for k in db.scalars(select(Media).where(Media.parent_media_id==video.id))}  # another run may have made them meanwhile
            for fr in f['new_frames']:
                if fr['frame_index'] not in have:
                    have[fr['frame_index']]=Media(inspection_id=video.inspection_id,parent_media_id=video.id,content_type='image/jpeg',**fr);db.add(have[fr['frame_index']])
            db.flush();by_index={i:k.id for i,k in have.items()}
        for t in f['targets']:
            media_id=t['media_id'] if 'media_id' in t else by_index[t['frame_index']]
            nframes+=media_id!=f['id']
            db.add_all(Detection(job_id=job.id,inspection_id=job.inspection_id,media_id=media_id,frame_index=0,image_width=t['w'],image_height=t['h'],model_name=name,**b) for b in t['boxes'])
            counts.update(b['class_name'] for b in t['boxes'])
        ok_ids.append(f['id'])
    if failed and not ok_ids:raise RuntimeError('; '.join(failed))
    top=', '.join(f'{k} {v}' for k,v in counts.most_common(5))
    job.media_ids=ok_ids;job.status='completed';job.progress=100;job.stage='detection'  # failed files stay "not processed", so only_new retries them
    job.detail=f'{counts.total()} detections in {len(ok_ids)} file(s)'+(f' incl. {nframes} video frames' if nframes else '')+f' · {name}'+(f' · {top}' if top else '')+(f' · FAILED {len(failed)}: '+'; '.join(failed) if failed else '')
    db.commit();return {'job_id':job.id,'status':job.status}
def _detection_fail(db,error,job_id):
    db.rollback();job=db.get(Job,job_id);job.status='failed';job.detail=f"Detection failed: {error['detail']}"[:2000];db.commit()
gpu.register('run_detection',prepare=_detection_prepare,apply=_detection_apply,fail=_detection_fail,progress=_job_progress)
def run_detection(job_id:int):gpu.run_local('run_detection',{'job_id':job_id})
@app.get('/api/v1/jobs')
def jobs(db:Session=Depends(get_db)):return [to_dict(x) for x in db.scalars(select(Job).order_by(Job.id.desc())).all()]
@app.get('/api/v1/jobs/{job_id}')
def job_detail(job_id:int,db:Session=Depends(get_db)):
    x=db.get(Job,job_id)
    if not x:raise HTTPException(404,'Job not found')
    return to_dict(x)
@app.get('/api/v1/inspections/{inspection_id}/detections')
def detections(inspection_id:int,job_id:int|None=None,media_id:int|None=None,class_name:str|None=None,min_conf:float=0,db:Session=Depends(get_db)):
    q=select(Detection).where(Detection.inspection_id==inspection_id,Detection.confidence>=min_conf)
    if job_id is not None:q=q.where(Detection.job_id==job_id)
    else:  # per file, the latest completed run that covered it (runs can be limited to new uploads)
        runs=completed_runs(db,inspection_id);by_job={}
        for m in db.scalars(select(Media).where(Media.inspection_id==inspection_id)).all():
            j=next((j for j in runs if covers(j,m.parent_media_id or m.id)),None)
            if j:by_job.setdefault(j.id,[]).append(m.id)
        if not by_job:return []
        q=q.where(or_(*(and_(Detection.job_id==j,Detection.media_id.in_(ms)) for j,ms in by_job.items())))
    if media_id:q=q.where(Detection.media_id==media_id)
    if class_name:q=q.where(Detection.class_name==class_name)
    return [to_dict(x) for x in db.scalars(q.order_by(Detection.media_id,Detection.frame_index,Detection.confidence.desc())).all()]
def class_catalog(db):
    # model classes first, then any names already used in detections or approved annotations
    model={int(k):v for k,v in gpu.info('detector',{}).get('classes',{}).items()} if gpu.REMOTE else detector.class_names()
    names={v:k for k,v in model.items()}
    for m in (Detection,Annotation):
        for name,cid in db.execute(select(m.class_name,m.class_id).distinct()):names.setdefault(name,cid)
    return names
@app.get('/api/v1/classes')
def classes(db:Session=Depends(get_db)):return [{'id':i,'name':n} for n,i in sorted(class_catalog(db).items(),key=lambda x:(x[1],x[0]))]
@app.get('/api/v1/inspections/{inspection_id}/reviews')
def reviews(inspection_id:int,db:Session=Depends(get_db)):
    return [to_dict(x) for x in db.scalars(select(ImageReview).where(ImageReview.inspection_id==inspection_id).order_by(ImageReview.media_id,ImageReview.frame_index)).all()]
@app.get('/api/v1/media/{media_id}/annotations')
def media_annotations(media_id:int,frame_index:int=0,db:Session=Depends(get_db)):
    if not db.get(Media,media_id):raise HTTPException(404,'Media not found')
    r=db.scalar(select(ImageReview).where(ImageReview.media_id==media_id,ImageReview.frame_index==frame_index))
    return {'review':to_dict(r) if r else None,'annotations':[to_dict(a) for a in db.scalars(select(Annotation).where(Annotation.review_id==r.id).order_by(Annotation.id)).all()] if r else []}
def vendor_fields(b,class_name):
    # manufacturer only applies to vendor classes; a box relabelled to e.g. bolts drops it
    m=(b.manufacturer or '').strip()
    if class_name not in vendor.VENDOR_CLASSES or not m:return {}
    ai=b.vendor_source=='ai'
    return {'manufacturer':m,'product_model':(b.product_model or '').strip() or None,'vendor_source':'ai' if ai else 'reviewer','vendor_confidence':b.vendor_confidence if ai else None,'vendor_evidence':b.vendor_evidence if ai or b.vendor_basis=='reviewer_choice' else None,
            'vendor_basis':b.vendor_basis if ai or b.vendor_basis=='reviewer_choice' else None}  # keep the record of which AI candidates a reviewer picked from
def condition_fields(b):
    if b.condition_grade not in condition.GRADES:return {}
    ai=b.condition_source=='ai';keys=('type','severity','extent_percent','location','evidence','source','confirmed')
    defects=[{k:d.get(k) for k in keys if k in d} for d in (b.condition_defects or []) if d.get('type') in condition.DEFECT_TYPES][:20]
    return {'condition_grade':b.condition_grade,'condition_score':b.condition_score,'condition_defects':defects,'condition_source':'ai' if ai else 'reviewer',
            'condition_confidence':b.condition_confidence if ai else None,'condition_verified':b.condition_verified if ai else None,'condition_summary':(b.condition_summary or '')[:600] or None}
@app.put('/api/v1/media/{media_id}/annotations')
def approve_annotations(media_id:int,data:ReviewIn,db:Session=Depends(get_db)):
    """Approve the reviewed box set for one image: replaces its annotations and marks the model detections accepted/corrected/rejected."""
    m=db.get(Media,media_id)
    if not m:raise HTTPException(404,'Media not found')
    if data.workflow not in WORKFLOWS:raise HTTPException(422,'Invalid workflow')
    if data.job_id is not None and getattr(db.get(Job,data.job_id),'inspection_id',None)!=m.inspection_id:raise HTTPException(422,'Job does not belong to this inspection')
    ids=[b.detection_id for b in data.boxes if b.detection_id is not None]
    if len(set(ids))!=len(ids):raise HTTPException(422,'A detection can only be kept once')
    dets={d.id:d for d in db.scalars(select(Detection).where(Detection.id.in_(ids),Detection.media_id==media_id,Detection.frame_index==data.frame_index)).all()}
    if len(dets)!=len(ids):raise HTTPException(422,'Unknown detection for this image')
    W,H=data.image_width,data.image_height;names=class_catalog(db)
    r=db.scalar(select(ImageReview).where(ImageReview.media_id==media_id,ImageReview.frame_index==data.frame_index))
    if not r:r=ImageReview(inspection_id=m.inspection_id,media_id=media_id,frame_index=data.frame_index);db.add(r)
    r.status='approved';r.workflow=data.workflow;r.job_id=data.job_id;r.box_count=len(data.boxes);r.notes=data.notes;r.reviewed_at=now();db.flush()
    db.execute(delete(Annotation).where(Annotation.review_id==r.id))
    for b in data.boxes:
        x1,x2=sorted(min(max(v,0),W) for v in (b.x1,b.x2));y1,y2=sorted(min(max(v,0),H) for v in (b.y1,b.y2))
        if x2-x1<1 or y2-y1<1:db.rollback();raise HTTPException(422,f'Box for {b.class_name} is smaller than 1 px')
        name=b.class_name.strip()
        if name not in names:names[name]=max(names.values(),default=-1)+1
        d=dets.get(b.detection_id)
        same=d is not None and d.class_name==name and all(abs(getattr(d,k)-v)<0.5 for k,v in zip(('x1','y1','x2','y2'),(x1,y1,x2,y2)))
        if d:d.review_status='accepted' if same else 'corrected'
        v=vendor_fields(b,name)|condition_fields(b)
        db.add(Annotation(review_id=r.id,inspection_id=m.inspection_id,media_id=media_id,frame_index=data.frame_index,detection_id=b.detection_id,source='manual' if d is None else 'model' if same else 'model_edited',class_id=names[name],class_name=name,confidence=d.confidence if d else None,x1=x1,y1=y1,x2=x2,y2=y2,image_width=W,image_height=H,**v))
    if data.job_id is not None:
        db.execute(update(Detection).where(Detection.job_id==data.job_id,Detection.media_id==media_id,Detection.frame_index==data.frame_index,Detection.id.not_in(ids)).values(review_status='rejected'))
    db.commit();return media_annotations(media_id,data.frame_index,db)
def _annotate():
    try:
        from . import annotate
    except ImportError:raise HTTPException(503,'Annotated output needs Pillow: pip install -r requirements.txt')
    return annotate
def output_boxes(db,m,source,job_id=None,min_conf=0.0):
    """(boxes, size the boxes are measured in, source used) to draw on one image."""
    if source in ('auto','approved'):
        r=db.scalar(select(ImageReview).where(ImageReview.media_id==m.id,ImageReview.frame_index==0))
        if r:
            anns=db.scalars(select(Annotation).where(Annotation.review_id==r.id)).all()
            return [to_dict(a) for a in anns],(anns[0].image_width,anns[0].image_height) if anns else None,'approved'
        if source=='approved':return None,None,'approved'
    if job_id is None:  # the latest completed run that covered this file (runs can be limited to new uploads)
        job_id=next((j.id for j in completed_runs(db,m.inspection_id) if covers(j,m.parent_media_id or m.id)),None)
    dets=db.scalars(select(Detection).where(Detection.job_id==job_id,Detection.media_id==m.id,Detection.frame_index==0,Detection.confidence>=min_conf)).all()
    return [to_dict(d) for d in dets],(dets[0].image_width,dets[0].image_height) if dets else None,'model'
def output_name(m):
    p=Path(m.filename);return f'{p.stem}_detections{p.suffix.lower() or ".jpg"}'
@app.get('/api/v1/media/{media_id}/annotated')
def annotated_image(media_id:int,source:str='auto',job_id:int|None=None,min_conf:float=0,labels:bool=True,db:Session=Depends(get_db)):
    """The image with its boxes drawn on and its original metadata (EXIF/GPS/XMP/ICC/IPTC) kept.
    source: approved = QC'd annotations, model = detector output, auto = approved if the image was reviewed, else model."""
    if source not in {'auto','model','approved'}:raise HTTPException(422,'source must be auto, model or approved')
    m=db.get(Media,media_id)
    if not m:raise HTTPException(404,'Media not found')
    if not m.content_type.startswith('image/'):raise HTTPException(422,'Annotated output is available for images only')
    annotate=_annotate();boxes,seen,used=output_boxes(db,m,source,job_id,min_conf)
    if boxes is None:raise HTTPException(404,'Image has no approved annotations')
    data,mime=annotate.render(storage.fetch(ROOT/m.storage_key),boxes,seen,labels)
    return Response(data,media_type=mime,headers={'Content-Disposition':f"attachment; filename*=UTF-8''{quote(output_name(m))}",'X-Annotation-Source':used,'X-Box-Count':str(len(boxes))})
@app.get('/api/v1/inspections/{inspection_id}/annotated.zip')
def annotated_zip(inspection_id:int,source:str='auto',min_conf:float=0,labels:bool=True,db:Session=Depends(get_db)):
    """Every image of an inspection with boxes drawn and metadata kept. With source=approved, unreviewed images are skipped."""
    if source not in {'auto','model','approved'}:raise HTTPException(422,'source must be auto, model or approved')
    items=[m for m in db.scalars(select(Media).where(Media.inspection_id==inspection_id).order_by(Media.id)).all() if m.content_type.startswith('image/')]
    if not items:raise HTTPException(404,'No images in this inspection')
    annotate=_annotate();tmp=tempfile.NamedTemporaryFile(suffix='.zip',delete=False);names=set();written=0
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:  # images are already compressed
            for m in items:
                boxes,seen,_=output_boxes(db,m,source,None,min_conf)
                if boxes is None:continue
                name=output_name(m)
                if name in names:name=f'{Path(name).stem}_{m.id}{Path(name).suffix}'
                names.add(name);z.writestr(name,annotate.render(storage.fetch(ROOT/m.storage_key),boxes,seen,labels)[0]);written+=1
        tmp.close()
    except Exception:
        tmp.close();os.unlink(tmp.name);raise
    if not written:os.unlink(tmp.name);raise HTTPException(404,'No approved images in this inspection')
    return FileResponse(tmp.name,media_type='application/zip',filename=f'INS-{inspection_id:04d}_annotated.zip',background=BackgroundTask(os.unlink,tmp.name))
def reviewer_tags(db,class_name):
    return db.scalars(select(Annotation).where(Annotation.class_name==class_name,Annotation.vendor_source=='reviewer',Annotation.manufacturer.is_not(None),Annotation.manufacturer!='Unknown').order_by(Annotation.id.desc())).all()
@app.get('/api/v1/vendors')
def vendors(db:Session=Depends(get_db)):
    d=dict(gpu.info('vendor',{})) if gpu.REMOTE else vendor.info()
    if not d.get('classes'):  # remote worker offline: the tagging lists still work, only the AI is unavailable
        d={'classes':vendor.VENDOR_CLASSES,'manufacturers':{c:vendor.manufacturers(c)+['Unknown'] for c in vendor.VENDOR_CLASSES},
           'ai':{'status':'worker_offline','model':vendor.VLM_MODEL or None,'compare_model':None,'url':None},'folder_references':{},'library_manufacturers':{}}
    d['reviewer_references']={c:len(reviewer_tags(db,c)) for c in vendor.VENDOR_CLASSES};return d
def vendor_references(db,class_name,media_id,box):
    """Reviewer-confirmed crops of known equipment [{manufacturer, storage_key, box, size}], newest first. The box being
    identified is left out so it can't match itself; catalog photos are added by the GPU side (tasks.vendor_suggestion)."""
    refs,taken=[],{}
    for a in reviewer_tags(db,class_name):
        if taken.get(a.manufacturer,0)>=vendor.REFS_PER_VENDOR:continue
        ix=max(0,min(a.x2,box.x2)-max(a.x1,box.x1))*max(0,min(a.y2,box.y2)-max(a.y1,box.y1))
        if a.media_id==media_id and ix>0.5*min((a.x2-a.x1)*(a.y2-a.y1),(box.x2-box.x1)*(box.y2-box.y1)):continue
        m=db.get(Media,a.media_id)
        if not m:continue
        refs.append({'manufacturer':a.manufacturer,'storage_key':m.storage_key,'box':{k:getattr(a,k) for k in ('x1','y1','x2','y2')},'size':[a.image_width,a.image_height]})
        taken[a.manufacturer]=taken.get(a.manufacturer,0)+1
    return refs
@app.get('/api/v1/vendor-references')
def vendor_reference_library(class_name:str,manufacturer:str|None=None):
    """Reference photos for the reviewer to compare against; match=false views are shown but not used by the AI."""
    return [{k:r.get(k,'') for k in ('manufacturer','model','view','status','page','notes')}|{'match':r['match'],'url':'/vendor-refs/'+quote(r['file'])}
            for r in sorted(vendor.library(class_name),key=vendor.view_rank) if manufacturer is None or r['manufacturer'].lower()==manufacturer.lower()]
@app.post('/api/v1/media/{media_id}/vendor-suggestion')
def vendor_suggestion(media_id:int,box:VendorBoxIn):
    """Ask the local VLM which manufacturer made the equipment in one box. A suggestion only; nothing is saved."""
    return gpu.call('vendor_suggestion',media_id=media_id,box=box.model_dump())
def _vendor_prepare(db,media_id,box):
    m=db.get(Media,media_id);box=VendorBoxIn(**box)
    if not m:raise HTTPException(404,'Media not found')
    if box.class_name not in vendor.VENDOR_CLASSES:raise HTTPException(422,f'Manufacturer suggestions cover {", ".join(vendor.VENDOR_CLASSES)}')
    return {'storage_key':m.storage_key,'box':box.model_dump(),'class_name':box.class_name,'refs':vendor_references(db,box.class_name,media_id,box)}
gpu.register('vendor_suggestion',prepare=_vendor_prepare,apply=lambda db,result,**_:result,lane='quick')
@app.post('/api/v1/media/{media_id}/condition-assessment')
def condition_assessment(media_id:int,box:VendorBoxIn):
    """Condition of the equipment in one box from the local VLM, re-checked by a second model when defects are found. Nothing is saved."""
    return gpu.call('condition_assessment',media_id=media_id,box=box.model_dump())
def _condition_prepare(db,media_id,box):
    m=db.get(Media,media_id)
    if not m:raise HTTPException(404,'Media not found')
    return {'storage_key':m.storage_key,'box':VendorBoxIn(**box).model_dump()}
gpu.register('condition_assessment',prepare=_condition_prepare,apply=lambda db,result,**_:result,lane='quick')
def defects_text(ds):return '; '.join(f"{d.get('type')} {d.get('severity','')} ~{d.get('extent_percent','?')}% @ {d.get('location','')}" for d in ds or [])
def rust_percent(ds):return max([d.get('extent_percent') or 0 for d in ds if d.get('type') in ('rust','corrosion')],default=0) if ds is not None else None
@app.get('/api/v1/inspections/{inspection_id}/annotations.csv')
def annotations_csv(inspection_id:int,db:Session=Depends(get_db)):
    """Approved annotations of an inspection, one row per box, including manufacturer tags."""
    output=io.StringIO();w=csv.writer(output)
    w.writerow(['image','media_id','class','x1','y1','x2','y2','box_source','model_confidence','manufacturer','product_model','vendor_source','vendor_confidence','vendor_basis','vendor_evidence','condition_grade','health_score','rust_percent','defects','condition_source','condition_verified','condition_summary','reviewed_at','workflow'])
    for a,m,r in db.execute(select(Annotation,Media,ImageReview).join(Media,Media.id==Annotation.media_id).join(ImageReview,ImageReview.id==Annotation.review_id).where(Annotation.inspection_id==inspection_id).order_by(Media.id,Annotation.id)):
        w.writerow([m.filename,m.id,a.class_name,round(a.x1,1),round(a.y1,1),round(a.x2,1),round(a.y2,1),a.source,a.confidence,a.manufacturer,a.product_model,a.vendor_source,a.vendor_confidence,a.vendor_basis,a.vendor_evidence,a.condition_grade,a.condition_score,rust_percent(a.condition_defects),defects_text(a.condition_defects),a.condition_source,a.condition_verified,a.condition_summary,r.reviewed_at.isoformat(),r.workflow])
    return StreamingResponse(iter([output.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="INS-{inspection_id:04d}_annotations.csv"'})
@app.get('/api/v1/sites/{site_id}/inventory')
def site_inventory(site_id:int,inspection_id:int|None=None,db:Session=Depends(get_db)):
    """Observed equipment of a tower from approved images: per image, its approved boxes with manufacturer and condition,
    plus summaries. One physical asset seen in two images appears twice - matching them is identity reconciliation, not done here."""
    site=db.get(Site,site_id)
    if not site:raise HTTPException(404,'Site not found')
    insp=db.scalars(select(Inspection).where(Inspection.site_id==site_id).order_by(Inspection.id.desc())).all()
    ids=[i.id for i in insp if inspection_id in (None,i.id)]
    allm=db.scalars(select(Media).where(Media.inspection_id.in_(ids)).order_by(Media.inspection_id.desc(),Media.id)).all();names={m.id:m.filename for m in allm}
    media=[m for m in allm if m.content_type.startswith('image/')]
    reviews={r.media_id:r for r in db.scalars(select(ImageReview).where(ImageReview.inspection_id.in_(ids),ImageReview.frame_index==0)).all()}
    items={}
    for a in db.scalars(select(Annotation).where(Annotation.review_id.in_([r.id for r in reviews.values()])).order_by(Annotation.id)).all():items.setdefault(a.media_id,[]).append(to_dict(a))
    by_class,grades,makers,defect_types,verified={},Counter(),Counter(),Counter(),Counter()
    images=[]
    for m in media:
        r=reviews.get(m.id);its=items.get(m.id,[])
        for a in its:
            c=by_class.setdefault(a['class_name'],{'class_name':a['class_name'],'class_id':a['class_id'],'count':0,'manufacturers':Counter(),'grades':Counter(),'defects':0})
            g=a['condition_grade'] or 'Not assessed';c['count']+=1;c['grades'][g]+=1;grades[g]+=1
            if a['manufacturer']:c['manufacturers'][a['manufacturer']]+=1;makers[a['manufacturer']]+=1
            for d in a['condition_defects'] or []:defect_types[d.get('type')]+=1;c['defects']+=1
            if a['condition_verified']:verified[a['condition_verified']]+=1
        images.append({'media_id':m.id,'inspection_id':m.inspection_id,'filename':m.filename,'url':f'/uploads/{m.storage_key}','status':r.status if r else 'pending',
                       'video':names.get(m.parent_media_id),'frame_index':m.frame_index,'frame_time':m.frame_time,
                       'reviewed_at':r.reviewed_at if r else None,'workflow':r.workflow if r else None,'notes':r.notes if r else '',
                       'image_width':its[0]['image_width'] if its else None,'image_height':its[0]['image_height'] if its else None,'items':its})
    issues=sum(grades[g] for g in ('Fair','Poor','Critical'))
    return {'site':to_dict(site),'inspections':[to_dict(i) for i in insp],'images':images,'summary':{
        'images_total':len(images),'images_approved':sum(i['status']=='approved' for i in images),'items':sum(len(i['items']) for i in images),
        'items_with_defects':issues,'grades':dict(grades),'manufacturers':dict(makers.most_common()),'defect_types':dict(defect_types.most_common()),'verified':dict(verified),
        'by_class':sorted(({**c,'manufacturers':dict(c['manufacturers'].most_common()),'grades':dict(c['grades'])} for c in by_class.values()),key=lambda c:-c['count'])}}
def model_dict(x):
    d=to_dict(x);files=(x.quality or {}).get('files',{});base=f'/recon/{x.inspection_id}_{x.job_id}/'
    return d|{'urls':{k:base+v for k,v in files.items()}}
@app.get('/api/v1/reconstruction')
def reconstruction_info():return gpu.info('reconstruction',{'status':'worker_offline','colmap':None}) if gpu.REMOTE else reconstruct.info()
def colmap_state():return reconstruction_info().get('status','worker_offline')
@app.get('/api/v1/inspections/{inspection_id}/models')
def models_3d(inspection_id:int,db:Session=Depends(get_db)):
    return [model_dict(x) for x in db.scalars(select(Model3D).where(Model3D.inspection_id==inspection_id,Model3D.job_id.is_not(None)).order_by(Model3D.id.desc())).all()]
@app.get('/api/v1/inspections/{inspection_id}/model')
def model(inspection_id:int,db:Session=Depends(get_db)):
    x=db.scalar(select(Model3D).where(Model3D.inspection_id==inspection_id,Model3D.status=='completed').order_by(Model3D.id.desc()).limit(1))
    return model_dict(x) if x else {'status':'not_available','message':'No 3D model yet: build one on the 3D Reconstruction page.'}
@app.get('/api/v1/models/{model_id}/registrations')
def registrations(model_id:int,db:Session=Depends(get_db)):
    return [to_dict(x) for x in db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model_id).order_by(RegistrationCandidate.class_name,RegistrationCandidate.id)).all()]
REGISTRATION_REVIEW_STATUSES={'approved','rejected','needs_adjustment','needs_review'}
def _registration_status(status:str):
    if status not in REGISTRATION_REVIEW_STATUSES:raise HTTPException(422,'3D box status must be approved, rejected, needs_adjustment or needs_review')
    return status
def _registration_payload(candidate):
    data=dict(candidate);position=data.pop('position')
    data.update(x=float(position[0]),y=float(position[2]),z=float(position[1]))
    for key in ('confidence','spread','size_w','size_h','size_d','yaw','fit_iou'):
        if data.get(key) is not None:data[key]=float(data[key])
    for key in ('evidence_count','media_count','point_count','box_points'):
        if data.get(key) is not None:data[key]=int(data[key])
    data['detection_ids']=[int(value) for value in data.get('detection_ids',[])]
    return data
def _registration_match_score(row,data):
    if row.class_name!=data['class_name']:return None
    old_ids=set(row.detection_ids or []);new_ids=set(data.get('detection_ids') or []);shared=len(old_ids&new_ids)
    distance=math.sqrt((row.x-data['x'])**2+(row.y-data['y'])**2+(row.z-data['z'])**2)
    if shared:return (2,shared,shared/max(1,len(old_ids|new_ids)),-distance)
    old_diag=math.sqrt(sum(float(getattr(row,key) or 0)**2 for key in ('size_w','size_h','size_d')))
    new_diag=math.sqrt(sum(float(data.get(key) or 0)**2 for key in ('size_w','size_h','size_d')))
    close=distance<=max(.015,.30*min(old_diag,new_diag)) if old_diag and new_diag else distance<=.015
    return (1,0,0,-distance) if close else None
def reconcile_registrations(db,model,inspection,candidates):
    """Stable upsert: preserve reviewed decisions and IDs; insert only genuinely new physical candidates."""
    existing=list(db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model.id)))
    locked=[row for row in existing if row.status in {'approved','rejected','needs_adjustment'}];mutable=[row for row in existing if row.status not in {'approved','rejected','needs_adjustment'}]
    used=set();created=updated=preserved=0
    for candidate in candidates:
        data=_registration_payload(candidate)
        # An approved/rejected physical box is authoritative. A rebuild must never create its regenerated twin.
        if any(_registration_match_score(row,data) is not None for row in locked):preserved+=1;continue
        matches=[(score,row) for row in mutable if row.id not in used and (score:=_registration_match_score(row,data)) is not None]
        if matches:
            _,row=max(matches,key=lambda item:item[0]);used.add(row.id)
            for key,value in data.items():setattr(row,key,value)
            updated+=1
        else:
            row=RegistrationCandidate(model_id=model.id,inspection_id=inspection.id,site_id=inspection.site_id,status='needs_review',**data)
            db.add(row);created+=1
    removed=0
    for row in mutable:
        if row.id not in used:
            if row.status=='needs_review':db.delete(row);removed+=1
            else:preserved+=1  # keep an explicit needs-adjustment decision even if a rebuild cannot reproduce it
    return {'created':created,'updated':updated,'preserved':preserved,'removed':removed}
@app.patch('/api/v1/registrations/{candidate_id}')
def review_registration(candidate_id:int,data:RegistrationReviewIn,db:Session=Depends(get_db)):
    row=db.get(RegistrationCandidate,candidate_id)
    if not row:raise HTTPException(404,'3D registration candidate not found')
    row.status=_registration_status(data.status);db.commit();db.refresh(row);return to_dict(row)
@app.patch('/api/v1/registrations/{candidate_id}/geometry')
def move_registration(candidate_id:int,data:RegistrationGeometryIn,db:Session=Depends(get_db)):
    row=db.get(RegistrationCandidate,candidate_id)
    if not row:raise HTTPException(404,'3D registration candidate not found')
    if row.status!='needs_adjustment':raise HTTPException(409,'Mark the 3D box as needs adjustment before moving it')
    row.x,row.y,row.z=data.x,data.y,data.z
    if data.yaw is not None:row.yaw=(data.yaw+math.pi)%(2*math.pi)-math.pi
    db.commit();db.refresh(row);return to_dict(row)
@app.post('/api/v1/models/{model_id}/registrations/review')
def review_registrations(model_id:int,data:RegistrationBulkReviewIn,db:Session=Depends(get_db)):
    if not db.get(Model3D,model_id):raise HTTPException(404,'3D model not found')
    ids=set(data.candidate_ids);rows=list(db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model_id,RegistrationCandidate.id.in_(ids))))
    if len(rows)!=len(ids):raise HTTPException(404,'One or more 3D registration candidates were not found in this model')
    status=_registration_status(data.status)
    for row in rows:row.status=status
    db.commit();all_rows=db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model_id).order_by(RegistrationCandidate.class_name,RegistrationCandidate.id)).all()
    return {'model_id':model_id,'updated_count':len(rows),'status':status,'candidates':[to_dict(row) for row in all_rows]}
@app.post('/api/v1/models/{model_id}/register-assets')
def register_assets(model_id:int):
    """Create conservative, review-only 3D candidates from high-confidence 2D boxes and COLMAP feature tracks."""
    return gpu.call('register_assets',model_id=model_id)
def _register_prepare(db,model_id):
    model=db.get(Model3D,model_id)
    if not model or model.status!='completed' or model.job_id is None:raise HTTPException(404,'Completed 3D model not found')
    inspection=db.get(Inspection,model.inspection_id)
    if not inspection:raise HTTPException(404,'Inspection not found')
    quality=model.quality or {};transform=quality.get('transform')
    if not transform:raise HTTPException(409,'This model has no viewer coordinate transform')
    media=db.scalars(select(Media).where(Media.inspection_id==inspection.id)).all()
    detections=db.scalars(select(Detection).where(Detection.inspection_id==inspection.id,Detection.review_status!='rejected')).all()
    reviewed=db.scalars(select(ImageReview.media_id).where(ImageReview.inspection_id==inspection.id,ImageReview.status=='approved',ImageReview.frame_index==0)).all()
    # a dense model made from a sparse one has no camera model of its own: use the source build's
    return {'work':quality.get('source_model_dir') or f'{inspection.id}_{model.job_id}','transform':transform,'reviewed':sorted(set(reviewed)),
            'dense':f'{inspection.id}_{model.job_id}/'+(quality.get('files') or {}).get('dense_ply','dense/fused.ply'),
            'media':[{k:getattr(m,k) for k in ('id','parent_media_id','content_type','filename','storage_key')} for m in media],
            'detections':[{k:getattr(d,k) for k in ('id','media_id','class_name','confidence','x1','y1','x2','y2','image_width','image_height')} for d in detections]}
def _register_apply(db,candidates,model_id):
    model=db.get(Model3D,model_id);inspection=db.get(Inspection,model.inspection_id)
    reconciliation=reconcile_registrations(db,model,inspection,candidates);db.commit()
    rows=db.scalars(select(RegistrationCandidate).where(RegistrationCandidate.model_id==model.id).order_by(RegistrationCandidate.class_name,RegistrationCandidate.id)).all()
    return {'model_id':model.id,'status':'needs_review','candidate_count':len(rows),'candidates':jsonable_encoder([to_dict(x) for x in rows]),
            'reconciliation':reconciliation,'warning':'Geometry-derived review candidates; approve verified boxes before promoting them to canonical assets.'}
gpu.register('register_assets',prepare=_register_prepare,apply=_register_apply,lane='quick')
@app.post('/api/v1/inspections/{inspection_id}/reconstruct',status_code=202)
def start_reconstruction(inspection_id:int,background:BackgroundTasks,dense:bool=False,media_ids:list[int]|None=Query(None),db:Session=Depends(get_db)):
    """Build a 3D model from the inspection's photos and videos (COLMAP). dense=true adds a detailed point cloud and mesh (much slower)."""
    if not db.get(Inspection,inspection_id):raise HTTPException(404,'Inspection not found')
    if (s:=colmap_state())!='ready':
        if s=='worker_offline':raise HTTPException(503,'The GPU worker is offline: start backend/run-worker.ps1 on the GPU machine')
        exe=reconstruction_info().get('colmap')
        raise HTTPException(503,f'COLMAP_PATH points to {exe}, which does not exist' if s=='missing' else 'COLMAP not found: unzip the COLMAP Windows CUDA build to D:/tools/colmap or set COLMAP_PATH in backend/.env, then restart the backend')
    q=select(Media).where(Media.inspection_id==inspection_id,Media.parent_media_id.is_(None))
    items=db.scalars(q if not media_ids else q.where(Media.id.in_(media_ids))).all()
    photos=[m for m in items if m.content_type.startswith('image/')];videos=[m for m in items if m.content_type.startswith('video/')]
    if len(photos)<3 and not videos:raise HTTPException(409,'Need at least 3 overlapping photos or a video')
    job=Job(inspection_id=inspection_id,kind='reconstruction',status='queued',stage='reconstruction',media_ids=[m.id for m in items],
            detail=f'Queued 3D reconstruction ({"dense" if dense else "sparse"}) · {len(photos)} photo(s), {len(videos)} video(s)')
    db.add(job);db.flush();m3=Model3D(inspection_id=inspection_id,job_id=job.id,dense=dense,status='queued',quality={});db.add(m3);db.commit()
    gpu.start(background,'run_reconstruction',job_id=job.id,model_id=m3.id,dense=dense);return to_dict(job)|{'model_id':m3.id}
@app.post('/api/v1/models/{model_id}/densify',status_code=202)
def densify_model(model_id:int,background:BackgroundTasks,db:Session=Depends(get_db)):
    """Dense point cloud + mesh from a finished sparse model, reusing its camera solution (no new feature matching)."""
    src=db.get(Model3D,model_id)
    if not src or src.job_id is None:raise HTTPException(404,'Model not found')
    if src.status!='completed' or src.dense:raise HTTPException(409,'Only a finished sparse model can be made dense')
    if (s:=colmap_state())!='ready':raise HTTPException(503,'The GPU worker is offline: start backend/run-worker.ps1 on the GPU machine' if s=='worker_offline' else f'COLMAP {s.replace("_"," ")}')
    old=db.get(Job,src.job_id)
    job=Job(inspection_id=src.inspection_id,kind='reconstruction',status='queued',stage='reconstruction',media_ids=old.media_ids if old else [],
            detail=f'Queued dense model from sparse model #{src.id}')
    db.add(job);db.flush();m3=Model3D(inspection_id=src.inspection_id,job_id=job.id,dense=True,status='queued',quality={'source_model':src.id});db.add(m3);db.commit()
    gpu.start(background,'run_reconstruction',job_id=job.id,model_id=m3.id,dense=True,source_model_id=src.id);return to_dict(job)|{'model_id':m3.id}
# 3D job: prepare (file list or source build) -> tasks.reconstruct_model on the GPU -> apply (model row)
def _recon_prepare(db,job_id,model_id,dense,source_model_id=None):
    job,m3=db.get(Job,job_id),db.get(Model3D,model_id);job.status=m3.status='running';db.commit()
    items=db.scalars(select(Media).where(Media.id.in_(job.media_ids or []))).all()
    order=lambda m:(m.filename.lower(),m.id)  # capture order (DJI_0001 ...) helps sequential matching
    src=db.get(Model3D,source_model_id) if source_model_id else None
    return {'work':f'{job.inspection_id}_{job.id}','dense':dense,
            'photos':[m.storage_key for m in sorted(items,key=order) if m.content_type.startswith('image/')] if not src else [],
            'videos':[m.storage_key for m in sorted(items,key=order) if m.content_type.startswith('video/')] if not src else [],
            'source':{'dir':f'{src.inspection_id}_{src.job_id}','quality':src.quality or {}} if src else None}
def _recon_apply(db,r,job_id,model_id,dense,source_model_id=None):
    job,m3=db.get(Job,job_id),db.get(Model3D,model_id)
    if source_model_id:r={**r,'source_model':source_model_id}
    m3.quality=r;m3.status='completed';m3.url=f'/recon/{job.inspection_id}_{job.id}/'+r['files']['web_points']
    job.status='completed';job.progress=100;job.stage='done'
    job.detail=(f"3D model{' (dense from model #'+str(source_model_id)+')' if source_model_id else ''}: {r['registered_images']}/{r['images_total']} images placed · {r['sparse_points']:,} sparse points"
                +(f" · {r.get('dense_points',0):,} dense points · {r.get('mesh_faces',0):,} mesh faces" if r.get('dense') else '')+f" · {r['seconds']}s")
    db.commit();return {'model_id':m3.id,'status':m3.status}
def _recon_fail(db,error,job_id,model_id,dense,source_model_id=None):
    db.rollback();job,m3=db.get(Job,job_id),db.get(Model3D,model_id);job.status=m3.status='failed'
    m3.quality={'error':str(error['detail'])[:500],'files':{'log':'colmap.log'},**({'source_model':source_model_id} if source_model_id else {})}
    job.detail=f"3D reconstruction failed: {error['detail']}"[:2000];db.commit()
gpu.register('run_reconstruction',prepare=_recon_prepare,apply=_recon_apply,fail=_recon_fail,progress=_job_progress)
def run_reconstruction(job_id:int,model_id:int,dense:bool,source_model_id:int|None=None):
    gpu.run_local('run_reconstruction',{'job_id':job_id,'model_id':model_id,'dense':dense,'source_model_id':source_model_id})
@app.get('/api/v1/assets')
def assets(site_id:int|None=None,db:Session=Depends(get_db)):
    q=select(Asset)
    if site_id:q=q.where(Asset.site_id==site_id)
    return [to_dict(x) for x in db.scalars(q.order_by(Asset.id)).all()]
@app.patch('/api/v1/assets/{asset_id}')
def update_asset(asset_id:int,data:AssetUpdate,db:Session=Depends(get_db)):
    x=db.get(Asset,asset_id)
    if not x:raise HTTPException(404,'Asset not found')
    for k,v in data.model_dump(exclude_unset=True).items():setattr(x,k,v)
    db.commit();db.refresh(x);return to_dict(x)
@app.get('/api/v1/defects')
def defects(db:Session=Depends(get_db)):return [to_dict(x) for x in db.scalars(select(Defect).order_by(Defect.id)).all()]
@app.post('/api/v1/defects/{defect_id}/review')
def review(defect_id:int,data:DefectDecision,db:Session=Depends(get_db)):
    if data.status not in {'approved','rejected','needs_evidence'}:raise HTTPException(422,'Invalid review status')
    x=db.get(Defect,defect_id)
    if not x:raise HTTPException(404,'Defect not found')
    x.review_status=data.status;x.notes=data.notes;db.commit();db.refresh(x);return to_dict(x)
@app.get('/api/v1/exports/assets.csv')
def assets_csv(db:Session=Depends(get_db)):
    output=io.StringIO();writer=csv.writer(output)
    writer.writerow(['asset_id','site_id','category','condition','identity_status','localization_status','x','y','z','description'])
    for a in db.scalars(select(Asset)).all():writer.writerow([a.code,a.site_id,a.category,a.condition,a.identity_status,a.localization_status,a.x,a.y,a.z,a.description])
    return StreamingResponse(iter([output.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="aeromind-assets.csv"'})

# ---------- the built web app (production: one service serves the API and the frontend) ----------
DIST=Path(os.getenv('FRONTEND_DIST',str(Path(__file__).resolve().parent.parent.parent/'frontend'/'dist'))).resolve()
if (DIST/'index.html').is_file():
    @app.get('/{path:path}',include_in_schema=False)
    def web_app(path:str):
        if path.startswith(('api/','uploads/','recon/','vendor-refs/')):raise HTTPException(404,'Not found')
        f=(DIST/path).resolve()
        if path and f.is_file() and DIST in f.parents:return FileResponse(f)
        return FileResponse(DIST/'index.html',headers={'Cache-Control':'no-cache'})  # client-side routes (/twin, /reconstruction ...)
