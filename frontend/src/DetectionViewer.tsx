import {Fragment,useEffect,useRef,useState} from 'react';import {ScanEye,Play,Image as ImageIcon,Film,AlertTriangle,CheckCircle2,Download} from 'lucide-react';
import {API,request,Inspection,Job,DetectorInfo,Media} from './api';import AnnotationEditor,{colorOf,isVideo,useReviewData,useDirtyGuard,orderMedia,frameLabel} from './AnnotationEditor';
const running=(j?:Job)=>!!j&&['queued','running'].includes(j.status);
export default function DetectionViewer({inspections,jobs,onChanged}:{inspections:Inspection[];jobs:Job[];onChanged:()=>void}){
 const [inspectionId,setInspectionId]=useState<number|null>(null),[detector,setDetector]=useState<DetectorInfo|null>(null),[mediaId,setMediaId]=useState<number|null>(null),[minConf,setMinConf]=useState(0.25),[hidden,setHidden]=useState<Set<string>>(new Set());
 const shell=useRef<HTMLDivElement>(null);const {onDirty,guard}=useDirtyGuard();
 const inspection=inspections.find(i=>i.id===inspectionId)||inspections[0];const insJobs=jobs.filter(j=>j.inspection_id===inspection?.id),job=insJobs.find(j=>['queued','running'].includes(j.status))||insJobs[0];// a running job first, else the newest
 // jobs arrive newest-first
 const {media,detections,reviews,classes,vendors,error,setError,saved}=useReviewData(inspection?.id,insJobs.map(j=>`${j.id}:${j.status}`).join());
 useEffect(()=>{request<DetectorInfo>('/api/v1/detector').then(d=>{setDetector(d);setMinConf(d.conf)}).catch(()=>setDetector(null))},[]);
 const ordered=orderMedia(media),top=ordered.filter(m=>m.parent_media_id==null),framesOf=(id:number)=>ordered.filter(k=>k.parent_media_id===id);
 const current=ordered.find(m=>m.id===mediaId)||ordered[0];
 const forMedia=detections.filter(d=>d.media_id===current?.id),passing=forMedia.filter(d=>d.confidence>=minConf),visible=passing.filter(d=>!hidden.has(d.class_name)),jobId=forMedia[0]?.job_id??null;// each file shows its own latest run
 const counts=new Map<string,{id:number;n:number}>();passing.forEach(d=>{const c=counts.get(d.class_name)||{id:d.class_id,n:0};c.n++;counts.set(d.class_name,c)});
 const toggle=(name:string)=>setHidden(h=>{const n=new Set(h);n.has(name)?n.delete(name):n.add(name);return n});
 const reviewOf=(id:number)=>reviews.find(r=>r.media_id===id&&r.frame_index===0);
 const idx=current?ordered.indexOf(current):-1,open=(id:number)=>guard(()=>setMediaId(id));
 async function run(){if(!inspection)return;try{await request(`/api/v1/inspections/${inspection.id}/process`,{method:'POST'});onChanged()}catch(e){setError(String(e))}}
 const frames=new Map<number,number>();if(current&&isVideo(current))visible.forEach(d=>frames.set(d.frame_index,(frames.get(d.frame_index)||0)+1));
 const hint=detector?.status!=='ready'?`Detector ${detector?.status.replaceAll('_',' ')||'unavailable'}. Set YOLO_MODEL_PATH on the backend.`:!media.length?'Upload media to this inspection first.':running(job)?'Detection in progress…':!detections.length?'Run detection to see observations.':'No detections above this confidence.';
 const notice=<div className="notice"><AlertTriangle size={17}/>Detections are model observations{detector?.model?` from ${detector.model}`:''}, not verified assets. Fix wrong boxes or classes, then approve to save the reviewed set for this image.</div>;
 if(!inspections.length)return <section className="panel"><div className="placeholder-vision"><ScanEye size={54}/><h2>No inspections for this tower</h2><p>Create an inspection and upload survey imagery, then run detection.</p></div></section>;
 return <div className="viewer-shell det-shell" ref={shell}><div className="viewer-left"><h3>Inspection</h3><select className="det-select" aria-label="Inspection" value={inspection?.id} onChange={e=>{const id=+e.target.value;guard(()=>{setInspectionId(id);setMediaId(null)})}}>{inspections.map(i=><option key={i.id} value={i.id}>INS-{String(i.id).padStart(4,'0')} · {i.label}</option>)}</select>
  <button className="btn primary wide" onClick={()=>guard(run)} disabled={insJobs.some(running)||!media.length||detector?.status!=='ready'}><Play size={15}/>{running(job)?`Running… ${job!.progress}%`:detections.length?'Re-run detection':'Run detection'}</button>
  {job&&<div className="det-job"><span className={'badge '+(job.status==='completed'?'good':job.status==='failed'?'poor':'neutral')}>{job.status.replaceAll('_',' ')}</span><small>Job #{job.id} · {job.detail}</small></div>}
  {detector&&<div className="det-job"><span className={'badge '+(detector.status==='ready'?'good':'unknown')}>detector {detector.status.replaceAll('_',' ')}</span></div>}
  {error&&<div className="det-job"><small className="det-error">{error}</small></div>}
  <h4>Media ({top.length}{ordered.length>top.length?` · ${ordered.length-top.length} video frames`:''})</h4>{top.map(m=>{const kids=framesOf(m.id),showKids=kids.length>0&&(current?.id===m.id||kids.some(k=>k.id===current?.id));
   const row=(x:Media,frame:boolean)=>{const r=reviewOf(x.id),n=detections.filter(d=>d.media_id===x.id).length;return <button key={x.id} className={'tree-asset '+(frame?'frame ':'')+(current?.id===x.id?'chosen':'')} onClick={()=>open(x.id)}>{r?<CheckCircle2 size={15} className="ok"/>:isVideo(x)?<Film size={15}/>:<ImageIcon size={15}/>}<div><b>{frame?frameLabel(x):x.filename}</b><small>{isVideo(x)&&kids.length?`${kids.length} frames · ${kids.reduce((t,k)=>t+detections.filter(d=>d.media_id===k.id).length,0)} detections · ${kids.filter(k=>reviewOf(k.id)).length} approved`:`${n} detections${r?` · approved ${r.box_count}`:''}`}</small></div></button>};
   return <Fragment key={m.id}>{row(m,false)}{showKids&&kids.map(k=>row(k,true))}</Fragment>})}{!media.length&&<p className="muted">No media uploaded for this inspection.</p>}
  {media.some(m=>!isVideo(m))&&<a className="text-btn det-zip" href={`${API}/api/v1/inspections/${inspection!.id}/annotated.zip`} title="Each image with its approved boxes (or AI boxes if not reviewed yet), original metadata kept"><Download size={13}/>Download annotated images (.zip)</a>}</div>
  {current&&!isVideo(current)?<AnnotationEditor vendors={vendors} key={current.id} media={current} workflow="ai_detection" detections={forMedia} jobId={jobId} classes={classes} minConf={minConf} onMinConf={setMinConf} hidden={hidden} onToggleClass={toggle} shell={shell} onSaved={saved} onDirty={onDirty} notice={notice}
    nav={{position:`${idx+1} / ${ordered.length}`,prev:idx>0?()=>open(ordered[idx-1].id):undefined,next:idx<ordered.length-1?()=>open(ordered[idx+1].id):undefined}}/>
  :<><div className="viewer-middle"><div className="viewer-toolbar"><span><span className="live-dot"/>{current?.filename||'No media'}</span><span>{visible.length} of {forMedia.length} boxes shown</span></div><div className="det-stage">
   {!current?<div className="det-empty"><ScanEye size={46}/><p>Upload imagery to this inspection, then run detection.</p></div>:<video className="det-video" src={API+current.url} controls/>}</div></div>
  <div className="viewer-right"><div className="panel-head"><h2>Detections</h2></div>
   <label className="det-conf"><span>Min confidence <b>{minConf.toFixed(2)}</b></span><input type="range" min={0} max={1} step={0.05} value={minConf} onChange={e=>setMinConf(+e.target.value)}/></label>
   {[...counts].sort((a,b)=>b[1].n-a[1].n).map(([name,{id,n}])=><button key={name} title={hidden.has(name)?'Show class':'Hide class'} className={'det-class '+(hidden.has(name)?'off':'')} onClick={()=>toggle(name)}><i style={{background:colorOf(id)}}/><span>{name}</span><b>{n}</b></button>)}
   {!counts.size&&!(current&&framesOf(current.id).length)&&<p className="muted">{hint}</p>}
   {current&&isVideo(current)&&framesOf(current.id).length>0?<><h4 className="det-sub">Extracted frames ({framesOf(current.id).length})</h4>{framesOf(current.id).map(k=>{const r=reviewOf(k.id);return <button key={k.id} className="det-class" onClick={()=>open(k.id)}>{r?<CheckCircle2 size={13} className="ok"/>:<ImageIcon size={13}/>}<span>{frameLabel(k)}</span><b>{detections.filter(d=>d.media_id===k.id).length}</b></button>})}</>
   :frames.size>0&&<><h4 className="det-sub">Sampled frames</h4>{[...frames].sort((a,b)=>a[0]-b[0]).slice(0,60).map(([f,n])=><div className="info" key={f}><small>Frame {f}</small><span>{n} boxes</span></div>)}</>}
   {current&&<p className="muted">{framesOf(current.id).length?'Each extracted frame is reviewed like a photo: open one to edit, assess and approve it.':'Run detection to split this video into still frames (every 30th frame by default) for review and approval.'}</p>}
   {notice}</div></>}</div>
}
