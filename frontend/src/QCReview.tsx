import {Fragment,useRef,useState} from 'react';import {useSearchParams} from 'react-router-dom';import {ClipboardCheck,Image as ImageIcon,CheckCircle2,AlertTriangle,Download,Film,ChevronDown,ChevronRight} from 'lucide-react';
import {API,Inspection,Job,Media} from './api';import AnnotationEditor,{isVideo,useReviewData,useDirtyGuard,orderMedia,frameLabel} from './AnnotationEditor';
type Filter='todo'|'done'|'all';
/** Manual QC: go image by image, correct or draw boxes by hand and approve each image's final annotation set. */
export default function QCReview({inspections,jobs}:{inspections:Inspection[];jobs:Job[]}){
 const [params]=useSearchParams(),num=(k:string)=>Number(params.get(k))||null;// deep link from the inventory pages
 const [inspectionId,setInspectionId]=useState<number|null>(num('inspection')),[mediaId,setMediaId]=useState<number|null>(num('media')),[filter,setFilter]=useState<Filter>(num('media')?'all':'todo'),[minConf,setMinConf]=useState(0),[hidden,setHidden]=useState<Set<string>>(new Set());
 const shell=useRef<HTMLDivElement>(null);const {onDirty,guard}=useDirtyGuard();
 const inspection=inspections.find(i=>i.id===inspectionId)||inspections[0];const insJobs=jobs.filter(j=>j.inspection_id===inspection?.id),job=insJobs.find(j=>['queued','running'].includes(j.status))||insJobs[0];// a running job first, else the newest
 
 const {media,detections,reviews,classes,vendors,error,saved,loading}=useReviewData(inspection?.id,insJobs.map(j=>`${j.id}:${j.status}`).join());
 const [openVid,setOpenVid]=useState<Set<number>>(new Set()),toggleVid=(id:number)=>setOpenVid(s=>{const n=new Set(s);n.has(id)?n.delete(id):n.add(id);return n});
 const images=orderMedia(media).filter(m=>!isVideo(m)),reviewOf=(id:number)=>reviews.find(r=>r.media_id===id&&r.frame_index===0),done=(m:Media)=>reviewOf(m.id)?.status==='approved';
 const approved=images.filter(done).length,shown=images.filter(m=>filter==='all'||(filter==='done')===done(m));
 // look the current image up in the full list so it stays open after approving it under "To review"
 const current=images.find(m=>m.id===mediaId)||shown[0]||images[0],pos=current?images.indexOf(current):-1;
 const next=shown.find(m=>images.indexOf(m)>pos),prev=[...shown].reverse().find(m=>images.indexOf(m)<pos),jobId=detections.find(d=>d.media_id===current?.id)?.job_id??null;
 const open=(id:number)=>guard(()=>setMediaId(id));
 const toggle=(name:string)=>setHidden(h=>{const n=new Set(h);n.has(name)?n.delete(name):n.add(name);return n});
 if(!inspections.length)return <section className="panel"><div className="placeholder-vision"><ClipboardCheck size={54}/><h2>No inspections for this tower</h2><p>Create an inspection and upload survey imagery to start quality control.</p></div></section>;
 return <div className="viewer-shell det-shell" ref={shell}><div className="viewer-left"><h3>Inspection</h3><select className="det-select" aria-label="Inspection" value={inspection?.id} onChange={e=>{const id=+e.target.value;guard(()=>{setInspectionId(id);setMediaId(null)})}}>{inspections.map(i=><option key={i.id} value={i.id}>INS-{String(i.id).padStart(4,'0')} · {i.label}</option>)}</select>
  <div className="qc-progress"><span>Approved <b>{approved}</b> of {images.length} images</span><div className="qc-bar"><div style={{width:`${approved/Math.max(1,images.length)*100}%`}}/></div></div>
  <div className="qc-tabs" role="tablist">{([['todo','To review'],['done','Approved'],['all','All']] as [Filter,string][]).map(([f,l])=><button key={f} role="tab" aria-selected={filter===f} className={filter===f?'on':''} onClick={()=>setFilter(f)}>{l} ({f==='all'?images.length:f==='done'?approved:images.length-approved})</button>)}</div>
  {error&&<div className="det-job"><small className="det-error">{error}</small></div>}
  <h4>Images</h4>{shown.map((m,i)=>{const r=reviewOf(m.id),n=detections.filter(d=>d.media_id===m.id).length,pid=m.parent_media_id;
   const grp=pid==null?[]:shown.filter(x=>x.parent_media_id===pid),head=pid!=null&&shown[i-1]?.parent_media_id!==pid,vis=pid==null||openVid.has(pid)||grp.some(x=>x.id===current?.id);
   return <Fragment key={m.id}>{head&&<button className="tree-asset video-head" aria-expanded={vis} onClick={()=>toggleVid(pid!)}>{vis?<ChevronDown size={15}/>:<ChevronRight size={15}/>}<Film size={15}/><div><b>{media.find(v=>v.id===pid)?.filename||'Video'}</b><small>{images.filter(x=>x.parent_media_id===pid).length} frames · {images.filter(x=>x.parent_media_id===pid&&done(x)).length} approved</small></div></button>}
    {vis&&<button className={'tree-asset '+(pid!=null?'frame ':'')+(current?.id===m.id?'chosen':'')} onClick={()=>open(m.id)}>{r?<CheckCircle2 size={15} className="ok"/>:<ImageIcon size={15}/>}<div><b>{pid!=null?frameLabel(m):m.filename}</b><small>{r?`Approved · ${r.box_count} boxes`:n?`${n} AI boxes to check`:'No AI boxes · label manually'}</small></div></button>}</Fragment>})}
  {!loading&&!shown.length&&<p className="muted">{!images.length?'No images uploaded for this inspection.':filter==='todo'?'Every image in this inspection is approved.':'No approved images yet.'}</p>}
  {approved>0&&<a className="text-btn det-zip" href={`${API}/api/v1/inspections/${inspection!.id}/annotations.csv`} title="One row per approved box, with manufacturer tags"><Download size={13}/>Download annotations (.csv)</a>}
  {approved>0&&<a className="text-btn det-zip" href={`${API}/api/v1/inspections/${inspection!.id}/annotated.zip?source=approved`} title="Approved images with their QC'd boxes drawn, original metadata kept"><Download size={13}/>Download approved images (.zip)</a>}
  {media.length>images.length&&<p className="muted">{media.length-images.length} video file(s): their extracted frames are listed above once detection has run.</p>}</div>
  {current?<AnnotationEditor focusKey={num('item')&&current.id===num('media')?'a'+num('item'):undefined} vendors={vendors} key={current.id} media={current} workflow="manual_qc" detections={detections.filter(d=>d.media_id===current.id)} jobId={jobId} classes={classes} minConf={minConf} onMinConf={setMinConf} hidden={hidden} onToggleClass={toggle} shell={shell} onSaved={saved} onDirty={onDirty}
    nav={{position:`${pos+1} / ${images.length}`,prev:prev&&(()=>open(prev.id)),next:next&&(()=>open(next.id))}}
    notice={<div className="notice"><AlertTriangle size={17}/>Approving saves exactly the boxes shown as this image's QC'd annotations. AI boxes you delete are recorded as rejected detections.</div>}/>
  :<><div className="viewer-middle"><div className="viewer-toolbar"><span>Manual QC</span></div><div className="det-stage"><div className="det-empty"><ClipboardCheck size={46}/><p>{loading?'Loading…':'Upload images to this inspection to start QC.'}</p></div></div></div><div className="viewer-right"><div className="panel-head"><h2>Annotations</h2></div><p className="muted">Pick an image to review.</p></div></>}</div>
}
