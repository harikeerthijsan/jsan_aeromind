import {useEffect,useMemo,useRef,useState} from 'react';
import {AlertTriangle,ArrowRight,Box,CheckCircle2,Cpu,FileBox,Film,Image as ImageIcon,Plus,UploadCloud,X} from 'lucide-react';
import {Inspection,Media,Site,json,request,uploadChunked,uploadFile} from './api';
import PointCloudPanel,{PointCloudFile,buildFromPointCloud,mb} from './PointCloudPanel';
import './upload.css';

/** Media Upload: 1 photos & videos -> 2 point cloud (optional) -> 3 processing, for one chosen inspection.
 * Uploads can be cancelled: the file in transfer is aborted and the files of that batch already stored are removed. */
const TYPES:Record<string,string>={'image/jpeg':'JPEG','image/png':'PNG','image/tiff':'TIFF','video/mp4':'MP4','video/quicktime':'MOV'};
const MAX=500*1024*1024;
type Batch={files:File[];active:string[];sent:number;bytes:number;done:number[]};
const PARALLEL=4;  // one connection gets ~4.5 MB/s to the server (distance); four together ~13 MB/s, three times faster
type Msg={kind:'ok'|'warn'|'error';text:string;action?:{label:string;page:string}}|null;

function Step({n,title,hint,done,children}:{n:number;title:string;hint:string;done?:boolean;children:React.ReactNode}){
 return <section className={'up-step'+(done?' done':'')}><div className="up-step-head"><span className="up-step-no">{done?<CheckCircle2 size={16}/>:n}</span><div><h3>{title}</h3><small>{hint}</small></div></div><div className="up-step-body">{children}</div></section>}

export default function UploadPage({site,inspections,go,onChanged}:{site?:Site;inspections:Inspection[];go:(page:string)=>void;onChanged:()=>unknown}){
 const mine=useMemo(()=>inspections.filter(i=>i.site_id===site?.id).sort((a,b)=>b.id-a.id),[inspections,site?.id]);
 const [inspectionId,setInspectionId]=useState<number|null>(null),[newLabel,setNewLabel]=useState<string|null>(null);
 const [media,setMedia]=useState<Media[]>([]),[clouds,setClouds]=useState<PointCloudFile[]>([]);
 const [batch,setBatch]=useState<Batch|null>(null),[msg,setMsg]=useState<Msg>(null),[busy,setBusy]=useState(''),[drag,setDrag]=useState(false);
 const [auto,setAuto]=useState(()=>{try{return localStorage.getItem('aeromind.upload.auto')!=='false'}catch{return true}});
 const abort=useRef<AbortController|null>(null);
 const inspection=mine.find(i=>i.id===inspectionId)||mine[0];
 useEffect(()=>{setInspectionId(mine[0]?.id??null);setMsg(null)},[site?.id,mine.length]);
 const loadMedia=()=>{if(inspection)request<Media[]>(`/api/v1/inspections/${inspection.id}/media`).then(setMedia).catch(()=>setMedia([]));else setMedia([])};
 useEffect(()=>{loadMedia()},[inspection?.id]);
 useEffect(()=>()=>abort.current?.abort(),[]);
 const top=media.filter(m=>m.parent_media_id==null),photos=top.filter(m=>m.content_type.startsWith('image/')).length,videos=top.filter(m=>m.content_type.startsWith('video/')).length;
 const cloudFiles=clouds.filter(c=>c.role==='cloud');

 async function createInspection(){if(!site||!newLabel||newLabel.trim().length<2)return;setBusy('inspection');
  try{const x=await request<Inspection>('/api/v1/inspections',json('POST',{site_id:site.id,label:newLabel.trim()}));setNewLabel(null);await onChanged();setInspectionId(x.id)}
  catch(e){setMsg({kind:'error',text:String((e as Error).message||e)})}finally{setBusy('')}}

 async function upload(list:FileList|File[]){
  if(!inspection){setMsg({kind:'warn',text:'Create an inspection for this tower first.'});return}
  const all=Array.from(list),bad=all.filter(f=>!TYPES[f.type]||f.size>MAX),files=all.filter(f=>TYPES[f.type]&&f.size<=MAX);
  if(!files.length){setMsg({kind:'warn',text:`Nothing to upload: ${bad.map(f=>f.name).slice(0,3).join(', ')} ${bad.length>1?'are':'is'} not a supported photo/video or over 500 MB.`});return}
  const ctl=new AbortController();abort.current=ctl;const done:number[]=[];setMsg(null);
  const bytes=files.reduce((a,f)=>a+f.size,0),sent=new Map<File,number>(),active=new Set<File>();
  const show=()=>setBatch({files,active:[...active].map(f=>f.name),sent:[...sent.values()].reduce((a,b)=>a+b,0),bytes,done:[...done]});show();
  const send=async(f:File)=>{const onProgress=(loaded:number)=>{sent.set(f,loaded);show()};
   for(let attempt=1;;attempt++){
    try{return f.size>32*1024*1024?await uploadChunked<Media>(inspection.id,'media',f,onProgress,ctl.signal)  // videos in parts: one request must finish within the proxy's 5 minutes
     :await uploadFile<Media>(`/api/v1/inspections/${inspection.id}/media`,f,onProgress,ctl.signal)}
    catch(e){const st=(e as {status?:number}).status??0;sent.set(f,0);if((e as Error).name==='AbortError'||attempt>=2||(st>=400&&st<500))throw e;await new Promise(r=>setTimeout(r,2000))}}};  // one retry after a network drop
  let next=0,failed:unknown=null;
  const worker=async()=>{while(next<files.length&&!failed&&!ctl.signal.aborted){const f=files[next++];active.add(f);show();
   try{const x=await send(f);done.push(x.id);sent.set(f,f.size)}catch(e){failed=failed||e}finally{active.delete(f);show()}}};
  try{
   await Promise.all(Array.from({length:Math.min(PARALLEL,files.length)},worker));
   if(failed)throw failed;
   const skipped=bad.length?` ${bad.length} file(s) skipped (unsupported or over 500 MB).`:'';
   if(auto){
    try{await request(`/api/v1/inspections/${inspection.id}/process?${done.map(i=>'media_ids='+i).join('&')}`,{method:'POST'});
     setMsg({kind:'ok',text:`${done.length} file(s) uploaded. Detection started for them.${skipped}`,action:{label:'Follow processing',page:'processing'}})}
    catch(e){setMsg({kind:'warn',text:`${done.length} file(s) uploaded, but detection could not start: ${(e as Error).message}${skipped}`})}}
   else setMsg({kind:'ok',text:`${done.length} file(s) uploaded.${skipped} Run detection in step 3 when you are ready.`});
   onChanged()}
  catch(e){
   if((e as Error).name==='AbortError'){  // cancelled: remove the files of this batch that were already stored
    setBusy('cancelling');const left=await Promise.all(done.map(id=>request(`/api/v1/media/${id}`,{method:'DELETE'}).then(()=>0,()=>1)));
    const kept=left.reduce((a,b)=>a+b,0);setMsg({kind:'warn',text:`Upload cancelled.${done.length?` ${done.length-kept} uploaded file(s) removed.`:''}${kept?` ${kept} could not be removed.`:''}`})}
   else setMsg({kind:'error',text:`Upload stopped: ${(e as Error).message}. Files uploaded before the error were kept.`})}
  finally{abort.current=null;setBatch(null);setBusy('');loadMedia()}}

 async function runDetection(){if(!inspection)return;setBusy('detect');
  try{await request(`/api/v1/inspections/${inspection.id}/process?only_new=true`,{method:'POST'});setMsg({kind:'ok',text:'Detection started for the files that have not been processed yet.',action:{label:'Follow processing',page:'processing'}});onChanged()}
  catch(e){setMsg({kind:'warn',text:String((e as Error).message||e)})}finally{setBusy('')}}
 async function buildCloud(){if(!inspection)return;setBusy('cloud');
  try{await buildFromPointCloud(inspection.id);setMsg({kind:'ok',text:'3D model build from the point cloud started.',action:{label:'Open 3D Reconstruction',page:'reconstruction'}});onChanged()}
  catch(e){setMsg({kind:'warn',text:String((e as Error).message||e)})}finally{setBusy('')}}

 const uploading=!!batch,pct=batch&&batch.bytes?Math.round(batch.sent/batch.bytes*100):0;
 return <div className="up-grid">
  <div className="up-main">
   <section className="panel up-target">
    <div><small>Tower</small><b>{site?`${site.code} · ${site.name}`:'Select a tower above'}</b></div>
    <div className="up-target-ins"><small>Inspection</small>
     {newLabel===null?<div className="up-row"><select value={inspection?.id??''} onChange={e=>setInspectionId(+e.target.value)} disabled={uploading||!mine.length}>
       {mine.length?mine.map(i=><option key={i.id} value={i.id}>INS-{String(i.id).padStart(4,'0')} · {i.label}</option>):<option value="">No inspections yet</option>}</select>
      <button className="btn outline" disabled={uploading||!site} onClick={()=>setNewLabel('Drone survey '+new Date().toLocaleDateString())}><Plus size={14}/>New inspection</button></div>
     :<form className="up-row" onSubmit={e=>{e.preventDefault();createInspection()}}><input autoFocus value={newLabel} onChange={e=>setNewLabel(e.target.value)} placeholder="Inspection label"/>
       <button className="btn primary" disabled={busy==='inspection'||newLabel.trim().length<2}>Create</button><button type="button" className="btn outline" onClick={()=>setNewLabel(null)}>Cancel</button></form>}</div>
    <div className="up-counts"><span><ImageIcon size={14}/>{photos} photos</span><span><Film size={14}/>{videos} videos</span><span><FileBox size={14}/>{cloudFiles.length} point cloud{cloudFiles.length===1?'':'s'}</span></div>
   </section>

   <Step n={1} title="Photos & videos" hint="Drone photos and videos of the tower · JPEG, PNG, TIFF, MP4, MOV · up to 500 MB each" done={photos+videos>0&&!uploading}>
    {!uploading?<label className={'up-drop'+(drag?' drag':'')+(!inspection?' disabled':'')} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);upload(e.dataTransfer.files)}}>
      <UploadCloud size={34}/><b>Drop photos and videos here, or click to choose</b><span>Several files at once · videos are split into still frames (every 30th) for review</span>
      <input type="file" multiple accept={Object.keys(TYPES).join(',')} disabled={!inspection} onChange={e=>{if(e.target.files)upload(e.target.files);e.target.value=''}}/></label>
    :<div className="up-progress">
      <div className="up-progress-head"><b>{busy==='cancelling'?'Cancelling…':`Uploading ${batch.files.length} file${batch.files.length===1?'':'s'} · ${Math.min(PARALLEL,batch.files.length)} at a time`}</b><strong>{pct}%</strong></div>
      <div className="up-bar"><i style={{width:pct+'%'}}/></div>
      <div className="up-progress-file"><span>{batch.active.slice(0,2).join(', ')}{batch.active.length>2?` +${batch.active.length-2}`:''}</span><small>{mb(batch.sent)} / {mb(batch.bytes)}</small></div>
      <div className="up-progress-foot"><small>{batch.done.length} uploaded · {batch.files.length-batch.done.length} to go</small>
       <button className="btn outline up-cancel" disabled={busy==='cancelling'} onClick={()=>abort.current?.abort()}><X size={14}/>Cancel upload</button></div>
      <small className="muted">Cancelling stops the transfer and removes the files of this batch that were already uploaded.</small></div>}
    <label className="up-check"><input type="checkbox" checked={auto} disabled={uploading} onChange={e=>{setAuto(e.target.checked);try{localStorage.setItem('aeromind.upload.auto',String(e.target.checked))}catch{/* private mode */}}}/>
     <span><b>Run detection automatically after the upload</b><small>Only the newly uploaded files are processed.</small></span></label>
   </Step>

   <Step n={2} title="Point cloud (optional)" hint="A LiDAR scan or a point cloud from other software replaces the slow photo reconstruction" done={cloudFiles.length>0}>
    <PointCloudPanel embedded hideBuild inspectionId={inspection?.id??null} inspectionLabel={inspection?.label} onFilesChange={setClouds}/>
   </Step>

   <Step n={3} title="Process" hint="Detect equipment in the photos, then build the 3D model">
    <div className="up-actions">
     <div className="up-action"><Cpu size={20}/><div><b>AI detection</b><small>Runs the YOLO model on photos and video frames not processed yet.</small></div>
      <button className="btn primary" disabled={!inspection||!top.length||uploading||busy==='detect'} onClick={runDetection}>{busy==='detect'?'Starting…':'Run detection'}</button></div>
     <div className="up-action"><Box size={20}/><div><b>3D model</b><small>{cloudFiles.length?`From the uploaded point cloud${photos>=3?`; the ${photos} photos are aligned to it for the 3D boxes`:''}.`:'From the photos (COLMAP): sparse in minutes, dense + mesh in up to an hour.'}</small></div>
      {cloudFiles.length?<button className="btn primary" disabled={uploading||busy==='cloud'} onClick={buildCloud}>{busy==='cloud'?'Starting…':'Build from point cloud'}</button>
       :<button className="btn outline" disabled={!inspection||photos+videos<1} onClick={()=>go('reconstruction')}>Build from photos <ArrowRight size={14}/></button>}</div>
    </div>
   </Step>
   {msg&&<div className={'up-msg '+msg.kind}>{msg.kind==='ok'?<CheckCircle2 size={16}/>:<AlertTriangle size={16}/>}<span>{msg.text}</span>{msg.action&&<button className="text-btn" onClick={()=>go(msg.action!.page)}>{msg.action.label} <ArrowRight size={13}/></button>}<button className="up-msg-x" onClick={()=>setMsg(null)} aria-label="Dismiss"><X size={14}/></button></div>}
  </div>
  <aside className="panel up-side"><div className="panel-head"><h2>Capture checklist</h2></div>
   {['Full-tower establishing shots','Upper / middle / lower orbital passes','Close-up antenna and mount photos','Camera metadata and timestamps','GPS and calibrated poses, if available','Coverage for occluded/hidden sides'].map(x=><div className="checklist" key={x}><CheckCircle2 size={17}/>{x}</div>)}
   <div className="notice">Files are stored in the inspection's folder in cloud storage. Detection and 3D builds run on the GPU worker.</div></aside>
 </div>;
}
