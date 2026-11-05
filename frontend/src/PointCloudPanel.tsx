import {useEffect,useRef,useState} from 'react';
import {AlertTriangle,Box,CheckCircle2,FileBox,Trash2,UploadCloud,X} from 'lucide-react';
import {request,uploadFile} from './api';
import './pointcloud.css';

/** Optional point cloud for an inspection (LiDAR / photogrammetry export). When present, "Build 3D model from point
 * cloud" uses it as the 3D model instead of COLMAP's dense reconstruction; photos (with GPS) are only aligned to it so
 * the 3D equipment boxes still work. */
export type PointCloudFile={id:number;inspection_id:number;filename:string;file_format:string;size_bytes:number;created_at:string;role:'cloud'|'sidecar'};
const ACCEPT='.las,.laz,.lax,.e57,.ply,.xyz,.pts,.txt,.csv,.prj';
export const mb=(n:number)=>n>=1e9?`${(n/1e9).toFixed(2)} GB`:n>=1e6?`${(n/1e6).toFixed(1)} MB`:`${Math.max(1,Math.round(n/1e3))} kB`;

export async function buildFromPointCloud(inspectionId:number){await request(`/api/v1/inspections/${inspectionId}/pointcloud-model`,{method:'POST'})}

export default function PointCloudPanel({inspectionId,inspectionLabel,onBuilt,onFilesChange,compact=false,embedded=false,hideBuild=false}:{inspectionId:number|null;inspectionLabel?:string;onBuilt?:()=>void;onFilesChange?:(files:PointCloudFile[])=>void;compact?:boolean;embedded?:boolean;hideBuild?:boolean}){
 const [photoCount,setPhotoCount]=useState(0);
 const [files,setFiles]=useState<PointCloudFile[]>([]),[busy,setBusy]=useState(''),[progress,setProgress]=useState<{name:string;loaded:number;total:number;n:number;of:number}|null>(null),[error,setError]=useState(''),[note,setNote]=useState('');
 const abort=useRef<AbortController|null>(null),changed=useRef(onFilesChange);changed.current=onFilesChange;
 const load=()=>{if(!inspectionId)return;request<PointCloudFile[]>(`/api/v1/inspections/${inspectionId}/pointclouds`).then(f=>{setFiles(f);changed.current?.(f)}).catch(e=>setError(String(e.message||e)));
  request<{content_type:string;parent_media_id:number|null}[]>(`/api/v1/inspections/${inspectionId}/media`).then(m=>setPhotoCount(m.filter(x=>x.parent_media_id==null&&x.content_type.startsWith('image/')).length)).catch(()=>{})};
 useEffect(()=>{setFiles([]);setError('');setNote('');load();return()=>abort.current?.abort()},[inspectionId]);
 async function choose(list:FileList|File[]|null){if(!list||!list.length||!inspectionId)return;const chosen=Array.from(list);setError('');setNote('');setBusy('upload');
  const ctl=new AbortController();abort.current=ctl;const done:PointCloudFile[]=[];
  try{for(const [i,f] of chosen.entries()){setProgress({name:f.name,loaded:0,total:f.size,n:i+1,of:chosen.length});
    done.push(await uploadFile<PointCloudFile>(`/api/v1/inspections/${inspectionId}/pointclouds`,f,(loaded,total)=>setProgress({name:f.name,loaded,total,n:i+1,of:chosen.length}),ctl.signal))}
   setNote(`${done.length} file${done.length===1?'':'s'} uploaded.${hideBuild?'':' Build the 3D model when all tiles are in.'}`)}
  catch(e){if((e as Error).name==='AbortError'){  // cancelled: remove what this batch already stored
     await Promise.all(done.map(f=>request(`/api/v1/pointclouds/${f.id}`,{method:'DELETE'}).catch(()=>{})));
     setNote(`Upload cancelled${done.length?` · ${done.length} uploaded file${done.length===1?'':'s'} removed`:''}.`)}
   else setError(String((e as Error).message||e))}
  finally{abort.current=null;setBusy('');setProgress(null);load()}}
 async function remove(f:PointCloudFile){setBusy('remove');try{await request(`/api/v1/pointclouds/${f.id}`,{method:'DELETE'});load()}catch(e){setError(String((e as Error).message||e))}finally{setBusy('')}}
 async function build(){if(!inspectionId)return;setBusy('build');setError('');
  try{await buildFromPointCloud(inspectionId);setNote('3D model build started: follow it on 3D Reconstruction.');onBuilt?.()}
  catch(e){setError(String((e as Error).message||e))}finally{setBusy('')}}
 const clouds=files.filter(f=>f.role==='cloud');
 if(!inspectionId)return <div className="pc-panel"><p className="muted">Create an inspection for this tower first.</p></div>;
 return <div className={'pc-panel'+(compact?' compact':'')+(embedded?' embedded':'')}>
  {!embedded&&<div className="pc-head"><FileBox size={18}/><div><b>Point cloud <em>optional</em></b><small>{inspectionLabel?`${inspectionLabel} · `:''}LAS, LAZ, E57, PLY, XYZ / PTS / TXT / CSV · sidecars .lax, .prj</small></div></div>}
  {!compact&&!embedded&&<p className="pc-explain">Have a LiDAR scan or a point cloud from other software? Upload it and the 3D model is built from it instead of the slow photo reconstruction.{photoCount>=3?` The ${photoCount} photos are still used, only to place them on the cloud (a few minutes), so the 3D equipment boxes work as usual.`:' Without photos you get the 3D view; equipment boxes need the photos.'} Clouds need a coordinate system (LAS header, E57 metadata or a .prj) to line up with the photos' GPS.</p>}
  {busy!=='upload'?<label className="pc-drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();choose(e.dataTransfer.files)}}><UploadCloud size={20}/><span><b>Choose or drop point cloud files</b><small>LAS, LAZ, E57, PLY, XYZ / PTS / TXT / CSV · several tiles are combined; large surveys are cropped to the area around the drone flight</small></span>
   <input type="file" multiple accept={ACCEPT} disabled={!!busy} onChange={e=>{choose(e.target.files);e.target.value=''}}/></label>
  :progress&&<div className="pc-progress"><div><span>Uploading {progress.n}/{progress.of} · {progress.name}</span><b>{mb(progress.loaded)} / {mb(progress.total)}</b></div><i><s style={{width:`${progress.total?progress.loaded/progress.total*100:0}%`}}/></i>
   <button type="button" className="btn outline pc-cancel" onClick={()=>abort.current?.abort()}><X size={14}/>Cancel upload</button></div>}
  {files.length>0&&<ul className="pc-files">{files.map(f=><li key={f.id}><span className={'pc-fmt '+f.role}>{f.file_format}</span><b>{f.filename}</b><small>{mb(f.size_bytes)}{f.role==='sidecar'?' · sidecar':''}</small>
   <button title="Remove" disabled={!!busy} onClick={()=>remove(f)}><Trash2 size={14}/></button></li>)}</ul>}
  {!hideBuild&&clouds.length>0&&<button className="btn primary pc-build" disabled={!!busy} onClick={build}><Box size={15}/>{busy==='build'?'Starting…':`Build 3D model from point cloud${clouds.length>1?` (${clouds.length} files)`:''}`}</button>}
  {note&&<div className="pc-note"><CheckCircle2 size={14}/>{note}</div>}
  {error&&<div className="pc-error"><AlertTriangle size={14}/>{error}</div>}
 </div>;
}
