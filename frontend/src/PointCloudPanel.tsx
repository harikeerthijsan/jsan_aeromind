import {useEffect,useState} from 'react';
import {AlertTriangle,Box,CheckCircle2,FileBox,Trash2,UploadCloud} from 'lucide-react';
import {API,getAuthToken,request} from './api';
import './pointcloud.css';

/** Optional point cloud for an inspection (LiDAR / photogrammetry export). When present, "Build 3D model from point
 * cloud" uses it as the 3D model instead of COLMAP's dense reconstruction; photos (with GPS) are only aligned to it so
 * the 3D equipment boxes still work. */
export type PointCloudFile={id:number;inspection_id:number;filename:string;file_format:string;size_bytes:number;created_at:string;role:'cloud'|'sidecar'};
const ACCEPT='.las,.laz,.lax,.ply,.xyz,.pts,.txt,.csv,.prj';
const mb=(n:number)=>n>=1e9?`${(n/1e9).toFixed(2)} GB`:`${(n/1e6).toFixed(1)} MB`;

/** XHR rather than fetch: uploads of several GB need a progress bar. */
function upload(inspectionId:number,file:File,onProgress:(loaded:number)=>void):Promise<PointCloudFile>{
 return new Promise((resolve,reject)=>{const x=new XMLHttpRequest();x.open('POST',`${API}/api/v1/inspections/${inspectionId}/pointclouds`);x.withCredentials=true;
  const t=getAuthToken();if(t)x.setRequestHeader('Authorization',`Bearer ${t}`);
  x.upload.onprogress=e=>onProgress(e.loaded);x.onerror=()=>reject(Error('Network error while uploading '+file.name));
  x.onload=()=>{if(x.status>=200&&x.status<300)resolve(JSON.parse(x.responseText));else{let d=x.responseText;try{d=JSON.parse(d).detail}catch{/* plain text */}reject(Error(`${file.name}: ${d||x.status}`))}};
  const body=new FormData();body.append('file',file);x.send(body)})}

export default function PointCloudPanel({inspectionId,inspectionLabel,onBuilt,compact=false}:{inspectionId:number|null;inspectionLabel?:string;onBuilt?:()=>void;compact?:boolean}){
 const [photoCount,setPhotoCount]=useState(0);
 const [files,setFiles]=useState<PointCloudFile[]>([]),[busy,setBusy]=useState(''),[progress,setProgress]=useState<{name:string;loaded:number;total:number}|null>(null),[error,setError]=useState(''),[note,setNote]=useState('');
 const load=()=>{if(!inspectionId)return;request<PointCloudFile[]>(`/api/v1/inspections/${inspectionId}/pointclouds`).then(setFiles).catch(e=>setError(String(e.message||e)));
  request<{content_type:string;parent_media_id:number|null}[]>(`/api/v1/inspections/${inspectionId}/media`).then(m=>setPhotoCount(m.filter(x=>x.parent_media_id==null&&x.content_type.startsWith('image/')).length)).catch(()=>{})};
 useEffect(()=>{setFiles([]);setError('');setNote('');load()},[inspectionId]);
 async function choose(list:FileList|null){if(!list?.length||!inspectionId)return;setError('');setNote('');setBusy('upload');
  try{for(const f of Array.from(list)){setProgress({name:f.name,loaded:0,total:f.size});await upload(inspectionId,f,loaded=>setProgress({name:f.name,loaded,total:f.size}))}setNote('Uploaded. Build the 3D model when all tiles are in.')}
  catch(e){setError(String((e as Error).message||e))}finally{setBusy('');setProgress(null);load()}}
 async function remove(f:PointCloudFile){setBusy('remove');try{await request(`/api/v1/pointclouds/${f.id}`,{method:'DELETE'});load()}catch(e){setError(String((e as Error).message||e))}finally{setBusy('')}}
 async function build(){if(!inspectionId)return;setBusy('build');setError('');
  try{await request(`/api/v1/inspections/${inspectionId}/pointcloud-model`,{method:'POST'});setNote('3D model build started: follow it on 3D Reconstruction.');onBuilt?.()}
  catch(e){setError(String((e as Error).message||e))}finally{setBusy('')}}
 const clouds=files.filter(f=>f.role==='cloud');
 if(!inspectionId)return <div className="pc-panel"><p className="muted">Create an inspection for this tower first.</p></div>;
 return <div className={'pc-panel'+(compact?' compact':'')}>
  <div className="pc-head"><FileBox size={18}/><div><b>Point cloud <em>optional</em></b><small>{inspectionLabel?`${inspectionLabel} · `:''}LAS, LAZ, PLY, XYZ / PTS / TXT / CSV · sidecars .lax, .prj</small></div></div>
  {!compact&&<p className="pc-explain">Have a LiDAR scan or a point cloud from other software? Upload it and the 3D model is built from it instead of the slow photo reconstruction.{photoCount>=3?` The ${photoCount} photos are still used, only to place them on the cloud (a few minutes), so the 3D equipment boxes work as usual.`:' Without photos you get the 3D view; equipment boxes need the photos.'} Clouds need a coordinate system (LAS header or .prj) to line up with the photos' GPS.</p>}
  <label className={'pc-drop'+(busy?' disabled':'')}><UploadCloud size={20}/><span><b>Choose point cloud files</b><small>Several tiles of one survey are combined; large surveys are cropped to the area around the drone flight</small></span>
   <input type="file" multiple accept={ACCEPT} disabled={!!busy} onChange={e=>{choose(e.target.files);e.target.value=''}}/></label>
  {progress&&<div className="pc-progress"><div><span>{progress.name}</span><b>{mb(progress.loaded)} / {mb(progress.total)}</b></div><i><s style={{width:`${progress.total?progress.loaded/progress.total*100:0}%`}}/></i></div>}
  {files.length>0&&<ul className="pc-files">{files.map(f=><li key={f.id}><span className={'pc-fmt '+f.role}>{f.file_format}</span><b>{f.filename}</b><small>{mb(f.size_bytes)}{f.role==='sidecar'?' · sidecar':''}</small>
   <button title="Remove" disabled={!!busy} onClick={()=>remove(f)}><Trash2 size={14}/></button></li>)}</ul>}
  {clouds.length>0&&<button className="btn primary pc-build" disabled={!!busy} onClick={build}><Box size={15}/>{busy==='build'?'Starting…':`Build 3D model from point cloud${clouds.length>1?` (${clouds.length} files)`:''}`}</button>}
  {note&&<div className="pc-note"><CheckCircle2 size={14}/>{note}</div>}
  {error&&<div className="pc-error"><AlertTriangle size={14}/>{error}</div>}
 </div>;
}
