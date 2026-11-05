export const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '');  // production build: same origin as the page
export type AuthRole='viewer'|'inspector'|'manager'|'admin';
export type AuthUser={id:number;email:string;name:string;role:AuthRole;active:boolean;created_at:string;last_login:string|null};
const TOKEN_KEY='aeromind_session';
export const getAuthToken=()=>localStorage.getItem(TOKEN_KEY);
export function setAuthToken(token:string|null){if(token)localStorage.setItem(TOKEN_KEY,token);else localStorage.removeItem(TOKEN_KEY)}
export type Site={id:number;code:string;name:string;region:string;tower_type:string;height_m:number|null;status:string;latitude?:number|null;longitude?:number|null;operator?:string|null;address?:string|null;notes?:string|null;created_at?:string|null};
export type Inspection={id:number;site_id:number;label:string;status:string;created_at:string};
export type Asset={id:number;site_id:number;code:string;category:string;condition:string;identity_status:string;localization_status:string;x:number|null;y:number|null;z:number|null;description:string};
export type Defect={id:number;asset_id:number;defect_type:string;proposed_grade:string;review_status:string;notes:string};
export type Job={id:number;inspection_id:number;status:string;stage:string;progress:number;detail:string;kind?:string;media_ids?:number[]|null};
export type Media={id:number;inspection_id:number;filename:string;content_type:string;size_bytes:number;url:string;parent_media_id:number|null;frame_index:number|null;frame_time:number|null};
export type Detection={id:number;job_id:number;media_id:number;asset_id:number|null;frame_index:number;class_id:number;class_name:string;confidence:number;x1:number;y1:number;x2:number;y2:number;image_width:number;image_height:number;model_name:string;review_status:string};
export type Review={id:number;inspection_id:number;media_id:number;frame_index:number;status:string;workflow:string;job_id:number|null;box_count:number;notes:string;reviewed_at:string};
export type Annotation={id:number;review_id:number;media_id:number;detection_id:number|null;source:string;class_id:number;class_name:string;confidence:number|null;x1:number;y1:number;x2:number;y2:number;image_width:number;image_height:number;manufacturer:string|null;product_model:string|null;vendor_source:string|null;vendor_confidence:number|null;vendor_evidence:string|null;vendor_basis:string|null;condition_grade:string|null;condition_score:number|null;condition_defects:ConditionDefect[]|null;condition_source:string|null;condition_confidence:number|null;condition_verified:string|null;condition_summary:string|null};
export type MediaAnnotations={review:Review|null;annotations:Annotation[]};
export type ClassInfo={id:number;name:string};
export type VendorInfo={classes:string[];manufacturers:Record<string,string[]>;ai:{status:string;model:string|null;url:string};reference_dir:string;folder_references:Record<string,number>;reviewer_references:Record<string,number>;library_manufacturers:Record<string,string[]>};
export type VendorRef={manufacturer:string;model:string;view:string;status:string;page:string;notes:string;match:boolean;url:string};
export type ConditionDefect={type:string;severity:string;extent_percent:number;location:string;evidence:string;source?:string;confirmed?:boolean};
export type ConditionResult={grade:string;health_score:number|null;defects:ConditionDefect[];rust_percent:number;image_quality:string;confidence:number;summary:string;recommended_action:string;verified?:string;skipped?:boolean;model:string|null;seconds:number};
export type VendorCandidate={manufacturer:string;similarity:number;brands:string[];matching_features:string[]};
export type VendorSuggestion={candidates:VendorCandidate[];ambiguous:boolean;manufacturer:string;product_model:string|null;confidence:number;basis:string;evidence:string;features:Record<string,string>;reference_vendors:string[];visible_text:string;model:string;seconds:number};
export type DetectorInfo={status:string;model:string|null;imgsz:number;conf:number;device:string;vid_stride:number;loaded:boolean;classes?:Record<string,string>};
export type RegistrationCandidate={id:number;model_id:number;inspection_id:number;site_id:number;class_name:string;x:number;y:number;z:number;confidence:number;status:string;evidence_count:number;media_count:number;point_count:number;spread:number;detection_ids:number[];created_at:string;size_w?:number|null;size_h?:number|null;size_d?:number|null;yaw?:number|null;box_method?:string|null;box_points?:number|null;fit_iou?:number|null};
export async function request<T>(path:string, options:RequestInit={}):Promise<T>{
 const headers=new Headers(options.headers);const token=getAuthToken();if(token)headers.set('Authorization',`Bearer ${token}`);options={credentials:'include',...options};  // include: the session cookie for images/files in local dev
 const r=await fetch(API+path,{...options,headers});
 if(!r.ok){
  if(r.status===401&&!path.endsWith('/auth/login')){setAuthToken(null);window.dispatchEvent(new Event('aeromind-auth-expired'))}
  const raw=(await r.text()).slice(0,500);let detail=raw;try{detail=JSON.parse(raw).detail||raw}catch{/* plain-text API error */}
  throw Error(detail||`Request failed (${r.status})`)
 }
 return r.status===204?undefined as T:r.json();
}
export const json=(method:string,body:unknown):RequestInit=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
export async function download(path:string,filename:string){const headers=new Headers();const token=getAuthToken();if(token)headers.set('Authorization',`Bearer ${token}`);const r=await fetch(API+path,{headers,credentials:'include'});if(!r.ok)throw Error((await r.text()).slice(0,250));const url=URL.createObjectURL(await r.blob());const anchor=document.createElement('a');anchor.href=url;anchor.download=filename;document.body.appendChild(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(url)}
/** Upload one file with byte progress and a way to cancel it (fetch has neither). Rejects with name 'AbortError' when cancelled. */
export function uploadFile<T>(path:string,file:File,onProgress:(loaded:number,total:number)=>void,signal?:AbortSignal):Promise<T>{
 const body=new FormData();body.append('file',file);return send<T>('POST',path,body,file.name,l=>onProgress(l,file.size),signal)}
function send<T>(method:string,path:string,body:XMLHttpRequestBodyInit,label:string,onProgress:(loaded:number)=>void,signal?:AbortSignal):Promise<T>{
 return new Promise((resolve,reject)=>{const x=new XMLHttpRequest();x.open(method,API+path);x.withCredentials=true;
  const t=getAuthToken();if(t)x.setRequestHeader('Authorization',`Bearer ${t}`);
  if(signal?.aborted){reject(Object.assign(Error('Upload cancelled'),{name:'AbortError'}));return}
  const abort=()=>x.abort();signal?.addEventListener('abort',abort,{once:true});
  x.upload.onprogress=e=>onProgress(e.loaded);
  x.onabort=()=>reject(Object.assign(Error('Upload cancelled'),{name:'AbortError'}));
  x.onerror=()=>{signal?.removeEventListener('abort',abort);reject(Object.assign(Error(`Network error while uploading ${label}`),{status:0}))};
  x.onload=()=>{signal?.removeEventListener('abort',abort);if(x.status>=200&&x.status<300)resolve(JSON.parse(x.responseText));
   else{let d=x.responseText;try{d=JSON.parse(d).detail}catch{/* plain text */}if(x.status===401)window.dispatchEvent(new Event('aeromind-auth-expired'));reject(Object.assign(Error(`${label}: ${d||x.status}`),{status:x.status}))}};
  x.send(body)})}
/** Large file in 32 MB parts (3 at a time, each retried): no single request comes near the hosting proxy's 5-minute
 * limit. kind 'media' (photo/video) or 'pointcloud'; resolves with the stored Media / PointCloudFile row. */
export async function uploadChunked<T>(inspectionId:number,kind:'media'|'pointcloud',file:File,onProgress:(loaded:number,total:number)=>void,signal?:AbortSignal):Promise<T>{
 const s=await request<{upload_id:string;part_size:number;parts:number}>(`/api/v1/inspections/${inspectionId}/chunked-uploads`,{...json('POST',{kind,filename:file.name,size:file.size,content_type:file.type}),signal});
 const sent=new Map<number,number>(),report=()=>onProgress([...sent.values()].reduce((a,b)=>a+b,0),file.size);
 let next=1,failed=false;
 const worker=async()=>{while(next<=s.parts&&!failed){const n=next++,blob=file.slice((n-1)*s.part_size,Math.min(file.size,n*s.part_size));
  for(let attempt=1;;attempt++){
   try{await send('PUT',`/api/v1/chunked-uploads/${s.upload_id}/parts/${n}`,blob,file.name,l=>{sent.set(n,l);report()},signal);sent.set(n,blob.size);report();break}
   catch(e){const st=(e as {status?:number}).status??0;sent.set(n,0);report();
    if((e as Error).name==='AbortError'||attempt>=5||(st>=400&&st<500&&st!==408&&st!==429)){failed=true;throw e}
    await new Promise(r=>setTimeout(r,1500*attempt*attempt))}}}};
 try{await Promise.all(Array.from({length:Math.min(3,s.parts)},worker));
  return await request<T>(`/api/v1/chunked-uploads/${s.upload_id}/complete`,{method:'POST'})}
 catch(e){request(`/api/v1/chunked-uploads/${s.upload_id}`,{method:'DELETE'}).catch(()=>{});throw e}}
