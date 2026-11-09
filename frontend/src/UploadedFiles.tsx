import {useMemo,useState} from 'react';
import {AlertTriangle,CheckCircle2,ChevronDown,ChevronRight,Film,Image as ImageIcon,Search,Trash2} from 'lucide-react';
import {Media,json,request} from './api';

/** The inspection's uploaded photos and videos, to delete files that were uploaded by mistake: from storage and the
 * database, together with their detections and reviews (that part needs the manager role; the server checks). */
const SHOW=300;
const size=(n:number)=>n>=1e9?`${(n/1e9).toFixed(2)} GB`:n>=1e6?`${(n/1e6).toFixed(1)} MB`:`${Math.max(1,Math.round(n/1e3))} kB`;
type Result={deleted:number;frames:number;detections:number;reviews:number;approved_boxes:number;models_affected:number};

export default function UploadedFiles({inspectionId,media,onChanged,disabled}:{inspectionId:number;media:Media[];onChanged:()=>void;disabled?:boolean}){
 const [open,setOpen]=useState(false),[q,setQ]=useState(''),[sel,setSel]=useState<Set<number>>(new Set()),[busy,setBusy]=useState(false);
 const [msg,setMsg]=useState<{ok:boolean;text:string}|null>(null);
 const top=useMemo(()=>media.filter(m=>m.parent_media_id==null).sort((a,b)=>a.filename.localeCompare(b.filename,undefined,{numeric:true})),[media]);
 const shown=useMemo(()=>{const s=q.trim().toLowerCase();return s?top.filter(m=>m.filename.toLowerCase().includes(s)):top},[top,q]);
 const chosen=top.filter(m=>sel.has(m.id)),processed=chosen.filter(m=>m.processed).length,allShown=shown.length>0&&shown.every(m=>sel.has(m.id));
 const toggle=(id:number)=>setSel(s=>{const n=new Set(s);n.has(id)?n.delete(id):n.add(id);return n});
 async function remove(){
  if(!chosen.length)return;
  const bytes=chosen.reduce((a,m)=>a+m.size_bytes,0);
  if(!confirm(`Delete ${chosen.length} file${chosen.length===1?'':'s'} (${size(bytes)}) from storage?`
   +(processed?`\n\n${processed} of them already have detections or reviews: those are deleted with them, and 3D models built with them must be rebuilt.`:'')
   +'\n\nThis cannot be undone.'))return;
  setBusy(true);setMsg(null);
  try{const r=await request<Result>(`/api/v1/inspections/${inspectionId}/media/delete`,json('POST',{media_ids:chosen.map(m=>m.id)}));
   setMsg({ok:true,text:`Deleted ${r.deleted} file${r.deleted===1?'':'s'}`+(r.frames?` (and ${r.frames} video frames)`:'')
    +(r.detections||r.reviews||r.approved_boxes?` with ${r.detections} detections, ${r.reviews} reviews and ${r.approved_boxes} approved boxes`:'')+'.'
    +(r.models_affected?` ${r.models_affected} 3D model${r.models_affected===1?' was':'s were'} built with them: rebuild before placing 3D boxes.`:'')});
   setSel(new Set());onChanged()}
  catch(e){setMsg({ok:false,text:String((e as Error).message||e)})}finally{setBusy(false)}}
 if(!top.length&&!msg)return null;  // keep the result in view after deleting everything
 return <div className="up-files">
  <button type="button" className="up-files-head" onClick={()=>setOpen(o=>!o)}>{open?<ChevronDown size={15}/>:<ChevronRight size={15}/>}
   <b>Uploaded files</b><small>{top.length} file{top.length===1?'':'s'} · {size(top.reduce((a,m)=>a+m.size_bytes,0))} · delete files uploaded by mistake</small></button>
  {open&&<div className="up-files-body">
   <div className="up-files-tools">
    <label className="up-files-search"><Search size={14}/><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search file names"/></label>
    <label className="up-files-all"><input type="checkbox" checked={allShown} disabled={disabled||!shown.length} onChange={()=>setSel(s=>{const n=new Set(s);for(const m of shown)allShown?n.delete(m.id):n.add(m.id);return n})}/>
     Select {q?'matching':'all'} ({shown.length})</label>
    <button type="button" className="btn outline up-files-delete" disabled={disabled||busy||!chosen.length} onClick={remove}><Trash2 size={14}/>{busy?'Deleting…':`Delete selected${chosen.length?` (${chosen.length})`:''}`}</button>
   </div>
   <ul className="up-files-list">{shown.slice(0,SHOW).map(m=><li key={m.id} className={sel.has(m.id)?'on':''} onClick={()=>!disabled&&toggle(m.id)}>
    <input type="checkbox" checked={sel.has(m.id)} disabled={disabled} onChange={()=>toggle(m.id)} onClick={e=>e.stopPropagation()}/>
    {m.content_type.startsWith('video/')?<Film size={14}/>:<ImageIcon size={14}/>}<span title={m.filename}>{m.filename}</span><small>{size(m.size_bytes)}</small>
    {m.processed&&<em title="Has detections or reviews">processed</em>}</li>)}</ul>
   {shown.length>SHOW&&<small className="muted">Showing {SHOW} of {shown.length}: search to narrow the list. "Select all" includes every matching file.</small>}
   {msg&&<div className={'up-files-msg '+(msg.ok?'ok':'error')}>{msg.ok?<CheckCircle2 size={14}/>:<AlertTriangle size={14}/>}{msg.text}</div>}
  </div>}
 </div>;
}
