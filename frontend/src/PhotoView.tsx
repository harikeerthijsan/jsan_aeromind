import {useEffect,useMemo,useRef,useState} from 'react';
import {AlertTriangle,ArrowUpFromLine,ChevronLeft,ChevronRight,Eye,EyeOff,Maximize,Ruler,Trash2} from 'lucide-react';
import {API,Inspection,Site,json,request} from './api';
import {Model3D,classColor} from './Reconstruction';
import './photoview.css';

/** Photo 3D view: the drone photos, with the 3D model's point cloud behind every pixel. A click on a photo is the 3D
 * point the camera sees there, so distances and heights are measured on the clear photo in real units. */
type Photo={media_id:number;filename:string;url:string;width:number;height:number;angle?:number};
type Point={x:number;y:number;z:number;height_m:number;distance_from_camera:number;snapped_px:number};
type Pick={u:number;v:number;p:Point|null};
type Measure={id:number;media_id:number;filename:string;kind:'distance'|'height';a:Pick;b?:Pick};
type Det={id:number;media_id:number;class_name:string;confidence:number;x1:number;y1:number;x2:number;y2:number;image_width:number;image_height:number};
type Tool='distance'|'height'|'look';
const insCode=(id:number)=>`INS-${String(id).padStart(4,'0')}`;

export default function PhotoView({site,inspections}:{site?:Site;inspections:Inspection[]}){
 const [insId,setInsId]=useState<number|null>(null),[models,setModels]=useState<Model3D[]>([]),[modelId,setModelId]=useState<number|null>(null);
 const [photos,setPhotos]=useState<Photo[]>([]),[scaled,setScaled]=useState(true),[idx,setIdx]=useState(0),[err,setErr]=useState(''),[loading,setLoading]=useState('');
 const [tool,setTool]=useState<Tool>('distance'),[pending,setPending]=useState<Pick|null>(null),[measures,setMeasures]=useState<Measure[]>([]),[busy,setBusy]=useState(false);
 const [unit,setUnit]=useState<'m'|'ft'>('m'),[showBoxes,setShowBoxes]=useState(true),[dets,setDets]=useState<Det[]>([]),[note,setNote]=useState('');
 const inspection=inspections.find(i=>i.id===insId)||inspections[0];
 useEffect(()=>{setInsId(null)},[site?.id]);
 // the inspection's latest completed 3D model
 useEffect(()=>{if(!inspection)return;let live=true;setModels([]);setModelId(null);setPhotos([]);setErr('');
  request<Model3D[]>(`/api/v1/inspections/${inspection.id}/models`).then(m=>{if(!live)return;const done=m.filter(x=>x.status==='completed').sort((a,b)=>b.id-a.id);setModels(done);setModelId(done[0]?.id??null)}).catch(e=>live&&setErr(String(e.message||e)));
  return()=>{live=false}},[inspection?.id]);
 useEffect(()=>{if(!modelId)return;let live=true;setLoading('Loading the photos and the 3D model…');setErr('');setIdx(0);setPending(null);
  request<{scaled:boolean;photos:Photo[]}>(`/api/v1/models/${modelId}/photos`).then(r=>{if(!live)return;setPhotos(r.photos);setScaled(r.scaled)})
   .catch(e=>live&&setErr(String(e.message||e))).finally(()=>live&&setLoading(''));
  return()=>{live=false}},[modelId]);
 const photo=photos[idx];
 useEffect(()=>{if(!photo||!inspection)return;let live=true;setDets([]);setPending(null);setNote('');
  request<Det[]>(`/api/v1/inspections/${inspection.id}/detections?media_id=${photo.media_id}`).then(d=>live&&setDets(d)).catch(()=>{});
  return()=>{live=false}},[photo?.media_id]);
 useEffect(()=>{const k=(e:KeyboardEvent)=>{if((e.target as HTMLElement)?.tagName==='INPUT'||(e.target as HTMLElement)?.tagName==='SELECT')return;
  if(e.key==='ArrowRight')setIdx(i=>Math.min(photos.length-1,i+1));if(e.key==='ArrowLeft')setIdx(i=>Math.max(0,i-1));if(e.key==='Escape')setPending(null)};
  window.addEventListener('keydown',k);return()=>window.removeEventListener('keydown',k)},[photos.length]);

 // ---------- the photo: zoom (wheel, at the cursor) and pan (drag); a click (no drag) measures ----------
 const box=useRef<HTMLDivElement>(null),[view,setView]=useState({k:1,x:0,y:0}),[full,setFull]=useState(false),drag=useRef<{x:number;y:number;vx:number;vy:number;moved:boolean}|null>(null);
 const fit=()=>{const el=box.current;if(!el||!photo)return;const r=el.getBoundingClientRect(),k=Math.min(r.width/photo.width,r.height/photo.height);setView({k,x:(r.width-photo.width*k)/2,y:(r.height-photo.height*k)/2})};
 useEffect(()=>{setFull(false);fit()},[photo?.media_id]);
 useEffect(()=>{const el=box.current;if(!el)return;const ro=new ResizeObserver(()=>fit());ro.observe(el);return()=>ro.disconnect()},[photo?.media_id]);
 useEffect(()=>{const el=box.current;if(!el)return;const w=(e:WheelEvent)=>{e.preventDefault();const r=el.getBoundingClientRect(),cx=e.clientX-r.left,cy=e.clientY-r.top,f=Math.exp(-e.deltaY*.0015);
  setView(v=>{const k=Math.min(8,Math.max(.02,v.k*f));return {k,x:cx-(cx-v.x)*k/v.k,y:cy-(cy-v.y)*k/v.k}});if(f>1)setFull(true)};
  el.addEventListener('wheel',w,{passive:false});return()=>el.removeEventListener('wheel',w)},[]);
 async function pick(u:number,v:number){if(!modelId||!photo||tool==='look')return;setBusy(true);setNote('');
  try{const r=await request<{points:(Point|null)[]}>(`/api/v1/models/${modelId}/photos/${photo.media_id}/locate`,json('POST',{points:[[u,v]]}));const p=r.points[0];
   if(!p){setNote('No 3D point here: the sky, or outside the 3D model. Click on the structure.');return}
   const here:Pick={u,v,p};
   if(tool==='height'){setMeasures(m=>[...m,{id:Date.now(),media_id:photo.media_id,filename:photo.filename,kind:'height',a:here}]);return}
   if(!pending){setPending(here);return}
   setMeasures(m=>[...m,{id:Date.now(),media_id:photo.media_id,filename:photo.filename,kind:'distance',a:pending,b:here}]);setPending(null)}
  catch(e){setErr(String((e as Error).message||e))}finally{setBusy(false)}}
 const onDown=(e:React.PointerEvent)=>{drag.current={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y,moved:false};(e.target as Element).setPointerCapture?.(e.pointerId)};
 const onMove=(e:React.PointerEvent)=>{const d=drag.current;if(!d)return;const dx=e.clientX-d.x,dy=e.clientY-d.y;if(Math.hypot(dx,dy)>4)d.moved=true;if(d.moved)setView(v=>({...v,x:d.vx+dx,y:d.vy+dy}))};
 const onUp=(e:React.PointerEvent)=>{const d=drag.current;drag.current=null;if(!d||d.moved||!photo)return;const r=box.current!.getBoundingClientRect(),u=(e.clientX-r.left-view.x)/view.k,v=(e.clientY-r.top-view.y)/view.k;
  if(u>=0&&v>=0&&u<photo.width&&v<photo.height)pick(u,v)};
 const S=(u:number,v:number)=>[view.x+u*view.k,view.y+v*view.k] as const;

 const f=unit==='ft'?3.28084:1,fmt=(m:number)=>`${(m*f).toFixed(2)} ${scaled?unit:'units'}`;
 const dist=(a:Point,b:Point)=>({d:Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z),h:Math.hypot(a.x-b.x,a.y-b.y),v:Math.abs(a.z-b.z)});
 const here=measures.filter(m=>m.media_id===photo?.media_id);
 const order=useMemo(()=>photos.map((p,i)=>({p,i})),[photos]);
 if(!site)return <div className="pv-empty">Select a tower first.</div>;
 return <div className="pv-page">
  <aside className="pv-side">
   <label className="pv-field"><span>Inspection</span><select value={inspection?.id??''} onChange={e=>setInsId(+e.target.value)}>{inspections.map(i=><option key={i.id} value={i.id}>{insCode(i.id)} · {i.label}</option>)}</select></label>
   {models.length>1&&<label className="pv-field"><span>3D model</span><select value={modelId??''} onChange={e=>setModelId(+e.target.value)}>{models.map(m=><option key={m.id} value={m.id}>Model #{m.id} · {m.quality?.source==='pointcloud'?'LiDAR point cloud':m.dense?'Dense':'Sparse'}</option>)}</select></label>}
   <div className="pv-tools">
    <span>Measure</span>
    <button className={tool==='distance'?'on':''} onClick={()=>{setTool('distance');setPending(null)}}><Ruler size={14}/>Distance</button>
    <button className={tool==='height'?'on':''} onClick={()=>{setTool('height');setPending(null)}}><ArrowUpFromLine size={14}/>Height</button>
    <button className={tool==='look'?'on':''} onClick={()=>{setTool('look');setPending(null)}}><Eye size={14}/>Look only</button>
   </div>
   <p className="pv-help">{tool==='distance'?(pending?'Click the second point.':'Click two points on the photo: the 3D distance between them, its vertical and horizontal parts.'):tool==='height'?'Click a point: its height above the ground.':'Zoom with the wheel, drag to move.'}
    {' '}Clicks snap to the structure behind the photo (the 3D point cloud).</p>
   <div className="pv-row"><span>Units</span><div className="pv-seg">{(['m','ft'] as const).map(u=><button key={u} className={unit===u?'on':''} onClick={()=>setUnit(u)}>{u}</button>)}</div>
    <button className="pv-link" onClick={()=>setShowBoxes(s=>!s)}>{showBoxes?<EyeOff size={13}/>:<Eye size={13}/>}{showBoxes?'Hide':'Show'} detections</button></div>
   {!scaled&&<div className="pv-warn"><AlertTriangle size={14}/>This 3D model has no real-world scale yet: values are in model units. Detect the scale in the 3D Digital Twin.</div>}
   <div className="pv-list"><div className="pv-list-head"><b>Measurements</b>{measures.length>0&&<button className="pv-link" onClick={()=>setMeasures([])}><Trash2 size={13}/>Clear</button>}</div>
    {!measures.length&&<small className="muted">None yet.</small>}
    {measures.map(m=>{const a=m.a.p!,b=m.b?.p;const r=b?dist(a,b):null;return <div key={m.id} className={'pv-item'+(m.media_id===photo?.media_id?' here':'')} onClick={()=>{const i=photos.findIndex(p=>p.media_id===m.media_id);if(i>=0)setIdx(i)}}>
     <b>{m.kind==='distance'&&r?fmt(r.d):fmt(a.height_m)}</b><small>{m.kind==='distance'&&r?`vertical ${fmt(r.v)} · horizontal ${fmt(r.h)}`:'height above ground'}</small><em>{m.filename}</em>
     <button title="Remove" onClick={e=>{e.stopPropagation();setMeasures(ms=>ms.filter(x=>x.id!==m.id))}}><Trash2 size={12}/></button></div>})}
   </div>
  </aside>
  <section className="pv-main">
   <div className="pv-bar"><b>{photo?photo.filename:'Photo 3D view'}</b><small>{photos.length?`${idx+1} of ${photos.length} · ordered around the tower`:''}</small>
    <span className="pv-bar-tools"><button onClick={()=>setIdx(i=>Math.max(0,i-1))} disabled={idx<=0} title="Previous photo (←)"><ChevronLeft size={16}/></button><button onClick={()=>setIdx(i=>Math.min(photos.length-1,i+1))} disabled={idx>=photos.length-1} title="Next photo (→)"><ChevronRight size={16}/></button>
     <button onClick={fit} title="Fit the photo"><Maximize size={15}/></button></span></div>
   <div ref={box} className={'pv-view'+(tool==='look'?'':' measuring')+(busy?' busy':'')} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
    {photo&&<img key={photo.media_id+(full?'f':'t')} src={full?API+photo.url:`${API}/api/v1/media/${photo.media_id}/thumb?w=1280`} onLoad={()=>!full&&setFull(true)} draggable={false} alt={photo.filename}
     style={{position:'absolute',left:view.x,top:view.y,width:photo.width*view.k,height:photo.height*view.k}}/>}
    {photo&&<svg className="pv-overlay">
     {showBoxes&&dets.map(d=>{const sx=photo.width/(d.image_width||photo.width),sy=photo.height/(d.image_height||photo.height),[x1,y1]=S(d.x1*sx,d.y1*sy),[x2,y2]=S(d.x2*sx,d.y2*sy);
      return <g key={d.id}><rect x={x1} y={y1} width={x2-x1} height={y2-y1} fill="none" stroke={classColor(d.class_name)} strokeWidth={1.5}/>{x2-x1>60&&<text x={x1+3} y={y1-4} fill={classColor(d.class_name)} fontSize={11}>{d.class_name.replaceAll('_',' ')}</text>}</g>})}
     {here.map(m=>{const [ax,ay]=S(m.a.u,m.a.v);if(m.kind==='height')return <g key={m.id}><circle cx={ax} cy={ay} r={5} className="pv-pt"/><text x={ax+9} y={ay-8} className="pv-label">{fmt(m.a.p!.height_m)} high</text></g>;
      const [bx,by]=S(m.b!.u,m.b!.v),r=dist(m.a.p!,m.b!.p!);
      return <g key={m.id}><line x1={ax} y1={ay} x2={bx} y2={by} className="pv-line"/><circle cx={ax} cy={ay} r={5} className="pv-pt"/><circle cx={bx} cy={by} r={5} className="pv-pt"/><text x={(ax+bx)/2+8} y={(ay+by)/2-8} className="pv-label">{fmt(r.d)}</text></g>})}
     {pending&&(()=>{const [px,py]=S(pending.u,pending.v);return <circle cx={px} cy={py} r={6} className="pv-pt pending"/>})()}
    </svg>}
    {loading&&<div className="pv-msg">{loading}</div>}
    {!loading&&!photos.length&&!err&&<div className="pv-msg">{models.length?'This 3D model has no photos with camera positions.':'No completed 3D model for this inspection: build one on the 3D Reconstruction page.'}</div>}
    {err&&<div className="pv-msg error">{err}</div>}
    {note&&<div className="pv-note">{note}</div>}
   </div>
   <div className="pv-strip">{order.map(({p,i})=><button key={p.media_id} className={i===idx?'on':''} onClick={()=>setIdx(i)} title={p.filename}>
    <img loading="lazy" src={`${API}/api/v1/media/${p.media_id}/thumb?w=200`} alt=""/>{measures.some(m=>m.media_id===p.media_id)&&<i/>}</button>)}</div>
  </section>
 </div>;
}
