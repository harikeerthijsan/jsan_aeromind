import {useEffect,useMemo,useRef,useState} from 'react';
import {Canvas,ThreeEvent,useFrame,useThree} from '@react-three/fiber';import {OrbitControls,Grid,TransformControls} from '@react-three/drei';import {LabelAnchor} from './Label3D';
import * as THREE from 'three';import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
import {Box,Play,AlertTriangle,Download,Camera,Image as ImageIcon,Film,CheckCircle2,Layers,Maximize,Minimize} from 'lucide-react';
import {API,request,Inspection,Job,Media,RegistrationCandidate} from './api';

// 3D models built by COLMAP photogrammetry (backend app/reconstruct.py)
type ReconInfo={status:string;colmap:string|null;colmap_path_env:string|null;video_stride:number;max_images:number;dense_max_size:number};
export type Model3D={id:number;inspection_id:number;job_id:number;status:string;dense:boolean;created_at:string|null;urls:Record<string,string>;
 quality:{registered_images?:number;images_total?:number;sparse_points?:number;dense_points?:number;mesh_faces?:number;mean_reprojection_error_px?:number|null;mean_track_length?:number|null;models_found?:number;seconds?:number;error?:string;source_model?:number}};
type Cam={name:string;center:[number,number,number];dir:[number,number,number]};
const insCode=(id:number)=>'INS-'+String(id).padStart(4,'0');
const fmt=(n?:number|null)=>n==null?'—':n.toLocaleString();
const running=(j:Job)=>['queued','running'].includes(j.status);

function usePly(url?:string){
 const [geo,setGeo]=useState<THREE.BufferGeometry|null>(null),[err,setErr]=useState('');
 useEffect(()=>{setGeo(null);setErr('');if(!url)return;let live=true;new PLYLoader().load(API+url,g=>{if(live)setGeo(g)},undefined,e=>live&&setErr(String((e as Error)?.message||e)));return()=>{live=false}},[url]);
 return {geo,err};
}
function Cameras({url}:{url?:string}){
 const [cams,setCams]=useState<Cam[]>([]);
 useEffect(()=>{if(!url)return;fetch(API+url).then(r=>r.json()).then(setCams).catch(()=>setCams([]))},[url]);
 // dots stay a few pixels on screen at any zoom; direction ticks scale with the spacing between shots so they never pile up
 const objs=useMemo(()=>{if(!cams.length)return null;const P=cams.map(c=>new THREE.Vector3(...c.center));
  const gaps=P.slice(1).map((p,i)=>p.distanceTo(P[i])).filter(g=>g>0).sort((a,b)=>a-b),gap=gaps.length?gaps[Math.floor(gaps.length/2)]:.5,len=Math.min(Math.max(gap*.8,.05),.6);
  const dots=new THREE.Points(new THREE.BufferGeometry().setFromPoints(P),new THREE.PointsMaterial({color:'#ffb020',size:5,sizeAttenuation:false}));
  const ticks=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(cams.flatMap((c,i)=>[P[i],P[i].clone().add(new THREE.Vector3(...c.dir).normalize().multiplyScalar(len))])),
   new THREE.LineBasicMaterial({color:'#ffb020',transparent:true,opacity:.55}));
  const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(P),new THREE.LineBasicMaterial({color:'#ffb020',transparent:true,opacity:.3}));
  return [dots,ticks,path]},[cams]);
 return objs&&<group>{objs.map((o,i)=><primitive key={i} object={o}/>)}</group>;
}
// stable colour per equipment class (same palette as the 2D box editor)
const CLASS_PALETTE=['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000','#aaffc3'];
export const classColor=(name:string)=>CLASS_PALETTE[[...name].reduce((h,ch)=>(h*31+ch.charCodeAt(0))>>>0,7)%CLASS_PALETTE.length];
function EquipmentBox({m,on}:{m:RegistrationCandidate;on:boolean}){
 const c=classColor(m.class_name),geo=useMemo(()=>new THREE.BoxGeometry(m.size_w!,m.size_h!,m.size_d!),[m.size_w,m.size_h,m.size_d]),edges=useMemo(()=>new THREE.EdgesGeometry(geo),[geo]);
 return <><mesh geometry={geo} renderOrder={20}><meshBasicMaterial color={c} transparent opacity={on?.42:.16} depthTest={false} depthWrite={false}/></mesh>
  <lineSegments geometry={edges} renderOrder={21} raycast={()=>{}}><lineBasicMaterial color={on?'#ffffff':c} depthTest={false}/></lineSegments></>;
}
type MarkerEditMode='translate'|'rotate';
function YawEdgeHandles({m,onRotate}:{m:RegistrationCandidate;onRotate:(yaw:number)=>void}){
 const controls=useThree(state=>state.controls) as unknown as {enabled:boolean}|null,drag=useRef<{angle:number;yaw:number}|null>(null),center=useMemo(()=>new THREE.Vector3(m.x,m.z,m.y),[m.x,m.y,m.z]);
 const width=Math.max(m.size_w||.04,.04),depth=Math.max(m.size_d||.025,.025),rx=width*.72+.018,rz=depth*.72+.018,height=(m.size_h||.05)/2+.025,radius=Math.max(.009,Math.min(.018,Math.max(width,depth)*.14));
 const outline=useMemo(()=>new THREE.BufferGeometry().setFromPoints([[-rx,height,-rz],[rx,height,-rz],[rx,height,rz],[-rx,height,rz]].map(([x,y,z])=>new THREE.Vector3(x,y,z))),[rx,rz,height]);
 useEffect(()=>()=>{if(controls)controls.enabled=true;document.body.style.cursor='auto'},[controls]);
 const pointerAngle=(e:ThreeEvent<PointerEvent>)=>{const hit=new THREE.Vector3(),plane=new THREE.Plane(new THREE.Vector3(0,1,0),-center.y);return e.ray.intersectPlane(plane,hit)?Math.atan2(hit.x-center.x,hit.z-center.z):null};
 const start=(e:ThreeEvent<PointerEvent>)=>{e.stopPropagation();const angle=pointerAngle(e);if(angle==null)return;drag.current={angle,yaw:m.yaw||0};controls&&(controls.enabled=false);(e.target as unknown as {setPointerCapture:(id:number)=>void}).setPointerCapture(e.pointerId);document.body.style.cursor='grabbing'};
 const move=(e:ThreeEvent<PointerEvent>)=>{if(!drag.current)return;e.stopPropagation();const angle=pointerAngle(e);if(angle==null)return;let delta=angle-drag.current.angle;while(delta>Math.PI)delta-=2*Math.PI;while(delta<-Math.PI)delta+=2*Math.PI;onRotate(drag.current.yaw+delta)};
 const stop=(e:ThreeEvent<PointerEvent>)=>{if(!drag.current)return;e.stopPropagation();drag.current=null;controls&&(controls.enabled=true);(e.target as unknown as {releasePointerCapture:(id:number)=>void}).releasePointerCapture(e.pointerId);document.body.style.cursor='grab'};
 const handles:[number,number,number][]=[[-rx,height,-rz],[rx,height,-rz],[rx,height,rz],[-rx,height,rz]];
 return <group position={[m.x,m.z,m.y]} rotation={[0,m.yaw||0,0]} renderOrder={30}>
  <lineLoop geometry={outline} raycast={()=>{}}><lineBasicMaterial color="#ffd34e" depthTest={false}/></lineLoop>
  {handles.map((position,index)=><mesh key={index} position={position} renderOrder={31} onPointerDown={start} onPointerMove={move} onPointerUp={stop} onPointerCancel={stop} onPointerOver={()=>{if(!drag.current)document.body.style.cursor='grab'}} onPointerOut={()=>{if(!drag.current)document.body.style.cursor='auto'}}><sphereGeometry args={[radius,14,10]}/><meshBasicMaterial color="#ffd34e" depthTest={false}/></mesh>)}
 </group>;
}
function RegistrationMarker({m,selected,editing,editMode,onPick,onTransform}:{m:RegistrationCandidate;selected:boolean;editing:boolean;editMode:MarkerEditMode;onPick?:(id:number)=>void;onTransform?:(id:number,position:[number,number,number],yaw:number)=>void}){
 const ref=useRef<THREE.Group>(null!),box=m.size_w!=null&&m.size_h!=null&&m.size_d!=null;
 const marker=<group ref={ref} position={[m.x,m.z,m.y]} rotation={[0,m.yaw||0,0]}
  onPointerOver={(e:ThreeEvent<PointerEvent>)=>{e.stopPropagation();document.body.style.cursor='pointer'}} onPointerOut={()=>{document.body.style.cursor='auto'}}
  onClick={(e:ThreeEvent<MouseEvent>)=>{e.stopPropagation();onPick?.(m.id)}}>
  {box?<EquipmentBox m={m} on={selected}/>:<mesh scale={selected?1.35:1} renderOrder={20}><sphereGeometry args={[.07,14,10]}/><meshBasicMaterial color={selected?'#ffdd57':'#ff9f1c'} depthTest={false}/></mesh>}
 </group>;
 if(!editing)return marker;
 return <>{marker}{editMode==='translate'?<TransformControls object={ref} mode="translate" space="world" size={.72} onObjectChange={()=>{const object=ref.current;if(object)onTransform?.(m.id,[object.position.x,object.position.y,object.position.z],object.rotation.y)}}/>:<YawEdgeHandles m={m} onRotate={yaw=>onTransform?.(m.id,[m.x,m.z,m.y],yaw)}/>}</>;
}
function RegistrationMarkers({markers,selected,editing,editMode,onPick,onTransform}:{markers:RegistrationCandidate[];selected?:number|null;editing?:number|null;editMode:MarkerEditMode;onPick?:(id:number)=>void;onTransform?:(id:number,position:[number,number,number],yaw:number)=>void}){
 // oriented 3D boxes where the size is known; older candidates without a size stay as dots
 return <group>{markers.map(m=><RegistrationMarker key={m.id} m={m} selected={selected===m.id} editing={editing===m.id} editMode={editMode} onPick={onPick} onTransform={onTransform}/>)}</group>;
}
type FocusAnimation={fromPosition:THREE.Vector3;toPosition:THREE.Vector3;fromTarget:THREE.Vector3;toTarget:THREE.Vector3;progress:number};
function CameraFocus({marker,request=0}:{marker?:RegistrationCandidate;request?:number}){
 const {camera,controls}=useThree(),animation=useRef<FocusAnimation|null>(null);
 useEffect(()=>{const orbit=controls as unknown as {target:THREE.Vector3;update:()=>void}|null;if(!marker||!orbit)return;
  const target=new THREE.Vector3(marker.x,marker.z,marker.y),direction=camera.position.clone().sub(orbit.target);
  if(direction.lengthSq()<1e-8)direction.set(1,.65,1);direction.normalize();
  const fov=THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov||45),largest=Math.max(marker.size_w||0,marker.size_h||0,marker.size_d||0,.05);
  const distance=Math.max(.32,largest/(2*Math.tan(fov/2))*2.8);
  animation.current={fromPosition:camera.position.clone(),toPosition:target.clone().add(direction.multiplyScalar(distance)),fromTarget:orbit.target.clone(),toTarget:target,progress:0};
 },[marker?.id,request,camera,controls]);
 useFrame((_,delta)=>{const a=animation.current,orbit=controls as unknown as {target:THREE.Vector3;update:()=>void}|null;if(!a||!orbit)return;
  a.progress=Math.min(1,a.progress+delta/.55);const t=a.progress*a.progress*(3-2*a.progress);
  camera.position.lerpVectors(a.fromPosition,a.toPosition,t);orbit.target.lerpVectors(a.fromTarget,a.toTarget,t);orbit.update();if(a.progress===1)animation.current=null;
 });
 return null;
}
export function ModelView({model,mode,pointSize,showCams,markers=[],selectedMarker,editingMarker,markerEditMode='translate',focusRequest=0,onMarkerPick,onMarkerTransform}:{model:Model3D;mode:'points'|'mesh';pointSize:number;showCams:boolean;markers?:RegistrationCandidate[];selectedMarker?:number|null;editingMarker?:number|null;markerEditMode?:MarkerEditMode;focusRequest?:number;onMarkerPick?:(id:number)=>void;onMarkerTransform?:(id:number,position:[number,number,number],yaw:number)=>void}){
 const useMesh=mode==='mesh'&&!!model.urls.web_mesh;const {geo,err}=usePly(useMesh?model.urls.web_mesh:model.urls.web_points);
 const obj=useMemo(()=>{if(!geo)return null;const colors=!!geo.getAttribute('color');
  if(useMesh){geo.computeVertexNormals();return new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:colors,color:colors?'#ffffff':'#c8d3de',side:THREE.DoubleSide,roughness:.9}))}
  return new THREE.Points(geo,new THREE.PointsMaterial({size:pointSize,vertexColors:colors,color:colors?'#ffffff':'#9fd0ff',sizeAttenuation:true}))},[geo,useMesh]);
 useEffect(()=>{if(obj instanceof THREE.Points)(obj.material as THREE.PointsMaterial).size=pointSize},[obj,pointSize]);
 const label=useRef<HTMLDivElement>(null),sel=markers.find(m=>m.id===selectedMarker);// selected box label, positioned by LabelAnchor
 return <><div ref={label} className="three-label floating-label">{sel&&<><strong>{sel.class_name.replaceAll('_',' ')}</strong><br/><small>Estimated · {sel.media_count} images · {Math.round(sel.confidence*100)}%</small></>}</div>{err&&<div className="recon-msg">Could not load the model: {err}</div>}{!geo&&!err&&<div className="recon-msg">Loading 3D model…</div>}
  <Canvas camera={{position:[16,12,16],fov:45,near:.05,far:500}} gl={{antialias:true}}>{/* high enough to show the tower and the drone path above it */}
   <color attach="background" args={['#0f1f31']}/><ambientLight intensity={.9}/><directionalLight position={[8,12,6]} intensity={1.2}/>
   {obj&&<primitive object={obj}/>} {showCams&&<Cameras url={model.urls.web_cameras}/>}<RegistrationMarkers markers={markers} selected={selectedMarker} editing={editingMarker} editMode={markerEditMode} onPick={onMarkerPick} onTransform={onMarkerTransform}/><LabelAnchor target={label} position={sel?[sel.x,sel.z+(sel.size_h??.2)/2+.05,sel.y]:null}/>
   <Grid position={[0,-5.2,0]} args={[30,30]} cellColor="#22405c" sectionColor="#2d5878" fadeDistance={45}/><OrbitControls makeDefault enableDamping target={[0,1,0]}/><CameraFocus marker={sel} request={focusRequest}/>
  </Canvas></>;
}

export default function Reconstruction({inspections,jobs,onChanged}:{inspections:Inspection[];jobs:Job[];onChanged:()=>void}){
 const [inspectionId,setInspectionId]=useState<number|null>(null),[info,setInfo]=useState<ReconInfo|null>(null),[models,setModels]=useState<Model3D[]>([]),[media,setMedia]=useState<Media[]>([]);
 const [selected,setSelected]=useState<number|null>(null),[dense,setDense]=useState(false),[mode,setMode]=useState<'points'|'mesh'>('points'),[pointSize,setPointSize]=useState(.035),[showCams,setShowCams]=useState(false),[error,setError]=useState(''),[full,setFull]=useState(false);
 const shell=useRef<HTMLDivElement>(null);
 const inspection=inspections.find(i=>i.id===inspectionId)||inspections[0];
 const recJobs=jobs.filter(j=>j.inspection_id===inspection?.id&&j.kind==='reconstruction'),active=recJobs.find(running);
 useEffect(()=>{request<ReconInfo>('/api/v1/reconstruction').then(setInfo).catch(()=>setInfo(null))},[]);
 useEffect(()=>{if(!inspection)return;let live=true;
  Promise.all([request<Model3D[]>(`/api/v1/inspections/${inspection.id}/models`),request<Media[]>(`/api/v1/inspections/${inspection.id}/media`)]).then(([m,me])=>{if(live){setModels(m);setMedia(me)}}).catch(e=>live&&setError(String(e)));return()=>{live=false}},
  [inspection?.id,recJobs.map(j=>j.id+j.status).join()]);
 useEffect(()=>{const h=()=>setFull(!!document.fullscreenElement);document.addEventListener('fullscreenchange',h);return()=>document.removeEventListener('fullscreenchange',h)},[]);
 const done=models.filter(m=>m.status==='completed'),model=models.find(m=>m.id===selected&&m.status==='completed')||done[0],q=model?.quality||{};
 const photos=media.filter(m=>m.parent_media_id==null&&m.content_type.startsWith('image/')).length,videos=media.filter(m=>m.content_type.startsWith('video/')).length;
 async function densify(m:Model3D){setError('');try{await request(`/api/v1/models/${m.id}/densify`,{method:'POST'});onChanged()}catch(e){setError(String(e))}}
 async function build(){if(!inspection)return;setError('');try{await request(`/api/v1/inspections/${inspection.id}/reconstruct?dense=${dense}`,{method:'POST'});onChanged()}catch(e){setError(String(e))}}
 const ready=info?.status==='ready',enough=photos>=3||videos>0;
 const placed=q.registered_images!=null&&q.images_total?q.registered_images/q.images_total:null;
 return <div className="viewer-shell recon-shell" ref={shell}>
  <div className="viewer-left"><h3>Inspection</h3>
   <select className="det-select" aria-label="Inspection" value={inspection?.id} onChange={e=>{setInspectionId(+e.target.value);setSelected(null)}}>{inspections.map(i=><option key={i.id} value={i.id}>{insCode(i.id)} · {i.label}</option>)}</select>
   <div className="recon-inputs"><span><ImageIcon size={14}/>{photos} photo{photos===1?'':'s'}</span><span><Film size={14}/>{videos} video{videos===1?'':'s'}</span></div>
   {info&&!ready&&<div className="notice"><AlertTriangle size={16}/><span>{info.status==='missing'?<>COLMAP_PATH in <b>backend/.env</b> points to <code>{info.colmap}</code>, which doesn't exist. Fix the path (e.g. <code>D:/tools/colmap/bin/colmap.exe</code>) and restart the backend.</>
    :<>COLMAP wasn't found. Unzip the COLMAP Windows CUDA build to <code>D:	ools\colmap</code> (found automatically) or set <b>COLMAP_PATH</b> in <b>backend/.env</b> (not frontend/.env), then restart the backend.</>}</span></div>}
   {!info&&<div className="notice"><AlertTriangle size={16}/><span>Can't reach the backend to check COLMAP. Is it running?</span></div>}
   {ready&&<small className="muted recon-colmap" title={info?.colmap||''}>COLMAP: {info?.colmap}</small>}
   <h4>Quality</h4>
   <label className={'recon-opt'+(!dense?' on':'')}><input type="radio" checked={!dense} onChange={()=>setDense(false)}/><span><b>Sparse</b><small>Camera positions and a point cloud. Minutes.</small></span></label>
   <label className={'recon-opt'+(dense?' on':'')}><input type="radio" checked={dense} onChange={()=>setDense(true)}/><span><b>Dense + mesh</b><small>Detailed points and a surface mesh on the GPU. Can take 10–60 min.</small></span></label>
   <button className="btn primary wide" onClick={build} disabled={!ready||!enough||!!active}><Play size={15}/>{active?`Building… ${active.progress}%`:'Build 3D model'}</button>
   {!enough&&<small className="muted">Needs at least 3 overlapping photos or a video. A drone orbit around the tower works best.</small>}
   {active&&<div className="det-job"><span className="badge neutral">{active.stage.replaceAll('_',' ')}</span><small>{active.detail}</small><div className="qc-bar"><div style={{width:`${active.progress}%`}}/></div></div>}
   {error&&<div className="det-job"><small className="det-error">{error}</small></div>}
   <h4>Models ({models.length})</h4>
   {models.map(m=><button key={m.id} className={'tree-asset '+(model?.id===m.id?'chosen':'')} disabled={m.status!=='completed'} onClick={()=>setSelected(m.id)} title={m.quality.error||''}>
    {m.status==='completed'?<CheckCircle2 size={15} className="ok"/>:m.status==='failed'?<AlertTriangle size={15}/>:<Box size={15}/>}
    <div><b>#{m.id} · {m.dense?'dense':'sparse'}{m.quality.source_model?` (from #${m.quality.source_model})`:''}</b><small>{m.status==='completed'?`${m.quality.registered_images}/${m.quality.images_total} images · ${fmt(m.dense?m.quality.dense_points:m.quality.sparse_points)} pts`:m.status==='failed'?(m.quality.error||'failed').slice(0,60):m.status}</small></div></button>)}
   {!models.length&&<p className="muted">No 3D models yet for this inspection.</p>}
  </div>
  <div className="viewer-middle"><div className="viewer-toolbar ann-toolbar">
    <div className="ann-tools ann-title"><b>{model?`Model #${model.id} · ${insCode(model.inspection_id)}`:'3D model'}</b></div>
    <div className="ann-tools">
     <button className={mode==='points'?'on':''} onClick={()=>setMode('points')} title="Point cloud"><Layers size={15}/>Points</button>
     <button className={mode==='mesh'?'on':''} onClick={()=>setMode('mesh')} disabled={!model?.urls.web_mesh} title={model?.urls.web_mesh?'Surface mesh':'Mesh needs a dense build'}><Box size={15}/>Mesh</button><i className="ann-sep"/>
     <label className="recon-size" title="Point size">size<input type="range" min={.005} max={.12} step={.005} value={pointSize} onChange={e=>setPointSize(+e.target.value)}/></label>
     <button className={showCams?'on':''} onClick={()=>setShowCams(!showCams)} title="Drone camera positions"><Camera size={15}/></button>
     <button className={full?'on':''} title="Full screen" onClick={()=>document.fullscreenElement?document.exitFullscreen():shell.current?.requestFullscreen()}>{full?<Minimize size={15}/>:<Maximize size={15}/>}</button></div></div>
   <div className="recon-stage">{model?<ModelView key={model.id} model={model} mode={mode} pointSize={pointSize} showCams={showCams}/>
    :<div className="det-empty"><Box size={46}/><p>{active?'Building the model… it appears here when done.':'Build a 3D model from this inspection’s photos or video.'}</p></div>}
    {model&&<div className="ann-hint">Drag to orbit · right-drag to pan · wheel to zoom · camera button = show drone positions</div>}</div></div>
  <div className="viewer-right"><div className="panel-head"><h2>Model quality</h2>{model&&<span className={'badge '+(placed!=null&&placed>=.8?'good':placed!=null&&placed>=.5?'fair':'poor')}>{placed!=null?`${Math.round(placed*100)}% placed`:'—'}</span>}</div>
   {model?<>
    <div className="info"><small>Images placed</small><b>{q.registered_images} / {q.images_total}</b></div>
    <div className="info"><small>Sparse points</small><b>{fmt(q.sparse_points)}</b></div>
    {model.dense&&<><div className="info"><small>Dense points</small><b>{fmt(q.dense_points)}</b></div><div className="info"><small>Mesh faces</small><b>{fmt(q.mesh_faces)}</b></div></>}
    <div className="info"><small>Reprojection error</small><b>{q.mean_reprojection_error_px!=null?`${q.mean_reprojection_error_px.toFixed(2)} px`:'—'}</b></div>
    <div className="info"><small>Mean track length</small><b>{q.mean_track_length!=null?q.mean_track_length.toFixed(1):'—'}</b></div>
    <div className="info"><small>Build time</small><b>{q.seconds!=null?`${Math.round(q.seconds/60)} min ${q.seconds%60}s`:'—'}</b></div>
    {(q.models_found||1)>1&&<div className="notice"><AlertTriangle size={16}/>The images formed {q.models_found} separate groups; the largest is shown. Add images that bridge the gaps.</div>}
    {!model.dense&&<div className="recon-densify"><button className="btn primary" disabled={!ready||!!active} onClick={()=>densify(model)} title="Reuses this model's camera positions, so it skips feature matching"><Layers size={14}/>Make dense model from this</button>
     <small className="muted">{active?'Another build is running.':'Detailed points and a surface mesh from this sparse model. Roughly 15–40 min for 100 images.'}</small></div>}
    <h4 className="det-sub">Downloads</h4>
    <div className="recon-downloads">{[['dense_ply','Dense point cloud (PLY)'],['mesh_ply','Mesh (PLY)'],['sparse_ply','Sparse point cloud (PLY)'],['log','COLMAP log']].filter(([k])=>model.urls[k]).map(([k,l])=><a key={k} className="text-btn" href={API+model.urls[k]} download><Download size={13}/>{l}</a>)}</div>
    <div className="notice"><AlertTriangle size={16}/>Built from photos alone, so the model has no real-world scale or GPS position yet. Error under ~1 px and most images placed means a reliable model.</div>
   </>:<p className="muted">Quality figures appear once a model is built. Tips: fly an orbit around the tower at 2–3 heights with 70–80% overlap between shots; avoid motion blur.</p>}
   {models.find(m=>m.status==='failed'&&m.id===models[0]?.id)&&<div className="notice"><AlertTriangle size={16}/><span>Latest build failed: {models[0].quality.error} {models[0].urls.log&&<a className="text-btn" href={API+models[0].urls.log}>log</a>}</span></div>}
  </div></div>;
}
