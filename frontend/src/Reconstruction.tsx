import {useEffect,useMemo,useRef,useState} from 'react';
import {Canvas,ThreeEvent,useFrame,useThree} from '@react-three/fiber';import {OrbitControls,Grid,TransformControls} from '@react-three/drei';import {LabelAnchor} from './Label3D';
import PointCloudPanel from './PointCloudPanel';
import * as THREE from 'three';import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
import {Box,Play,AlertTriangle,Download,Camera,Image as ImageIcon,Film,CheckCircle2,Layers,Maximize,Minimize,X} from 'lucide-react';
import {API,request,Inspection,Job,Media,RegistrationCandidate,getAuthToken} from './api';

// 3D models built by COLMAP photogrammetry (backend app/reconstruct.py)
type ReconInfo={status:string;colmap:string|null;colmap_path_env:string|null;video_stride:number;max_images:number;dense_max_size:number};
export type Model3D={id:number;inspection_id:number;job_id:number;status:string;dense:boolean;created_at:string|null;urls:Record<string,string>;
 quality:{registered_images?:number;images_total?:number;sparse_points?:number;dense_points?:number;mesh_faces?:number;mean_reprojection_error_px?:number|null;mean_track_length?:number|null;models_found?:number;seconds?:number;error?:string;source_model?:number;scale?:ModelScale|null;
  source?:'pointcloud'|string;pointcloud?:{files:string[];points_total:number;points_used:number;crs:string|null;units:string;cropped_to_flight:boolean;alignment?:{gps_residual_m:number;vertical_shift_m:number;icp_rms_m:number|null;inliers_pct:number}};boxes_available?:boolean;boxes_note?:string;warning?:string}};
/** Metres per model unit (app/scale.py): from the drone's GPS / RTK positions, or calibrated by a reviewer. */
export type ModelScale={m_per_unit:number;method:'gps_rtk'|'gps'|'manual'|string;cameras?:number;rtk?:boolean;residual_median_m?:number;uncertainty_pct?:number|null;reference_m?:number;note?:string};
type Cam={name:string;center:[number,number,number];dir:[number,number,number]};
export type MeasurementMode='distance'|'area';
export type MeasurementPoint=[number,number,number];
const insCode=(id:number)=>'INS-'+String(id).padStart(4,'0');
const fmt=(n?:number|null)=>n==null?'—':n.toLocaleString();
const running=(j:Job)=>['queued','running'].includes(j.status);

/** 3D files need the signed-in user too: send the token header and, across ports in local dev, the session cookie. */
function authLoader<L extends THREE.Loader>(l:L):L{l.setWithCredentials(true);const t=getAuthToken();if(t)l.setRequestHeader({Authorization:`Bearer ${t}`});return l}
function usePly(url?:string){
 const [geo,setGeo]=useState<THREE.BufferGeometry|null>(null),[err,setErr]=useState('');
 useEffect(()=>{setGeo(null);setErr('');if(!url)return;let live=true;authLoader(new PLYLoader()).load(API+url,g=>{if(live)setGeo(g)},undefined,e=>live&&setErr(String((e as Error)?.message||e)));return()=>{live=false}},[url]);
 return {geo,err};
}
function Cameras({url,selected,onPick}:{url?:string;selected?:string|null;onPick:(cam:Cam)=>void}){
 const [cams,setCams]=useState<Cam[]>([]);
 const hits=useRef<THREE.InstancedMesh>(null!);
 useEffect(()=>{if(!url)return;const t=getAuthToken();fetch(API+url,{credentials:'include',headers:t?{Authorization:`Bearer ${t}`}:{}}).then(r=>r.json()).then(setCams).catch(()=>setCams([]))},[url]);
 useEffect(()=>{if(!hits.current)return;const matrix=new THREE.Matrix4();cams.forEach((c,i)=>{matrix.makeTranslation(...c.center);hits.current.setMatrixAt(i,matrix)});hits.current.instanceMatrix.needsUpdate=true},[cams]);
 // dots stay a few pixels on screen at any zoom; direction ticks scale with the spacing between shots so they never pile up
 const objs=useMemo(()=>{if(!cams.length)return null;const P=cams.map(c=>new THREE.Vector3(...c.center));
  const gaps=P.slice(1).map((p,i)=>p.distanceTo(P[i])).filter(g=>g>0).sort((a,b)=>a-b),gap=gaps.length?gaps[Math.floor(gaps.length/2)]:.5,len=Math.min(Math.max(gap*.8,.05),.6);
  const dots=new THREE.Points(new THREE.BufferGeometry().setFromPoints(P),new THREE.PointsMaterial({color:'#ffb020',size:5,sizeAttenuation:false}));
  const ticks=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(cams.flatMap((c,i)=>[P[i],P[i].clone().add(new THREE.Vector3(...c.dir).normalize().multiplyScalar(len))])),
   new THREE.LineBasicMaterial({color:'#ffb020',transparent:true,opacity:.55}));
  const path=new THREE.Line(new THREE.BufferGeometry().setFromPoints(P),new THREE.LineBasicMaterial({color:'#ffb020',transparent:true,opacity:.3}));
  return [dots,ticks,path]},[cams]);
 const chosen=cams.find(c=>c.name===selected);
 return objs&&<group>{objs.map((o,i)=><primitive key={i} object={o}/>)}
  <instancedMesh ref={hits} args={[undefined,undefined,cams.length]} onClick={(e:ThreeEvent<MouseEvent>)=>{e.stopPropagation();if(e.instanceId!=null)onPick(cams[e.instanceId])}} onPointerOver={e=>{e.stopPropagation();document.body.style.cursor='pointer'}} onPointerOut={()=>{document.body.style.cursor='auto'}}>
   <sphereGeometry args={[.025,8,6]}/><meshBasicMaterial transparent opacity={0} depthWrite={false}/>
  </instancedMesh>
  {chosen&&<mesh position={chosen.center} renderOrder={25}><sphereGeometry args={[.022,12,8]}/><meshBasicMaterial color="#fff4a6" depthTest={false}/></mesh>}
 </group>;
}
// stable colour per equipment class (same palette as the 2D box editor)
const CLASS_PALETTE=['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000','#aaffc3'];
export const classColor=(name:string)=>CLASS_PALETTE[[...name].reduce((h,ch)=>(h*31+ch.charCodeAt(0))>>>0,7)%CLASS_PALETTE.length];
const MIN_DRAWN=.012;  // bolts and clamps are a few millimetres in model units: draw them at least this big (the panel shows the true size)
function EquipmentBox({m,on}:{m:RegistrationCandidate;on:boolean}){
 const s=(v:number|null|undefined)=>Math.max(v||0,MIN_DRAWN);
 const c=classColor(m.class_name),geo=useMemo(()=>new THREE.BoxGeometry(s(m.size_w),s(m.size_h),s(m.size_d)),[m.size_w,m.size_h,m.size_d]),edges=useMemo(()=>new THREE.EdgesGeometry(geo),[geo]);
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
function MeasurementLayer({mode,points}:{mode:MeasurementMode;points:MeasurementPoint[]}){
 const path=useMemo(()=>{const pairs:THREE.Vector3[]=[];for(let i=1;i<points.length;i++)pairs.push(new THREE.Vector3(...points[i-1]),new THREE.Vector3(...points[i]));if(mode==='area'&&points.length===4)pairs.push(new THREE.Vector3(...points[3]),new THREE.Vector3(...points[0]));return new THREE.BufferGeometry().setFromPoints(pairs)},[mode,points]);
 const fill=useMemo(()=>{const g=new THREE.BufferGeometry();if(mode==='area'&&points.length===4){const v=[points[0],points[1],points[2],points[0],points[2],points[3]].flat();g.setAttribute('position',new THREE.Float32BufferAttribute(v,3));g.computeVertexNormals()}return g},[mode,points]);
 useEffect(()=>()=>{path.dispose();fill.dispose()},[path,fill]);
 return <group renderOrder={40}>
  {mode==='area'&&points.length===4&&<mesh geometry={fill} renderOrder={38} raycast={()=>{}}><meshBasicMaterial color="#42c8ff" transparent opacity={.2} side={THREE.DoubleSide} depthTest={false} depthWrite={false}/></mesh>}
  {points.length>1&&<lineSegments geometry={path} raycast={()=>{}} renderOrder={40}><lineBasicMaterial color="#72dcff" depthTest={false}/></lineSegments>}
  {points.map((point,index)=><ScreenDot key={index} position={point} color={index===0?'#fff257':'#42c8ff'}/>)}
 </group>;
}
/** A measurement point drawn at a fixed size on screen (radius in px) at any zoom. A sphere with a size in model units
 * was ~0.8 m wide on a scaled tower and hid the spot that was picked. */
function ScreenDot({position,color,px=4}:{position:MeasurementPoint;color:string;px?:number}){
 const g=useRef<THREE.Group>(null!),{camera,size}=useThree(),v=useRef(new THREE.Vector3());
 useFrame(()=>{if(!g.current)return;const fov=THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov||45);
  const perPx=v.current.set(...position).distanceTo(camera.position)*2*Math.tan(fov/2)/size.height;g.current.scale.setScalar(perPx)});
 return <group ref={g} position={position} renderOrder={41}>
  <mesh renderOrder={41} raycast={()=>{}}><sphereGeometry args={[px+1.5,16,10]}/><meshBasicMaterial color="#ffffff" depthTest={false}/></mesh>
  <mesh renderOrder={42} raycast={()=>{}}><sphereGeometry args={[px,16,10]}/><meshBasicMaterial color={color} depthTest={false}/></mesh>
 </group>;
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
/** Picking measurement points without hit-testing on every mouse move. Event handlers on the cloud made the 3D
 * library ray-test all of its points (1.5M on a dense model, with a 1-unit default tolerance) on each pointer move and
 * froze the page. Here only a click (not an orbit drag) is turned into a ray, once: for a point cloud the point nearest
 * that ray within a few pixels (front-most first), for a mesh the first triangle hit (else its nearest vertex). */
const PICK_PX=6;
function nearestPoint(points:THREE.Points|THREE.Mesh,ray:THREE.Ray,camera:THREE.Camera,heightPx:number):THREE.Vector3|null{
 const pos=points.geometry.getAttribute('position') as THREE.BufferAttribute,a=pos.array as ArrayLike<number>;
 const inv=new THREE.Matrix4().copy(points.matrixWorld).invert(),local=ray.clone().applyMatrix4(inv);
 const o=local.origin,d=local.direction.normalize(),ox=o.x,oy=o.y,oz=o.z,dx=d.x,dy=d.y,dz=d.z;
 const fov=THREE.MathUtils.degToRad((camera as THREE.PerspectiveCamera).fov||45),perPx=2*Math.tan(fov/2)/heightPx*PICK_PX;  // tolerance per unit distance
 // 1) every point within the tolerance of the click ray, 2) of those on the front-most surface (within a thin depth
 // band behind the nearest one) the one closest to the clicked pixel - not just the front-most, which can be pixels off
 const hits:number[]=[];let front=Infinity;
 for(let i=0,n=pos.count*3;i<n;i+=3){const vx=a[i]-ox,vy=a[i+1]-oy,vz=a[i+2]-oz,t=vx*dx+vy*dy+vz*dz;
  if(t<=0)continue;const tol=t*perPx;if(vx*vx+vy*vy+vz*vz-t*t<tol*tol){hits.push(i);if(t<front)front=t}}
 if(!hits.length)return null;
 const band=Math.max(front*perPx*2,front*.004);let best=-1,bestPx=Infinity;
 for(const i of hits){const vx=a[i]-ox,vy=a[i+1]-oy,vz=a[i+2]-oz,t=vx*dx+vy*dy+vz*dz;if(t>front+band)continue;
  const off=Math.sqrt(Math.max(0,vx*vx+vy*vy+vz*vz-t*t))/t;if(off<bestPx){bestPx=off;best=i}}  // angular offset from the ray ~ pixels
 return new THREE.Vector3(a[best],a[best+1],a[best+2]).applyMatrix4(points.matrixWorld)}
function MeasurePicker({object,onPick}:{object:THREE.Object3D;onPick:(point:MeasurementPoint)=>void}){
 const {gl,camera}=useThree(),pick=useRef(onPick);pick.current=onPick;
 useEffect(()=>{const el=gl.domElement,prev=el.style.cursor;let down:{x:number;y:number}|null=null;el.style.cursor='crosshair';
  const onDown=(e:PointerEvent)=>{down=e.button===0?{x:e.clientX,y:e.clientY}:null};
  const onUp=(e:PointerEvent)=>{const start=down;down=null;if(!start||e.button!==0||Math.hypot(e.clientX-start.x,e.clientY-start.y)>5)return;  // a drag orbits the view
   const r=el.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);
   const hit=object instanceof THREE.Points?nearestPoint(object,ray.ray,camera,r.height)
    :ray.intersectObject(object,false)[0]?.point??(object instanceof THREE.Mesh?nearestPoint(object,ray.ray,camera,r.height):null);  // the web mesh has gaps: fall back to its nearest vertex
   if(hit)pick.current([hit.x,hit.y,hit.z])};
  el.addEventListener('pointerdown',onDown);el.addEventListener('pointerup',onUp);
  return()=>{el.removeEventListener('pointerdown',onDown);el.removeEventListener('pointerup',onUp);el.style.cursor=prev}},[gl,camera,object]);
 return null}
export function ModelView({model,mode,pointSize,showCams,markers=[],selectedMarker,editingMarker,markerEditMode='translate',focusRequest=0,onMarkerPick,onMarkerTransform,measurementMode,measurementPoints=[],onMeasurementPoint}:{model:Model3D;mode:'points'|'mesh';pointSize:number;showCams:boolean;markers?:RegistrationCandidate[];selectedMarker?:number|null;editingMarker?:number|null;markerEditMode?:MarkerEditMode;focusRequest?:number;onMarkerPick?:(id:number)=>void;onMarkerTransform?:(id:number,position:[number,number,number],yaw:number)=>void;measurementMode?:MeasurementMode|null;measurementPoints?:MeasurementPoint[];onMeasurementPoint?:(point:MeasurementPoint)=>void}){
 const useMesh=mode==='mesh'&&!!model.urls.web_mesh;const {geo,err}=usePly(useMesh?model.urls.web_mesh:model.urls.web_points);
 const [selectedCam,setSelectedCam]=useState<Cam|null>(null);
 const obj=useMemo(()=>{if(!geo)return null;const colors=!!geo.getAttribute('color');
  if(useMesh){geo.computeVertexNormals();return new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:colors,color:colors?'#ffffff':'#c8d3de',side:THREE.DoubleSide,roughness:.9}))}
  return new THREE.Points(geo,new THREE.PointsMaterial({size:pointSize,vertexColors:colors,color:colors?'#ffffff':'#9fd0ff',sizeAttenuation:true}))},[geo,useMesh]);
 useEffect(()=>{if(obj instanceof THREE.Points)(obj.material as THREE.PointsMaterial).size=pointSize},[obj,pointSize]);
 useEffect(()=>{if(!showCams)setSelectedCam(null)},[showCams]);
 const label=useRef<HTMLDivElement>(null),sel=markers.find(m=>m.id===selectedMarker);// selected box label, positioned by LabelAnchor
 const cameraRoot=model.urls.web_cameras?.replace(/\/web\/cameras\.json(?:\?.*)?$/,'')||'',cameraImage=selectedCam?`${API}${cameraRoot}/images/${selectedCam.name.split('/').map(encodeURIComponent).join('/')}`:'';
 return <><div ref={label} className="three-label floating-label">{sel&&<><strong>{sel.class_name.replaceAll('_',' ')}</strong><br/><small>Estimated · {sel.media_count} images · {Math.round(sel.confidence*100)}%</small></>}</div>{err&&<div className="recon-msg">Could not load the model: {err}</div>}{!geo&&!err&&<div className="recon-msg">Loading 3D model…</div>}
  {selectedCam&&<div className="camera-preview"><div className="camera-preview-head"><span><Camera size={14}/><b>Drone capture</b></span><button type="button" onClick={()=>setSelectedCam(null)} title="Close image preview"><X size={15}/></button></div><img src={cameraImage} alt={`Drone capture ${selectedCam.name}`}/><div className="camera-preview-meta"><span>{selectedCam.name.split('/').pop()}</span><small>Captured from the selected yellow camera point</small></div></div>}
  <Canvas camera={{position:[16,12,16],fov:45,near:.05,far:500}} gl={{antialias:true}}>{/* high enough to show the tower and the drone path above it */}
   <color attach="background" args={['#0f1f31']}/><ambientLight intensity={.9}/><directionalLight position={[8,12,6]} intensity={1.2}/>
   {obj&&<primitive object={obj}/>}{obj&&measurementMode&&onMeasurementPoint&&<MeasurePicker object={obj} onPick={onMeasurementPoint}/>} {showCams&&<Cameras url={model.urls.web_cameras} selected={selectedCam?.name} onPick={setSelectedCam}/>} {!measurementMode&&<RegistrationMarkers markers={markers} selected={selectedMarker} editing={editingMarker} editMode={markerEditMode} onPick={onMarkerPick} onTransform={onMarkerTransform}/>} {measurementMode&&<MeasurementLayer mode={measurementMode} points={measurementPoints}/>}<LabelAnchor target={label} position={sel?[sel.x,sel.z+(sel.size_h??.2)/2+.05,sel.y]:null}/>
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
    :<>COLMAP wasn't found. Unzip the COLMAP Windows CUDA build to <code>{'D:\\tools\\colmap'}</code> (found automatically) or set <b>COLMAP_PATH</b> in <b>backend/.env</b> (not frontend/.env), then restart the backend.</>}</span></div>}
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
    <div><b>#{m.id} · {m.quality.source==='pointcloud'?'point cloud':m.dense?'dense':'sparse'}{m.quality.source_model?` (from #${m.quality.source_model})`:''}</b><small>{m.status==='completed'?(m.quality.source==='pointcloud'?`${fmt(m.quality.dense_points)} pts · ${m.quality.registered_images?`${m.quality.registered_images} photos aligned`:'no photos aligned'}`:`${m.quality.registered_images}/${m.quality.images_total} images · ${fmt(m.dense?m.quality.dense_points:m.quality.sparse_points)} pts`):m.status==='failed'?(m.quality.error||'failed').slice(0,60):m.status}</small></div></button>)}
   {!models.length&&<p className="muted">No 3D models yet for this inspection.</p>}
   <h4>Point cloud</h4>
   <PointCloudPanel compact inspectionId={inspection?.id??null} inspectionLabel={inspection?.label} onBuilt={onChanged}/>
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
    {model&&<div className="ann-hint">Drag to orbit · right-drag to pan · wheel to zoom · camera button = show positions · click a yellow point = source image</div>}</div></div>
  <div className="viewer-right"><div className="panel-head"><h2>Model quality</h2>{model&&(q.source==='pointcloud'&&!q.registered_images?<span className="badge neutral">Point cloud</span>:<span className={'badge '+(placed!=null&&placed>=.8?'good':placed!=null&&placed>=.5?'fair':'poor')}>{placed!=null?`${Math.round(placed*100)}% placed`:'—'}</span>)}</div>
   {model?<>
    <div className="info"><small>Images placed</small><b>{q.registered_images} / {q.images_total}</b></div>
    <div className="info"><small>Sparse points</small><b>{fmt(q.sparse_points)}</b></div>
    {model.dense&&<><div className="info"><small>Dense points</small><b>{fmt(q.dense_points)}</b></div><div className="info"><small>Mesh faces</small><b>{fmt(q.mesh_faces)}</b></div></>}
    <div className="info"><small>Reprojection error</small><b>{q.mean_reprojection_error_px!=null?`${q.mean_reprojection_error_px.toFixed(2)} px`:'—'}</b></div>
    <div className="info"><small>Mean track length</small><b>{q.mean_track_length!=null?q.mean_track_length.toFixed(1):'—'}</b></div>
    <div className="info"><small>Build time</small><b>{q.seconds!=null?`${Math.round(q.seconds/60)} min ${q.seconds%60}s`:'—'}</b></div>
    {(q.models_found||1)>1&&<div className="notice"><AlertTriangle size={16}/>The images formed {q.models_found} separate groups; the largest is shown. Add images that bridge the gaps.</div>}
    {q.source==='pointcloud'&&q.pointcloud&&<><h4 className="det-sub">Point cloud</h4>
     <div className="info"><small>Points in file(s)</small><b>{fmt(q.pointcloud.points_total)}</b></div>
     <div className="info"><small>Used{q.pointcloud.cropped_to_flight?' (around the flight)':''}</small><b>{fmt(q.pointcloud.points_used)}</b></div>
     <div className="info"><small>Coordinate system</small><b title={q.pointcloud.crs||''}>{q.pointcloud.crs?q.pointcloud.crs.slice(0,34):'none'}</b></div>
     {q.pointcloud.alignment&&<><div className="info"><small>Photos to cloud fit (ICP)</small><b>{q.pointcloud.alignment.icp_rms_m!=null?`${(q.pointcloud.alignment.icp_rms_m*100).toFixed(1)} cm`:'—'} · {q.pointcloud.alignment.inliers_pct}% on surface</b></div>
      <div className="info"><small>Height datum shift</small><b>{q.pointcloud.alignment.vertical_shift_m.toFixed(2)} m</b></div></>}
     {q.warning&&<div className="notice"><AlertTriangle size={16}/>{q.warning}</div>}
     {q.boxes_note&&<div className="notice"><AlertTriangle size={16}/>{q.boxes_note}</div>}</>}
    {!model.dense&&<div className="recon-densify"><button className="btn primary" disabled={!ready||!!active} onClick={()=>densify(model)} title="Reuses this model's camera positions, so it skips feature matching"><Layers size={14}/>Make dense model from this</button>
     <small className="muted">{active?'Another build is running.':'Detailed points and a surface mesh from this sparse model. Roughly 15–40 min for 100 images.'}</small></div>}
    <h4 className="det-sub">Downloads</h4>
    <div className="recon-downloads">{[['dense_ply','Dense point cloud (PLY)'],['mesh_ply','Mesh (PLY)'],['sparse_ply','Sparse point cloud (PLY)'],['log','COLMAP log']].filter(([k])=>model.urls[k]).map(([k,l])=><a key={k} className="text-btn" href={API+model.urls[k]} download><Download size={13}/>{l}</a>)}</div>
    {q.source!=='pointcloud'&&<div className="notice"><AlertTriangle size={16}/>Built from photos{q.scale?`, scaled from the drone's GPS (1 unit = ${q.scale.m_per_unit.toFixed(3)} m)`:' alone, so it has no real-world scale yet'}. Error under ~1 px and most images placed means a reliable model.</div>}
   </>:<p className="muted">Quality figures appear once a model is built. Tips: fly an orbit around the tower at 2–3 heights with 70–80% overlap between shots; avoid motion blur.</p>}
   {models.find(m=>m.status==='failed'&&m.id===models[0]?.id)&&<div className="notice"><AlertTriangle size={16}/><span>Latest build failed: {models[0].quality.error} {models[0].urls.log&&<a className="text-btn" href={API+models[0].urls.log}>log</a>}</span></div>}
  </div></div>;
}
