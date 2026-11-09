import {useEffect,useRef,useState} from 'react';
import {API,request} from './api';
import type {RegistrationCandidate} from './api';
import ViewToolbar,{ColourMode,ViewMode} from './ViewToolbar';

/** Every point of the uploaded LiDAR, streamed by level of detail with Potree from the COPC copy the GPU worker builds
 * (the way the PLA QC app shows its surveys), with the 3D equipment boxes drawn in the survey's own coordinates. */
type Cloud={id:number;filename:string;role:string;copc_url:string|null;copc_status:string|null;copc_points:number|null;copc_error:string|null};
type View={position:[number,number,number];target:[number,number,number]};
type Win=Window&{Potree?:any};
const BASE='/potree';
const SCRIPTS=['libs/jquery/jquery-3.1.1.min.js','libs/other/BinaryHeap.js','libs/tween/tween.min.js','libs/proj4/proj4.js','libs/copc/index.js','build/potree/potree.js'];
let runtime:Promise<any>|null=null;
function loadPotree():Promise<any>{  // Potree 1.8 is a set of classic scripts; it brings its own three.js, used for the boxes too
 if(runtime)return runtime;
 runtime=(async()=>{
  if(!document.querySelector('link[data-potree]')){const l=document.createElement('link');l.rel='stylesheet';l.href=`${BASE}/build/potree/potree.css`;l.dataset.potree='1';document.head.appendChild(l)}
  for(const src of SCRIPTS)await new Promise<void>((ok,fail)=>{const s=document.createElement('script');s.src=`${BASE}/${src}`;s.onload=()=>ok();s.onerror=()=>fail(Error(`Could not load ${src}`));document.head.appendChild(s)});
  const code=await (await fetch(`${BASE}/libs/three.js/build/three.module.js`)).text();  // Potree's own three.js version, as a module
  return import(/* @vite-ignore */ URL.createObjectURL(new Blob([code],{type:'text/javascript'})));
 })();
 runtime.catch(()=>{runtime=null});
 return runtime;
}
export async function copcClouds(inspectionId:number){return (await request<Cloud[]>(`/api/v1/inspections/${inspectionId}/pointclouds`)).filter(c=>c.role==='cloud')}

/** viewer units (the preview's frame) -> local metres of the survey (add `origin` for its CRS) */
function toLocal(M:number[][],v:[number,number,number]){
 const s=Math.hypot(M[0][0],M[1][0],M[2][0]),d=[v[0]-M[0][3],v[1]-M[1][3],v[2]-M[2][3]];
 return [0,1,2].map(j=>(M[0][j]*d[0]+M[1][j]*d[1]+M[2][j]*d[2])/(s*s)) as [number,number,number];
}
export function corners(m:RegistrationCandidate):[number,number,number][]{
 const w=(m.size_w||.05)/2,h=(m.size_h||.05)/2,d=(m.size_d||.05)/2,c=Math.cos(m.yaw||0),s=Math.sin(m.yaw||0),out:[number,number,number][]=[];
 for(const a of [-w,w])for(const b of [-h,h])for(const e of [-d,d])out.push([m.x+a*c+e*s,m.z+b,m.y-a*s+e*c]);
 return out;  // same layout as the preview: centre (x, z, y), yaw about the vertical
}
const EDGES=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
// ---------- navigation (as in the PLA QC app): views, orbit / pan, fit, zoom to a spot, focus and isolate a box ----------
const VIEWS:Record<ViewMode,{yaw:number;pitch:number}>={top:{yaw:0,pitch:-Math.PI/2},front:{yaw:0,pitch:0},side:{yaw:Math.PI/2,pitch:0},'3d':{yaw:Math.PI/4,pitch:-Math.PI/4}};
const ZOOM_IN=.4,ZOOM_OUT=2.5;
function easeView(viewer:any,target:any,toRadius:number,ms=320){  // move the focus point onto target while the distance changes, view angle kept
 const view=viewer.scene.view;viewer.orbitControls.stop?.();
 const fromPivot=view.getPivot(),fromRadius=view.radius,dir=view.direction.clone(),start=performance.now();
 const step=(now:number)=>{const t=Math.min(1,(now-start)/ms),k=1-Math.pow(1-t,3),pivot=fromPivot.clone().lerp(target,k),r=fromRadius+(toRadius-fromRadius)*k;
  view.radius=r;view.position.copy(pivot).sub(dir.clone().multiplyScalar(r));if(t<1)requestAnimationFrame(step)};
 requestAnimationFrame(step);
}
/** The LiDAR point under the cursor, else the spot at the current focus distance (open space). */
function spotAt(st:{viewer:any;THREE:any},x:number,y:number){
 const Potree=(window as Win).Potree,v=st.viewer,camera=v.scene.getActiveCamera(),dom=v.renderer.domElement;
 const hit=Potree.Utils.getMousePointCloudIntersection({x,y},camera,v,v.scene.pointclouds);if(hit?.location)return hit.location;
 const ray=new st.THREE.Vector3(x/dom.clientWidth*2-1,1-y/dom.clientHeight*2,.5).unproject(camera).sub(camera.position).normalize();
 return camera.position.clone().add(ray.multiplyScalar(v.scene.view.radius));
}
function zoomAt(st:{viewer:any;THREE:any},x:number,y:number,factor:number){
 const v=st.viewer,target=spotAt(st,x,y);if(!target)return;
 const r=Math.max(1.5,Math.min(v.scene.view.radius*factor,20000));
 easeView(v,factor<1?target:v.scene.view.getPivot().clone().lerp(target,.35),r);
}
/** Elevation / intensity ranges from the points in memory (2nd-98th percentile), so a few stray points (birds, noise)
 * don't wash the colours out; every file shares one range. As in the PLA QC app. */
function sampleRanges(st:{viewer:any;THREE:any}){
 const z:number[]=[],inten:number[]=[],v=new st.THREE.Vector3();
 for(const pc of st.viewer.scene.pointclouds)for(const node of pc.visibleNodes||[]){const sn=node.sceneNode,g=sn?.geometry;if(!g)continue;
  const pos=g.attributes.position,it=g.attributes.intensity,n=pos?.count||0,step=Math.max(1,Math.floor(n/400));
  for(let i=0;i<n;i+=step){v.fromBufferAttribute(pos,i).applyMatrix4(sn.matrixWorld);z.push(v.z);if(it)inten.push(it.array[i])}}
 if(z.length<200)return null;
 const pct=(a:number[],p:number)=>a[Math.min(a.length-1,Math.max(0,Math.floor(p*(a.length-1))))];z.sort((a,b)=>a-b);inten.sort((a,b)=>a-b);
 return {elevation:[pct(z,.02),pct(z,.98)] as [number,number],intensity:inten.length>200?[pct(inten,.01),pct(inten,.98)] as [number,number]:null};
}
const POTREE_COLOUR:Record<ColourMode,string>={rgb:'rgba',elevation:'elevation',intensity:'intensity'};
function setView(viewer:any,mode:ViewMode){
 const Potree=(window as Win).Potree,view=viewer.scene.view,pivot=view.getPivot();
 viewer.setCameraMode(mode==='3d'?Potree.CameraMode.PERSPECTIVE:Potree.CameraMode.ORTHOGRAPHIC);
 view.yaw=mode==='3d'?view.yaw:VIEWS[mode].yaw;view.pitch=VIEWS[mode].pitch;view.position.copy(pivot).sub(view.direction.multiplyScalar(view.radius));viewer.orbitControls.stop?.();
}


export default function PotreeView({clouds,transform,origin,view,markers,selected,colorOf,onPick}:{clouds:Cloud[];transform:number[][];origin:number[];view?:View;markers:RegistrationCandidate[];selected?:number|null;colorOf:(cls:string)=>string;onPick?:(id:number)=>void}){
 const host=useRef<HTMLDivElement>(null),[err,setErr]=useState(''),[hover,setHover]=useState<{m:RegistrationCandidate;x:number;y:number}|null>(null),[loaded,setLoaded]=useState(0);
 const state=useRef<{viewer:any;THREE:any;boxes:any;raf:number;picks:{mesh:any;m:RegistrationCandidate}[]}|null>(null);
 const [mode,setMode]=useState<ViewMode>('3d'),[pan,setPan]=useState(false),[size,setSize]=useState(1),[isolated,setIsolated]=useState<number|null>(null);
 const clip=useRef<any>(null),[colour,setColour]=useState<ColourMode>('rgb');
 useEffect(()=>{const st=state.current;if(!st)return;  // colour mode; ranges follow the points as more of them stream in
  const apply=()=>{const r=colour==='rgb'?null:sampleRanges(st);for(const pc of st.viewer.scene.pointclouds){const m=pc.material;m.activeAttributeName=POTREE_COLOUR[colour];
   if(colour==='elevation'&&r)m.elevationRange=r.elevation;if(colour==='intensity'&&r?.intensity&&r.intensity[1]>r.intensity[0])m.intensityRange=r.intensity}};
  apply();if(colour==='rgb')return;const t=setInterval(apply,2000);return()=>clearInterval(t)},[colour,loaded]);
 const urls=clouds.map(c=>c.copc_url).filter(Boolean).join('|');
 useEffect(()=>{let live=true;const el=host.current;if(!el)return;
  loadPotree().then(THREE=>{if(!live||!el)return;const Potree=(window as Win).Potree;
   const viewer=new Potree.Viewer(el,{useDefaultRenderLoop:false});
   viewer.setEDLEnabled(true);viewer.setFOV(60);viewer.setPointBudget(3_000_000);viewer.setBackground('gradient');viewer.setMinNodeSize(30);
   const boxes=new THREE.Group();boxes.position.set(origin[0],origin[1],origin[2]);const overlay=new THREE.Scene();overlay.add(boxes);
   // drawn after Potree has composited the points (eye-dome lighting): in its scene they would sit under the points
   viewer.addEventListener('render.pass.perspective_overlay',(e:any)=>e.viewer.renderer.render(overlay,e.viewer.scene.getActiveCamera()));
   const st={viewer,THREE,boxes,raf:0,picks:[] as {mesh:any;m:RegistrationCandidate}[]};state.current=st;(window as Win&{__potreeView?:unknown}).__potreeView=st;  // for diagnostics in the browser console
   const tick=(t:number)=>{if(!live)return;viewer.loop(t);st.raf=requestAnimationFrame(tick)};st.raf=requestAnimationFrame(tick);
   const dom=viewer.renderer.domElement,orbit=viewer.orbitControls;
   orbit.addEventListener('drag',(e:any)=>{if(e.drag.object!==null||e.drag.mouse!==4)return;  // middle-drag pans in orbit mode too
    orbit.panDelta.x+=e.drag.lastDrag.x/dom.clientWidth;orbit.panDelta.y+=e.drag.lastDrag.y/dom.clientHeight});
   let rdown:{x:number;y:number}|null=null,rlast:{t:number;x:number;y:number}|null=null;  // right double-click zooms in (Shift: out); right-drag still pans
   dom.addEventListener('mousedown',(e:MouseEvent)=>{rdown=e.button===2?{x:e.clientX,y:e.clientY}:null});
   dom.addEventListener('mouseup',(e:MouseEvent)=>{if(e.button!==2||!rdown)return;const moved=Math.hypot(e.clientX-rdown.x,e.clientY-rdown.y);rdown=null;if(moved>5){rlast=null;return}
    const now=performance.now();if(rlast&&now-rlast.t<400&&Math.hypot(e.clientX-rlast.x,e.clientY-rlast.y)<12){rlast=null;const r=dom.getBoundingClientRect();zoomAt(st,e.clientX-r.left,e.clientY-r.top,e.shiftKey?ZOOM_OUT:ZOOM_IN);return}
    rlast={t:now,x:e.clientX,y:e.clientY}});
   dom.addEventListener('contextmenu',(e:Event)=>e.preventDefault());
   let first=true;
   for(const url of urls.split('|').filter(Boolean))Potree.loadPointCloud(API+url,'LiDAR',(e:any)=>{if(!live)return;
    const pc=e.pointcloud,m=pc.material;m.size=size;m.pointSizeType=Potree.PointSizeType.ADAPTIVE;m.activeAttributeName='rgba';viewer.scene.addPointCloud(pc);setLoaded(n=>n+1);
    if(first){first=false;
     if(view){const p=toLocal(transform,view.position),q=toLocal(transform,view.target);
      viewer.scene.view.position.set(origin[0]+p[0],origin[1]+p[1],origin[2]+p[2]);viewer.scene.view.lookAt(new THREE.Vector3(origin[0]+q[0],origin[1]+q[1],origin[2]+q[2]))}
     else viewer.fitToScreen()}});
  }).catch(e=>live&&setErr(String((e as Error).message||e)));
  return()=>{live=false;const st=state.current;state.current=null;if(st){cancelAnimationFrame(st.raf);try{st.viewer.renderer.dispose();st.viewer.renderer.forceContextLoss()}catch{/* already gone */}}
   if(el)el.innerHTML=''};
 },[urls]);
 // the 3D boxes, in the survey's coordinates (relative to `origin`, so float precision holds)
 useEffect(()=>{const st=state.current;if(!st)return;const {THREE,boxes}=st;
  while(boxes.children.length){const c=boxes.children.pop();c.geometry?.dispose?.();c.material?.dispose?.()}st.picks=[];
  for(const m of markers.filter(x=>x.status!=='rejected')){
   const P=corners(m).map(v=>toLocal(transform,v)),on=m.id===selected;
   const line=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(EDGES.flatMap(([a,b])=>[...P[a],...P[b]]),3)),
    new THREE.LineBasicMaterial({color:on?'#ffffff':colorOf(m.class_name),depthTest:false}));
   line.renderOrder=on?11:10;boxes.add(line);
   const g=new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute([[0,1,3,0,3,2],[4,6,7,4,7,5],[0,4,5,0,5,1],[2,3,7,2,7,6],[0,2,6,0,6,4],[1,5,7,1,7,3]].flat().flatMap(i=>P[i]),3));
   const mesh=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:colorOf(m.class_name),transparent:true,opacity:on?.4:.2,depthTest:false,depthWrite:false,side:THREE.DoubleSide}));  // WebGL lines are 1 px: the faces make boxes visible
   mesh.renderOrder=9;boxes.add(mesh);st.picks.push({mesh,m});
  }
 },[markers,selected,loaded,transform,origin]);
 useEffect(()=>{const st=state.current;if(st)for(const pc of st.viewer.scene.pointclouds)pc.material.size=size},[size,loaded]);
 useEffect(()=>{const st=state.current;if(st)st.viewer.setControls(pan?st.viewer.earthControls:st.viewer.orbitControls)},[pan,loaded]);
 const sel=markers.find(m=>m.id===selected);
 function boxWorld(m:RegistrationCandidate){const P=corners(m).map(v=>toLocal(transform,v)),lo=[0,1,2].map(i=>Math.min(...P.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...P.map(p=>p[i])));
  return {centre:lo.map((v,i)=>origin[i]+(v+hi[i])/2),size:hi.map((v,i)=>v-lo[i])}}
 function focus(m:RegistrationCandidate){const st=state.current;if(!st)return;const b=boxWorld(m);
  easeView(st.viewer,new st.THREE.Vector3(...b.centre),Math.max(3,Math.max(...b.size)*4),450)}
 function isolate(m:RegistrationCandidate|null){const st=state.current;if(!st)return;const Potree=(window as Win).Potree,v=st.viewer;
  if(clip.current){v.scene.removeVolume(clip.current);clip.current=null}
  if(!m){v.setClipTask(Potree.ClipTask.HIGHLIGHT);setIsolated(null);return}
  const b=boxWorld(m),vol=new Potree.BoxVolume();vol.name='Close-up';vol.clip=true;  // the box and 1.5 m around it: everything else is hidden
  vol.position.set(b.centre[0],b.centre[1],b.centre[2]);vol.scale.set(b.size[0]+3,b.size[1]+3,b.size[2]+3);v.scene.addVolume(vol);vol.updateMatrixWorld(true);
  v.setClipTask(Potree.ClipTask.SHOW_INSIDE);clip.current=vol;setIsolated(m.id);focus(m)}
 useEffect(()=>{if(isolated!=null&&isolated!==selected)isolate(sel||null)},[selected]);  // isolating follows the selected box
 function pickAt(e:React.PointerEvent<HTMLDivElement>){const st=state.current,el=host.current;if(!st||!el)return null;
  const r=el.getBoundingClientRect(),ray=new st.THREE.Raycaster();ray.setFromCamera({x:(e.clientX-r.left)/r.width*2-1,y:-(e.clientY-r.top)/r.height*2+1},st.viewer.scene.getActiveCamera());
  const hit=ray.intersectObjects(st.picks.map(p=>p.mesh),false)[0];return hit?st.picks.find(p=>p.mesh===hit.object)?.m||null:null}
 const down=useRef<[number,number]|null>(null);
 return <div style={{position:'relative',width:'100%',height:'100%',minHeight:360}}
   onPointerDown={e=>{down.current=[e.clientX,e.clientY]}}
   onPointerUp={e=>{const d=down.current;down.current=null;if(!d||Math.hypot(e.clientX-d[0],e.clientY-d[1])>4)return;const m=pickAt(e);if(m)onPick?.(m.id)}}
   onPointerMove={e=>{if(e.buttons)return;const m=pickAt(e);const r=host.current?.getBoundingClientRect();setHover(m&&r?{m,x:e.clientX-r.left,y:e.clientY-r.top}:null)}}
   onPointerLeave={()=>setHover(null)}
   onDoubleClick={e=>{const st=state.current,r=host.current?.getBoundingClientRect();if(!st||!r)return;const m=pickAt(e as unknown as React.PointerEvent<HTMLDivElement>);
    if(m&&!e.shiftKey){onPick?.(m.id);focus(m);return}zoomAt(st,e.clientX-r.left,e.clientY-r.top,e.shiftKey?ZOOM_OUT:ZOOM_IN)}}>
  <div ref={host} style={{position:'absolute',inset:0}}/>
  {hover&&<div style={{position:'absolute',left:hover.x+14,top:hover.y+10,pointerEvents:'none',zIndex:7,padding:'6px 9px',borderRadius:8,background:'rgba(255,255,255,.95)',color:'#1d2f40',fontSize:12,boxShadow:'0 4px 14px rgba(0,0,0,.25)'}}><strong>{hover.m.class_name.replaceAll('_',' ')}</strong><br/><small>{hover.m.media_count} images · {Math.round(hover.m.confidence*100)}%</small></div>}
  {loaded>0&&<ViewToolbar orthographic mode={mode} onView={v=>{const st=state.current;if(st){setView(st.viewer,v);setMode(v)}}} pan={pan} onPan={()=>setPan(p=>!p)}
   onFit={()=>{const st=state.current;if(st){isolate(null);st.viewer.fitToScreen(.8)}}} canFocus={!!sel} onFocus={()=>sel&&focus(sel)}
   isolated={isolated!=null} onIsolate={()=>isolate(isolated!=null?null:sel||null)} size={size} onSize={setSize}
   colour={colour} onColour={setColour} colours={{rgb:true,elevation:true,intensity:true}}/>}
  {err&&<div className="recon-msg">Could not start the LiDAR viewer: {err}</div>}
  {!err&&!loaded&&<div className="recon-msg">Streaming the LiDAR point cloud…</div>}
 </div>;
}
