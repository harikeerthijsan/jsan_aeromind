import {useEffect,useMemo,useRef,useState} from 'react';
import {Canvas,useFrame,useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {API,getAuthToken} from './api';
import {heightRange,recolour,usePly} from './Reconstruction';

/** Photo walk (street-view style): stand where the drone took each photo, the photo as the backdrop and the LiDAR /
 * 3D points drawn over it in the same lens, then step from photo to photo with the camera flying between the poses. */
export type Pose={centre:number[];right:number[];down:number[];forward:number[];lens:number[]};  // lens: fx fy cx cy k1 k2 p1 p2
export type WalkPhoto={media_id:number;filename:string;width:number;height:number;pose?:Pose};
type Station={photo:WalkPhoto;pos:THREE.Vector3;quat:THREE.Quaternion;tex:THREE.Texture|null};

const v3=(a:number[])=>new THREE.Vector3(a[0],a[1],a[2]);
/** three.js cameras look down -Z with +Y up: right, up (= -down) and back (= -forward) are the camera's local axes. */
function poseQuat(p:Pose){const r=v3(p.right).normalize(),d=v3(p.down).normalize(),f=v3(p.forward).normalize();
 return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(r,d.negate(),f.negate()))}
const textures=new Map<number,Promise<THREE.Texture>>();  // photos already fetched, for stepping back and forth
function photoTexture(id:number){let t=textures.get(id);if(t)return t;
 const tok=getAuthToken();
 t=fetch(`${API}/api/v1/media/${id}/thumb?w=1280`,{credentials:'include',headers:tok?{Authorization:`Bearer ${tok}`}:{}})
  .then(r=>{if(!r.ok)throw Error(`photo ${id}: ${r.status}`);return r.blob()}).then(b=>createImageBitmap(b,{imageOrientation:'flipY'}))
  .then(bmp=>{const x=new THREE.Texture(bmp);x.flipY=false;x.colorSpace=THREE.SRGBColorSpace;x.anisotropy=4;x.needsUpdate=true;return x});
 t.catch(()=>textures.delete(id));textures.set(id,t);if(textures.size>40)textures.delete(textures.keys().next().value!);return t}

// The photo as a screen far in front of the camera, undistorted on the fly: each fragment is a direction from the
// camera (normalised image coordinates x, y); the lens model (OpenCV: k1 k2 radial, p1 p2 tangential) gives the
// pixel of the raw photo that shows it. Drawn first with no depth, so every 3D point lands on top of it.
const VERT=`attribute vec2 xy;varying vec2 vXY;void main(){vXY=xy;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`;
const FRAG=`uniform sampler2D map;uniform vec4 K;uniform vec4 D;uniform vec2 size;uniform float opacity;varying vec2 vXY;
void main(){float x=vXY.x,y=vXY.y,r2=x*x+y*y,rad=1.0+D.x*r2+D.y*r2*r2;
 float xd=x*rad+2.0*D.z*x*y+D.w*(r2+2.0*x*x),yd=y*rad+D.z*(r2+2.0*y*y)+2.0*D.w*x*y;
 vec2 uv=vec2((K.x*xd+K.z)/size.x,1.0-(K.y*yd+K.w)/size.y);
 if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0)discard;
 gl_FragColor=vec4(texture2D(map,uv).rgb,opacity);
 #include <colorspace_fragment>
}`;
function photoScreen(st:Station,dist:number){
 const p=st.photo.pose!,[fx,fy,cx,cy,k1,k2,p1,p2]=p.lens,w=st.photo.width,h=st.photo.height,pad=.06;
 const x0=-cx/fx*(1+pad),x1=(w-cx)/fx*(1+pad),y0=-cy/fy*(1+pad),y1=(h-cy)/fy*(1+pad);
 const c=v3(p.centre),r=v3(p.right).normalize(),d=v3(p.down).normalize(),f=v3(p.forward).normalize();
 const at=(x:number,y:number)=>c.clone().addScaledVector(f,dist).addScaledVector(r,x*dist).addScaledVector(d,y*dist);
 const N=24,pos:number[]=[],xy:number[]=[],idx:number[]=[];  // a grid, so the photo still lines up when it is partly behind the camera
 for(let j=0;j<=N;j++)for(let i=0;i<=N;i++){const x=x0+(x1-x0)*i/N,y=y0+(y1-y0)*j/N;pos.push(...at(x,y).toArray());xy.push(x,y)}
 for(let j=0;j<N;j++)for(let i=0;i<N;i++){const a=j*(N+1)+i;idx.push(a,a+1,a+N+1,a+1,a+N+2,a+N+1)}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('xy',new THREE.Float32BufferAttribute(xy,2));g.setIndex(idx);
 const m=new THREE.ShaderMaterial({vertexShader:VERT,fragmentShader:FRAG,transparent:true,depthTest:false,depthWrite:false,side:THREE.DoubleSide,
  uniforms:{map:{value:st.tex},K:{value:new THREE.Vector4(fx,fy,cx,cy)},D:{value:new THREE.Vector4(k1,k2,p1,p2)},size:{value:new THREE.Vector2(w,h)},opacity:{value:0}}});
 const mesh=new THREE.Mesh(g,m);mesh.renderOrder=-10;mesh.frustumCulled=false;return mesh}

type Look={yaw:number;pitch:number;zoom:number};
/** Overlay colours: by height (stands out on any photo), the cloud's own colours, or one bright colour. */
type Tint='elevation'|'rgb'|'highlight';
const TINTS:[Tint,string][]=[['elevation','Height'],['rgb','RGB'],['highlight','Highlight']];
function Walker({stations,cur,photoOpacity,lidar,surroundings,pointOpacity,pointSize,tint,look,onMoving,geo,ctxGeo}:{stations:Station[];cur:number;photoOpacity:number;lidar:boolean;surroundings:boolean;pointOpacity:number;pointSize:number;tint:Tint;look:React.MutableRefObject<Look>;onMoving:(b:boolean)=>void;geo:THREE.BufferGeometry|null;ctxGeo:THREE.BufferGeometry|null}){
 const {camera,size}=useThree(),cam=camera as THREE.PerspectiveCamera;
 const radius=useMemo(()=>{if(!geo)return 50;geo.computeBoundingSphere();return geo.boundingSphere!.radius},[geo]);
 const pts=useMemo(()=>[geo,ctxGeo].map(g=>{if(!g)return null;const o=new THREE.Points(g,new THREE.PointsMaterial({size:pointSize,sizeAttenuation:false,vertexColors:!!g.getAttribute('color'),color:g.getAttribute('color')?'#ffffff':'#9fd0ff',transparent:true}));
  o.renderOrder=5;o.frustumCulled=false;return o}),[geo,ctxGeo]);
 useEffect(()=>{const r=tint==='elevation'&&geo?heightRange(geo):null;  // one height range for the structure and its surroundings
  pts.forEach(o=>{if(!o)return;const m=o.material as THREE.PointsMaterial,g=o.geometry,rgb=!!g.getAttribute('color');
   if(rgb)recolour(g,tint==='elevation'?'elevation':'rgb',r);m.vertexColors=rgb&&tint!=='highlight';m.color.set(m.vertexColors?'#ffffff':'#2ee6ff');m.needsUpdate=true})},[pts,tint]);
 useEffect(()=>{pts.forEach((o,i)=>{if(!o)return;const m=o.material as THREE.PointsMaterial;m.size=pointSize;m.opacity=pointOpacity;o.visible=lidar&&(i===0||surroundings)})},[pts,pointSize,pointOpacity,lidar,surroundings]);
 // flight between two poses: position and orientation glide, the old photo fades out and the new one in
 const fly=useRef<{from:THREE.Vector3;fromQ:THREE.Quaternion;to:number;t0:number;prev:THREE.Mesh|null}|null>(null),screen=useRef<THREE.Mesh|null>(null),at=useRef<number>(-1);
 const group=useRef<THREE.Group>(null!);
 const dist=radius*6;
 const drop=(m:THREE.Mesh|null|undefined)=>{if(!m)return;group.current.remove(m);m.geometry.dispose();(m.material as THREE.Material).dispose()};
 useEffect(()=>{const st=stations[cur];if(!st)return;
  const old=screen.current,mesh=photoScreen(st,dist);group.current.add(mesh);drop(fly.current?.prev);  // a new step mid-flight: the oldest photo goes
  if(at.current<0||at.current===cur){drop(old);fly.current=null;cam.position.copy(st.pos);cam.quaternion.copy(st.quat);onMoving(false)}
  else{fly.current={from:cam.position.clone(),fromQ:cam.quaternion.clone(),to:cur,t0:performance.now(),prev:old};onMoving(true)}
  screen.current=mesh;at.current=cur},[cur,dist]);
 useEffect(()=>()=>{group.current?.clear()},[]);
 const base=useRef(new THREE.Quaternion()),tmp=useRef(new THREE.Quaternion()),eul=useRef(new THREE.Euler(0,0,0,'YXZ'));
 useFrame(()=>{const st=stations[cur];if(!st?.photo.pose)return;
  const p=st.photo.pose,[,fy,,,]=p.lens,aspect=size.width/size.height,ph=st.photo.height/2/fy,pw=st.photo.width/2/p.lens[0];
  cam.near=radius*.002;cam.far=radius*40;cam.fov=THREE.MathUtils.radToDeg(2*Math.atan(Math.max(ph,pw/aspect)/look.current.zoom));cam.aspect=aspect;cam.updateProjectionMatrix();
  let k=1;const f=fly.current;
  if(f){const T=900;k=Math.min(1,(performance.now()-f.t0)/T);const e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;
   cam.position.lerpVectors(f.from,st.pos,e);base.current.slerpQuaternions(f.fromQ,st.quat,e);
   if(f.prev)(f.prev.material as THREE.ShaderMaterial).uniforms.opacity.value=photoOpacity*Math.max(0,1-k*2.5);
   if(k>=1){drop(f.prev);fly.current=null;onMoving(false)}}
  else{cam.position.copy(st.pos);base.current.copy(st.quat)}
  // look around on the spot: turn about the photo's own up and right axes
  eul.current.set(look.current.pitch,look.current.yaw,0);tmp.current.setFromEuler(eul.current);cam.quaternion.copy(base.current).multiply(tmp.current);
  const m=screen.current;if(m){const u=(m.material as THREE.ShaderMaterial).uniforms;u.map.value=st.tex;u.opacity.value=st.tex?photoOpacity*(f?Math.max(0,(k-.55)/.45):1):0}});
 return <group ref={group}>{pts.map((o,i)=>o&&<primitive key={i} object={o}/>)}</group>;
}

export default function WalkView({photos,idx,setIdx,pointsUrl,contextUrl}:{photos:WalkPhoto[];idx:number;setIdx:(i:number)=>void;pointsUrl?:string;contextUrl?:string}){
 const {geo,err}=usePly(pointsUrl),ctx=usePly(contextUrl);
 const [texOf,setTexOf]=useState<Record<number,THREE.Texture>>({}),[photoOpacity,setPhotoOpacity]=useState(1),[lidar,setLidar]=useState(true),[pointSize,setPointSize]=useState(1),[pointOpacity,setPointOpacity]=useState(.35),[surroundings,setSurroundings]=useState(false),[tint,setTint]=useState<Tint>('elevation'),[flying,setFlying]=useState(false);
 const look=useRef<Look>({yaw:0,pitch:0,zoom:1});
 const stations=useMemo(()=>photos.map(p=>p.pose?{photo:p,pos:v3(p.pose.centre),quat:poseQuat(p.pose),tex:texOf[p.media_id]||null}:null),[photos,texOf]);
 const ok=stations.filter(Boolean) as Station[],cur=stations[idx]?ok.indexOf(stations[idx]!):-1;
 // the photo shown, then its neighbours in the background so the next step is instant
 useEffect(()=>{let live=true;const want=[idx,idx+1,idx-1,idx+2].map(i=>photos[i]).filter(Boolean);
  for(const p of want)photoTexture(p.media_id).then(t=>{if(live)setTexOf(m=>m[p.media_id]===t?m:{...m,[p.media_id]:t})}).catch(()=>{});
  return()=>{live=false}},[idx,photos]);
 useEffect(()=>{look.current={yaw:0,pitch:0,zoom:1}},[idx]);
 // ↑ / ↓: the nearest photo taken ahead of (or behind) where the camera is looking, like walking down a street
 const step=(dir:1|-1)=>{const s=stations[idx];if(!s)return;const f=new THREE.Vector3(0,0,-1).applyQuaternion(s.quat).multiplyScalar(dir);let best=-1,bd=Infinity;
  stations.forEach((o,i)=>{if(!o||i===idx)return;const d=o.pos.clone().sub(s.pos),l=d.length();if(l<1e-6||d.dot(f)/l<Math.cos(THREE.MathUtils.degToRad(40)))return;if(l<bd){bd=l;best=i}});
  if(best>=0)setIdx(best)};
 const next=(d:1|-1)=>{for(let i=idx+d;i>=0&&i<photos.length;i+=d)if(stations[i]){setIdx(i);return}};
 useEffect(()=>{const k=(e:KeyboardEvent)=>{const tag=(e.target as HTMLElement)?.tagName;if(tag==='INPUT'||tag==='SELECT')return;
  if(e.key==='ArrowUp'||e.key==='w'){e.preventDefault();step(1)}else if(e.key==='ArrowDown'||e.key==='s'){e.preventDefault();step(-1)}
  else if(e.key==='d'){next(1)}else if(e.key==='a'){next(-1)}
  else if(e.key==='r')look.current={yaw:0,pitch:0,zoom:1}};
  window.addEventListener('keydown',k,true);return()=>window.removeEventListener('keydown',k,true)},[idx,stations]);
 // drag: look around from where the photo was taken; wheel: zoom the lens; double-click: back to the photo's view
 const box=useRef<HTMLDivElement>(null),drag=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
 useEffect(()=>{const el=box.current;if(!el)return;const w=(e:WheelEvent)=>{e.preventDefault();const L=look.current;L.zoom=Math.min(12,Math.max(.5,L.zoom*Math.exp(-e.deltaY*.0015)))};
  el.addEventListener('wheel',w,{passive:false});return()=>el.removeEventListener('wheel',w)},[]);
 const turn=()=>{const st=stations[idx];return st?.photo.pose?Math.atan(st.photo.height/2/st.photo.pose.lens[1])*2/(box.current?.clientHeight||600)/look.current.zoom:.002};
 const onDown=(e:React.PointerEvent)=>{drag.current={x:e.clientX,y:e.clientY,yaw:look.current.yaw,pitch:look.current.pitch};(e.target as Element).setPointerCapture?.(e.pointerId)};
 const onMove=(e:React.PointerEvent)=>{const d=drag.current;if(!d)return;const k=turn();look.current.yaw=d.yaw+(e.clientX-d.x)*k;look.current.pitch=Math.max(-1.4,Math.min(1.4,d.pitch+(e.clientY-d.y)*k))};
 const missing=photos.length>0&&!ok.length,here=stations[idx],ahead=useMemo(()=>{if(!here)return [false,false];const f=new THREE.Vector3(0,0,-1).applyQuaternion(here.quat);
  return [1,-1].map(dir=>stations.some((o,i)=>{if(!o||i===idx)return false;const d=o.pos.clone().sub(here.pos),l=d.length();return l>1e-6&&d.dot(f)*dir/l>=Math.cos(THREE.MathUtils.degToRad(40))}))},[here,stations]);
 return <div className="pw-wrap">
  <div ref={box} className="pw-view" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={()=>{drag.current=null}} onDoubleClick={()=>{look.current={yaw:0,pitch:0,zoom:1}}}>
   <Canvas camera={{fov:50,near:.01,far:1000}} gl={{antialias:true}} onCreated={({gl})=>gl.setClearColor('#0b1622')}>
    {cur>=0&&<Walker stations={ok} cur={cur} photoOpacity={photoOpacity} lidar={lidar} surroundings={surroundings} pointOpacity={pointOpacity} pointSize={pointSize} tint={tint} look={look} onMoving={setFlying} geo={geo} ctxGeo={ctx.geo}/>}
   </Canvas>
   {!geo&&!err&&pointsUrl&&<div className="pv-msg">Loading the 3D points…</div>}
   {err&&<div className="pv-msg error">3D points: {err}</div>}
   {missing&&<div className="pv-msg">The camera poses are not available for these photos (the backend needs the latest update).</div>}
   {here&&!here.tex&&<div className="pw-loading">Loading photo…</div>}
   <div className="pw-hud">
    <button disabled={!ahead[1]} onClick={()=>step(-1)} title="Step back (↓ / S)">▼</button>
    <button disabled={!ahead[0]} onClick={()=>step(1)} title="Step forward, towards where the camera looks (↑ / W)">▲</button>
   </div>
  </div>
  <div className="pw-controls">
   <label className="pw-check"><input type="checkbox" checked={lidar} onChange={e=>setLidar(e.target.checked)}/>LiDAR overlay</label>
   <label className="pw-range"><span>Photo</span><input type="range" min={0} max={1} step={.05} value={photoOpacity} onChange={e=>setPhotoOpacity(+e.target.value)}/><em>{Math.round(photoOpacity*100)}%</em></label>
   <label className="pw-range"><span>Colour</span><select value={tint} disabled={!lidar} onChange={e=>setTint(e.target.value as Tint)}>{TINTS.map(([k,l])=><option key={k} value={k}>{l}</option>)}</select></label>
   {contextUrl&&<label className="pw-check" title="Trees, ground and buildings around the structure (the photo already shows them)"><input type="checkbox" checked={surroundings} disabled={!lidar} onChange={e=>setSurroundings(e.target.checked)}/>Trees &amp; ground</label>}
   <label className="pw-range"><span>LiDAR</span><input type="range" min={.1} max={1} step={.05} value={pointOpacity} disabled={!lidar} onChange={e=>setPointOpacity(+e.target.value)}/><em>{Math.round(pointOpacity*100)}%</em></label>
   <label className="pw-range"><span>Size</span><input type="range" min={1} max={6} step={.5} value={pointSize} disabled={!lidar} onChange={e=>setPointSize(+e.target.value)}/><em>{pointSize}px</em></label>
   <small className="pw-hint">{flying?'Moving…':'Drag to look around · wheel to zoom · double-click to reset · ← → previous / next photo · ↑ ↓ walk forward / back'}</small>
  </div>
 </div>;
}
