import {useEffect,useMemo,useRef,useState} from 'react';
import {Canvas,useFrame,useThree} from '@react-three/fiber';
import {Html,Line} from '@react-three/drei';
import {ArrowUpFromLine,Eye,Ruler} from 'lucide-react';
import * as THREE from 'three';
import {API,getAuthToken} from './api';
import {heightRange,recolour,usePly} from './Reconstruction';

/** Photo walk (street-view style): stand where the drone took each photo, the photo as the backdrop and the LiDAR /
 * 3D points drawn over it in the same lens, then step from photo to photo with the camera flying between the poses. */
export type Pose={centre:number[];right:number[];down:number[];forward:number[];lens:number[]};  // lens: fx fy cx cy k1 k2 p1 p2
export type WalkPhoto={media_id:number;filename:string;width:number;height:number;pose?:Pose};
/** A measurement drawn in the walk, in the 3D viewer's frame (the page measures; the walk draws). */
export type WalkMark={id:number;kind:'distance'|'height';a:number[];b?:number[];foot?:number[];label:string};
export type WalkTool='distance'|'height'|'look';
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
type Tint='photo'|'elevation'|'rgb'|'highlight';
const TINTS:[Tint,string][]=[['photo','Photo'],['elevation','Height'],['rgb','RGB'],['highlight','Highlight']];
// Points coloured from the photos ("Photo"): each point takes its colour from the photo shown or, outside it, from the
// neighbouring photos that see it (projected through each photo's pose and lens), so the scene carries on past the
// photo's edges and between steps instead of ending in black. Inside the photo shown the points stay a light overlay.
const SLOTS=5;  // the photo shown + 4 neighbours
const PT_VERT=`uniform float size;uniform float opacity;uniform int mode;uniform int nPhotos;
${Array.from({length:SLOTS},(_,i)=>`uniform sampler2D map${i};uniform vec3 C${i};uniform mat3 R${i};uniform vec4 K${i};uniform vec2 D${i};uniform vec2 S${i};`).join('')}
varying vec4 vCol;
vec4 seen(sampler2D map,vec3 C,mat3 R,vec4 K,vec2 D,vec2 S,vec3 p){vec3 q=R*(p-C);if(q.z<.05)return vec4(0.);vec2 xy=q.xy/q.z;float r2=dot(xy,xy);float rad=1.+D.x*r2+D.y*r2*r2;
 vec2 uv=vec2((K.x*xy.x*rad+K.z)/S.x,(K.y*xy.y*rad+K.w)/S.y);if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.)return vec4(0.);return vec4(texture2D(map,vec2(uv.x,1.-uv.y)).rgb,1.);}
#include <color_pars_vertex>
void main(){vec4 mv=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*mv;gl_PointSize=size;
 vec3 own=vec3(1.);
 #ifdef USE_COLOR
 own=color.rgb;
 #endif
 vCol=vec4(own,opacity);if(mode==1)vCol.rgb=vec3(.18,.9,1.);
 if(mode==2&&nPhotos>0){vec4 c=seen(map0,C0,R0,K0,D0,S0,position);
  if(c.a>0.)vCol=vec4(c.rgb,opacity);
  else{gl_PointSize=max(size,2.5);vCol.a=1.;
${Array.from({length:SLOTS-1},(_,j)=>`   if(c.a==0.&&nPhotos>${j+1}){c=seen(map${j+1},C${j+1},R${j+1},K${j+1},D${j+1},S${j+1},position);if(c.a>0.)vCol.rgb=c.rgb;}`).join('\n')}
  }}
}`;
const PT_FRAG=`varying vec4 vCol;void main(){gl_FragColor=vCol;
#include <colorspace_fragment>
}`;
function pointMaterial(g:THREE.BufferGeometry){
 const u:Record<string,THREE.IUniform>={size:{value:1},opacity:{value:.35},mode:{value:0},nPhotos:{value:0}};
 for(let i=0;i<SLOTS;i++)Object.assign(u,{[`map${i}`]:{value:null},[`C${i}`]:{value:new THREE.Vector3()},[`R${i}`]:{value:new THREE.Matrix3()},[`K${i}`]:{value:new THREE.Vector4(1,1,0,0)},[`D${i}`]:{value:new THREE.Vector2()},[`S${i}`]:{value:new THREE.Vector2(1,1)}});
 return new THREE.ShaderMaterial({vertexShader:PT_VERT,fragmentShader:PT_FRAG,uniforms:u,transparent:true,vertexColors:!!g.getAttribute('color')});
}
/** A station's photo into a point material's slot i (pose rows: right, down, forward; viewer frame). */
function setSlot(m:THREE.ShaderMaterial,i:number,st:Station){const p=st.photo.pose!,u=m.uniforms;u[`map${i}`].value=st.tex;u[`C${i}`].value.set(...(p.centre as [number,number,number]));
 const r=v3(p.right).normalize(),d=v3(p.down).normalize(),f=v3(p.forward).normalize();(u[`R${i}`].value as THREE.Matrix3).set(r.x,r.y,r.z,d.x,d.y,d.z,f.x,f.y,f.z);
 u[`K${i}`].value.set(p.lens[0],p.lens[1],p.lens[2],p.lens[3]);u[`D${i}`].value.set(p.lens[4],p.lens[5]);u[`S${i}`].value.set(st.photo.width,st.photo.height)}
function Walker({stations,cur,photoOpacity,lidar,surroundings,pointOpacity,pointSize,tint,look,onMoving,toPixel,marks,pendingAt,geo,ctxGeo}:{stations:Station[];cur:number;photoOpacity:number;lidar:boolean;surroundings:boolean;pointOpacity:number;pointSize:number;tint:Tint;look:React.MutableRefObject<Look>;onMoving:(b:boolean)=>void;toPixel:React.MutableRefObject<((nx:number,ny:number)=>[number,number]|null)|null>;marks:WalkMark[];pendingAt?:number[];geo:THREE.BufferGeometry|null;ctxGeo:THREE.BufferGeometry|null}){
 const {camera,size}=useThree(),cam=camera as THREE.PerspectiveCamera;
 const radius=useMemo(()=>{if(!geo)return 50;geo.computeBoundingSphere();return geo.boundingSphere!.radius},[geo]);
 const pts=useMemo(()=>[geo,ctxGeo].map(g=>{if(!g)return null;const o=new THREE.Points(g,pointMaterial(g));o.renderOrder=5;o.frustumCulled=false;return o}),[geo,ctxGeo]);
 useEffect(()=>{const r=tint==='elevation'&&geo?heightRange(geo):null;  // one height range for the structure and its surroundings
  pts.forEach(o=>{if(!o)return;const g=o.geometry;if(g.getAttribute('color'))recolour(g,tint==='elevation'?'elevation':'rgb',r);
   (o.material as THREE.ShaderMaterial).uniforms.mode.value=tint==='highlight'?1:tint==='photo'?2:0})},[pts,tint]);
 useEffect(()=>{pts.forEach((o,i)=>{if(!o)return;const u=(o.material as THREE.ShaderMaterial).uniforms;u.size.value=pointSize;u.opacity.value=pointOpacity;
  o.visible=lidar&&(i===0||surroundings||tint==='photo')})},[pts,pointSize,pointOpacity,lidar,surroundings,tint]);
 // the photos the points take their colours from: the one shown, then its neighbours (those already loaded)
 useEffect(()=>{const near=[cur,cur-1,cur+1,cur-2,cur+2].map(i=>stations[i]).filter(st=>st&&st.tex&&st.photo.pose).slice(0,SLOTS) as Station[];
  pts.forEach(o=>{if(!o)return;const m=o.material as THREE.ShaderMaterial;near.forEach((st,i)=>setSlot(m,i,st));m.uniforms.nPhotos.value=stations[cur]?.tex?near.length:0})},[pts,cur,stations]);
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
 // a click (screen position, -1..1) -> the pixel of the photo the camera stands at: the same lens model as the backdrop
 toPixel.current=(nx,ny)=>{const st=stations[cur],p=st?.photo.pose;if(!p||fly.current)return null;
  const d=new THREE.Vector3(nx,ny,.5).unproject(cam).sub(cam.position).normalize(),fz=d.dot(v3(p.forward).normalize());if(fz<=1e-6)return null;
  const x=d.dot(v3(p.right).normalize())/fz,y=d.dot(v3(p.down).normalize())/fz,[fx,fy,cx,cy,k1,k2,p1,p2]=p.lens,r2=x*x+y*y,rad=1+k1*r2+k2*r2*r2;
  const u=fx*(x*rad+2*p1*x*y+p2*(r2+2*x*x))+cx,v=fy*(y*rad+p1*(r2+2*y*y)+2*p2*x*y)+cy;
  return u>=0&&v>=0&&u<st.photo.width&&v<st.photo.height?[u,v]:null};
 return <group ref={group}>{pts.map((o,i)=>o&&<primitive key={i} object={o}/>)}<Marks marks={marks} pendingAt={pendingAt}/></group>;
}
/** Measurements over everything: thick magenta lines and dots (they stand out on the photo and on every point colour), the value next to them. */
const MARK='#ff3fd5';
function Marks({marks,pendingAt}:{marks:WalkMark[];pendingAt?:number[]}){
 const dots=useMemo(()=>{const a:number[]=[];for(const m of marks){a.push(...m.a);if(m.b)a.push(...m.b)}if(pendingAt)a.push(...pendingAt);
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(a,3));
  const o=new THREE.Points(g,new THREE.PointsMaterial({color:MARK,size:11,sizeAttenuation:false,depthTest:false,transparent:true}));o.renderOrder=52;o.frustumCulled=false;return o},[marks,pendingAt]);
 useEffect(()=>()=>{dots.geometry.dispose();(dots.material as THREE.Material).dispose()},[dots]);
 const v=(p:number[]):[number,number,number]=>[p[0],p[1],p[2]];
 const mid=(m:WalkMark):[number,number,number]=>m.b?[(m.a[0]+m.b[0])/2,(m.a[1]+m.b[1])/2,(m.a[2]+m.b[2])/2]:v(m.a);
 return <>{marks.map(m=>m.b?<Line key={m.id} points={[v(m.a),v(m.b)]} color={MARK} lineWidth={3} depthTest={false} transparent renderOrder={51}/>
   :m.foot?<Line key={m.id} points={[v(m.a),v(m.foot)]} color={MARK} lineWidth={2} dashed dashSize={.08} gapSize={.06} depthTest={false} transparent renderOrder={51}/>:null)}
  <primitive object={dots}/>
  {marks.map(m=><Html key={m.id} position={mid(m)} zIndexRange={[20,0]} style={{pointerEvents:'none'}}><span className="pw-label">{m.label}</span></Html>)}</>;
}

export default function WalkView({photos,idx,setIdx,pointsUrl,contextUrl,tool,onTool,marks=[],pendingAt,onPick,busy,note,result}:{photos:WalkPhoto[];idx:number;setIdx:(i:number)=>void;pointsUrl?:string;contextUrl?:string;
 tool:WalkTool;onTool:(t:WalkTool)=>void;marks?:WalkMark[];pendingAt?:number[];onPick?:(u:number,v:number)=>void;busy?:boolean;note?:string;result?:string}){
 const toPixel=useRef<((nx:number,ny:number)=>[number,number]|null)|null>(null);
 const {geo,err}=usePly(pointsUrl),ctx=usePly(contextUrl);
 const [texOf,setTexOf]=useState<Record<number,THREE.Texture>>({}),[photoOpacity,setPhotoOpacity]=useState(1),[lidar,setLidar]=useState(true),[pointSize,setPointSize]=useState(1),[pointOpacity,setPointOpacity]=useState(.35),[surroundings,setSurroundings]=useState(false),[tint,setTint]=useState<Tint>('photo'),[flying,setFlying]=useState(false);
 const look=useRef<Look>({yaw:0,pitch:0,zoom:1});
 const stations=useMemo(()=>photos.map(p=>p.pose?{photo:p,pos:v3(p.pose.centre),quat:poseQuat(p.pose),tex:texOf[p.media_id]||null}:null),[photos,texOf]);
 const ok=stations.filter(Boolean) as Station[],cur=stations[idx]?ok.indexOf(stations[idx]!):-1;
 // the photo shown, then its neighbours in the background so the next step is instant
 useEffect(()=>{let live=true;const want=[idx,idx+1,idx-1,idx+2,idx-2].map(i=>photos[i]).filter(Boolean);
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
 const onDown=(e:React.PointerEvent)=>{if((e.target as HTMLElement).closest('button'))return;drag.current={x:e.clientX,y:e.clientY,yaw:look.current.yaw,pitch:look.current.pitch};(e.target as Element).setPointerCapture?.(e.pointerId)};
 // a click without dragging measures: the point of the photo under the cursor, snapped to the structure behind it
 const [miss,setMiss]=useState('');
 const onUp=(e:React.PointerEvent)=>{const d=drag.current;drag.current=null;if(!d||Math.hypot(e.clientX-d.x,e.clientY-d.y)>4||tool==='look'||!onPick||busy)return;
  const r=box.current!.getBoundingClientRect(),px=toPixel.current?.((e.clientX-r.left)/r.width*2-1,-((e.clientY-r.top)/r.height)*2+1);
  if(!px){setMiss(flying?'Wait for the camera to arrive.':'Click on the photo: measurements are taken on what this photo shows.');return}setMiss('');onPick(px[0],px[1])};
 const onMove=(e:React.PointerEvent)=>{const d=drag.current;if(!d)return;const k=turn();look.current.yaw=d.yaw+(e.clientX-d.x)*k;look.current.pitch=Math.max(-1.4,Math.min(1.4,d.pitch+(e.clientY-d.y)*k))};
 const missing=photos.length>0&&!ok.length,here=stations[idx],ahead=useMemo(()=>{if(!here)return [false,false];const f=new THREE.Vector3(0,0,-1).applyQuaternion(here.quat);
  return [1,-1].map(dir=>stations.some((o,i)=>{if(!o||i===idx)return false;const d=o.pos.clone().sub(here.pos),l=d.length();return l>1e-6&&d.dot(f)*dir/l>=Math.cos(THREE.MathUtils.degToRad(40))}))},[here,stations]);
 return <div className="pw-wrap">
  <div ref={box} className={'pw-view'+(tool!=='look'?' measuring':'')+(busy?' busy':'')} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onDoubleClick={()=>{look.current={yaw:0,pitch:0,zoom:1}}}>
   <Canvas camera={{fov:50,near:.01,far:1000}} gl={{antialias:true}} onCreated={({gl})=>gl.setClearColor('#0b1622')}>
    {cur>=0&&<Walker stations={ok} cur={cur} photoOpacity={photoOpacity} lidar={lidar} surroundings={surroundings} pointOpacity={pointOpacity} pointSize={pointSize} tint={tint} look={look} onMoving={setFlying} toPixel={toPixel} marks={marks} pendingAt={pendingAt} geo={geo} ctxGeo={ctx.geo}/>}
   </Canvas>
   {!geo&&!err&&pointsUrl&&<div className="pv-msg">Loading the 3D points…</div>}
   {err&&<div className="pv-msg error">3D points: {err}</div>}
   {missing&&<div className="pv-msg">The camera poses are not available for these photos (the backend needs the latest update).</div>}
   {here&&!here.tex&&<div className="pw-loading">Loading photo…</div>}
   <div className="pw-tools">
    <button className={tool==='distance'?'on':''} onClick={()=>onTool('distance')} title="Distance: click two points"><Ruler size={14}/>Distance</button>
    <button className={tool==='height'?'on':''} onClick={()=>onTool('height')} title="Height above the ground: click a point"><ArrowUpFromLine size={14}/>Height</button>
    <button className={tool==='look'?'on':''} onClick={()=>onTool('look')} title="Only look around (clicks do not measure)"><Eye size={14}/>Look</button>
    {tool!=='look'&&<span>{busy?'Measuring…':pendingAt?'Click the second point':tool==='distance'?'Click two points':'Click a point'}</span>}
   </div>
   {result&&<div className="pw-result">{result}</div>}
   {(note||miss)&&<div className="pv-note">{note||miss}</div>}
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
   <small className="pw-hint">{flying?'Moving…':(tool==='look'?'':'Click to measure · ')+'Drag to look around · wheel to zoom · double-click to reset · ← → previous / next photo · ↑ ↓ walk forward / back'}</small>
  </div>
 </div>;
}
