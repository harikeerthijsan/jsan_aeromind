import React,{useEffect,useMemo,useRef,useState} from 'react';
import {MousePointer2,Square,ZoomIn,ZoomOut,Maximize,Minimize,Undo2,Redo2,Trash2,Tag,ChevronLeft,ChevronRight,CheckCircle2,RotateCcw,ScanEye,Eraser,Plus,Download,Sparkles} from 'lucide-react';
import {API,getAuthToken,request,json,Media,Detection,ClassInfo,Review,Annotation,MediaAnnotations,VendorInfo,VendorSuggestion,VendorRef,VendorCandidate,ConditionDefect,ConditionResult} from './api';
const palette=['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000','#aaffc3'];
export const colorOf=(id:number)=>palette[(id%palette.length+palette.length)%palette.length];export const isVideo=(m:Media)=>m.content_type.startsWith('video/');
export type Workflow='ai_detection'|'manual_qc';
// frames extracted from a video are images with parent_media_id; keep them right after their video
export const fmtTime=(s:number)=>`${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
export const frameLabel=(m:Media)=>m.frame_time==null?m.filename:`frame ${String(m.frame_index).padStart(6,'0')} · ${fmtTime(m.frame_time)}`;
export const orderMedia=(ms:Media[])=>[...ms].sort((a,b)=>(a.parent_media_id??a.id)-(b.parent_media_id??b.id)||(a.frame_index??-1)-(b.frame_index??-1));
type Vendor={manufacturer:string|null;product_model:string|null;vendor_source:string|null;vendor_confidence:number|null;vendor_evidence:string|null;vendor_basis:string|null};
const NO_VENDOR:Vendor={manufacturer:null,product_model:null,vendor_source:null,vendor_confidence:null,vendor_evidence:null,vendor_basis:null};
// what an AI manufacturer suggestion rests on
const BASIS:Record<string,string>={logo_text:'logo / text',reference_match:'matches reference photos',reviewer_choice:'reviewer picked from AI candidates',structure:'structure (model knowledge)',none:'no evidence'};
type Condition={condition_grade:string|null;condition_score:number|null;condition_defects:ConditionDefect[]|null;condition_source:string|null;condition_confidence:number|null;condition_verified:string|null;condition_summary:string|null};
const NO_CONDITION:Condition={condition_grade:null,condition_score:null,condition_defects:null,condition_source:null,condition_confidence:null,condition_verified:null,condition_summary:null};
const GRADES=['Good','Fair','Poor','Critical','Unknown'],ISSUE=new Set(['Fair','Poor','Critical']),GRADE_SCORE:Record<string,number>={Good:90,Fair:70,Poor:45,Critical:20};
const DEFECT_TYPES=['rust','corrosion','paint_peeling','discoloration','crack','dent','bent_or_deformed','broken','missing_part','loose_or_misaligned','radome_damage','cable_damage','connector_weathering','water_ingress','bird_nest','vegetation','other'];
type Box=Vendor&Condition&{key:string;detection_id:number|null;class_id:number;class_name:string;confidence:number|null;source:string;x1:number;y1:number;x2:number;y2:number};
type View={x:number;y:number;w:number;h:number};
type Drag={mode:'pan'|'move'|'resize'|'draw';sx:number;sy:number;px:number;py:number;view:View;before:Box[];key?:string;handle?:string;moved:boolean};
const fromDet=(d:Detection):Box=>({key:'d'+d.id,detection_id:d.id,class_id:d.class_id,class_name:d.class_name,confidence:d.confidence,source:'model',x1:d.x1,y1:d.y1,x2:d.x2,y2:d.y2,...NO_VENDOR,...NO_CONDITION});
const fromAnn=(a:Annotation):Box=>({key:'a'+a.id,detection_id:a.detection_id,class_id:a.class_id,class_name:a.class_name,confidence:a.confidence,source:a.source,x1:a.x1,y1:a.y1,x2:a.x2,y2:a.y2,manufacturer:a.manufacturer,product_model:a.product_model,vendor_source:a.vendor_source,vendor_confidence:a.vendor_confidence,vendor_evidence:a.vendor_evidence,vendor_basis:a.vendor_basis,condition_grade:a.condition_grade,condition_score:a.condition_score,condition_defects:a.condition_defects,condition_source:a.condition_source,condition_confidence:a.condition_confidence,condition_verified:a.condition_verified,condition_summary:a.condition_summary});
const edited=(b:Box):Box=>({...b,source:b.detection_id==null?'manual':'model_edited'});
const sig=(bs:Box[])=>bs.map(b=>[b.detection_id,b.class_name,b.x1,b.y1,b.x2,b.y2,b.manufacturer,b.product_model,b.vendor_source,b.condition_grade,b.condition_source,JSON.stringify(b.condition_defects)].join()).sort().join('|');
const HANDLES=['nw','n','ne','e','se','s','sw','w'];const SOURCE:Record<string,string>={model:'AI',model_edited:'AI · edited',manual:'Manual'};

let openGroupsMemory=new Set<string>();// box-list class groups the reviewer has expanded
const T=({title,on,onClick,disabled,children}:{title:string;on?:boolean;onClick:()=>void;disabled?:boolean;children:React.ReactNode})=><button title={title} aria-label={title} className={on?'on':''} onClick={onClick} disabled={disabled}>{children}</button>;// toolbar button

/** Text input that commits on blur or Enter, so typing doesn't create an undo step per keystroke. */
function Commit({value,onCommit,...rest}:{value:string;onCommit:(v:string)=>void}&Omit<React.InputHTMLAttributes<HTMLInputElement>,'value'|'onChange'>){const [v,setV]=useState(value);return <input {...rest} value={v} maxLength={100} onChange={e=>setV(e.target.value)} onBlur={()=>v.trim()!==value&&onCommit(v)} onKeyDown={e=>{if(e.key==='Enter')(e.target as HTMLInputElement).blur()}}/>}
/** Loads everything a review page needs for one inspection; `refresh` changes re-fetch it (e.g. a job finishing). */
export function useReviewData(inspectionId:number|undefined,refresh:unknown){
 const [data,setData]=useState<{id:number;media:Media[];detections:Detection[];reviews:Review[]}|null>(null),[classes,setClasses]=useState<ClassInfo[]>([]),[vendors,setVendors]=useState<VendorInfo|null>(null),[error,setError]=useState('');
 const loadClasses=()=>request<ClassInfo[]>('/api/v1/classes').then(setClasses).catch(()=>{});
 useEffect(()=>{if(!inspectionId)return;let live=true;Promise.all([request<Media[]>(`/api/v1/inspections/${inspectionId}/media`),request<Detection[]>(`/api/v1/inspections/${inspectionId}/detections`),request<Review[]>(`/api/v1/inspections/${inspectionId}/reviews`)]).then(([media,detections,reviews])=>{if(live){setData({id:inspectionId,media,detections,reviews});setError('')}}).catch(e=>live&&setError(String(e)));loadClasses();request<VendorInfo>('/api/v1/vendors').then(setVendors).catch(()=>{});return()=>{live=false}},[inspectionId,refresh]);
 const d=data?.id===inspectionId?data:null;// never show the previous inspection's media while the next one loads
 const saved=(r:Review)=>{setData(x=>x&&{...x,reviews:[...x.reviews.filter(y=>y.id!==r.id),r]});loadClasses()};
 return {media:d?.media||[],detections:d?.detections||[],reviews:d?.reviews||[],classes,vendors,error,setError,saved,loading:!d};
}
/** Tracks the open editor's unsaved state so switching image/inspection can ask first. */
export function useDirtyGuard(){const dirty=useRef(false);return {onDirty:(d:boolean)=>{dirty.current=d},guard:(fn:()=>void)=>{if(!dirty.current||confirm('Discard unsaved box changes on this image?')){dirty.current=false;fn()}}}}

export default function AnnotationEditor({media,workflow,detections,jobId,classes,minConf,onMinConf,hidden,onToggleClass,shell,nav,onSaved,onDirty,notice,vendors,focusKey}:{focusKey?:string;vendors:VendorInfo|null;media:Media;workflow:Workflow;detections:Detection[];jobId:number|null;classes:ClassInfo[];minConf:number;onMinConf:(v:number)=>void;hidden:Set<string>;onToggleClass:(name:string)=>void;shell:React.RefObject<HTMLDivElement|null>;nav?:{position:string;prev?:()=>void;next?:()=>void};onSaved:(r:Review)=>void;onDirty:(dirty:boolean)=>void;notice?:React.ReactNode}){
 const [natural,setNatural]=useState<{w:number;h:number}|null>(null),[imageUrl,setImageUrl]=useState(''),[previewOnly,setPreviewOnly]=useState(false),[loaded,setLoaded]=useState<MediaAnnotations|null>(null),[error,setError]=useState(''),[saving,setSaving]=useState(false),[notes,setNotes]=useState('');
 const [hist,setHist]=useState<{past:Box[][];present:Box[];future:Box[][]}>({past:[],present:[],future:[]}),[baseline,setBaseline]=useState<Box[]>([]);
 const [view,setView]=useState<View>({x:0,y:0,w:1,h:1}),[size,setSize]=useState({w:1,h:1}),[full,setFull]=useState(false),[panning,setPanning]=useState(false);
 const [tool,setTool]=useState<'select'|'draw'>('select'),[selected,setSelected]=useState<string|null>(null),[hover,setHover]=useState<string|null>(null),[labels,setLabels]=useState(false),[cursor,setCursor]=useState<{x:number;y:number}|null>(null),[draft,setDraft]=useState<Box|null>(null),[drawClass,setDrawClass]=useState(''),[extra,setExtra]=useState<string[]>([]),[newClass,setNewClass]=useState('');
 const jobRef=useRef<number|null>(null),// detection run the current boxes came from; approval rejects only that run's unkept detections
 svg=useRef<SVGSVGElement>(null),stage=useRef<HTMLDivElement>(null),list=useRef<HTMLDivElement>(null),drag=useRef<Drag|null>(null),spaceHeld=useRef(false),seq=useRef(0);
 const boxes=hist.present,review=loaded?.review||null;
 // the editor is keyed by media + detection run, so this runs once per image
 useEffect(()=>{let live=true,objectUrl='';const abort=new AbortController(),token=getAuthToken();
  // Protected media cannot rely on the session cookie when local development mixes localhost and 127.0.0.1.
  // Fetch it with the same bearer token as API requests, then let the SVG render the authenticated blob.
  // the stored preview (1280 px, ~250 KB) first, then the 12-16 MB original for sharp zooming; a broken download of the
  // original (net::ERR_FAILED on a slow line) is retried once, and the preview stays usable if it fails again
  const urls:string[]=[];let full=false;
  const get=(path:string)=>fetch(API+path,{credentials:'include',headers:token?{Authorization:`Bearer ${token}`}:{},signal:abort.signal}).then(async r=>{if(!r.ok)throw Error(`Could not load the image (${r.status}).`);return r.blob()});
  const show=(blob:Blob,original:boolean)=>new Promise<{w:number;h:number}|null>(done=>{if(!live||(full&&!original))return done(null);const u=URL.createObjectURL(blob);urls.push(u);const img=new Image();
   img.onload=()=>{if(live&&(original||!full)){if(original){full=true;setNatural({w:img.naturalWidth,h:img.naturalHeight});setPreviewOnly(false)}setImageUrl(u)}done({w:img.naturalWidth,h:img.naturalHeight})};img.onerror=()=>done(null);img.src=u});
  let previewSize:{w:number;h:number}|null=null;
  get(`/api/v1/media/${media.id}/thumb?w=1280`).then(b=>show(b,false)).then(s=>{previewSize=s}).catch(()=>{});
  const original=(tries:number):Promise<unknown>=>get(media.url).then(b=>show(b,true)).catch(e=>{if(!live||e?.name==='AbortError')return;
   if(tries<1)return new Promise(r=>setTimeout(r,1500)).then(()=>original(tries+1));
   if(previewSize){setPreviewOnly(true);setNatural(n=>n||previewSize)}else setError(String(e?.message||e))});
  original(0);objectUrl='';void objectUrl;
  request<MediaAnnotations>(`/api/v1/media/${media.id}/annotations`).then(r=>{if(!live)return;const init=r.review?r.annotations.map(fromAnn):detections.map(fromDet);jobRef.current=r.review?r.review.job_id:jobId;setLoaded(r);setHist({past:[],present:init,future:[]});setBaseline(init);setNotes(r.review?.notes||'')}).catch(e=>live&&setError(String(e)));return()=>{live=false;abort.abort();urls.forEach(u=>URL.revokeObjectURL(u))}},[]);
 // boxes are in the detector's pixel space; fall back to the browser's decoded size for images without any
 const ref=detections[0]||loaded?.annotations[0];const space=ref?{w:ref.image_width,h:ref.image_height}:natural;
 useEffect(()=>{if(space)setView({x:0,y:0,w:space.w,h:space.h})},[space?.w,space?.h]);
 useEffect(()=>{const el=stage.current;if(!el)return;const ro=new ResizeObserver(([e])=>setSize({w:e.contentRect.width||1,h:e.contentRect.height||1}));ro.observe(el);return()=>ro.disconnect()},[]);
 useEffect(()=>{const h=()=>setFull(!!document.fullscreenElement);h();document.addEventListener('fullscreenchange',h);return()=>document.removeEventListener('fullscreenchange',h)},[]);
 const k=Math.max(view.w/size.w,view.h/size.h);// image px per screen px
 const catalog=useMemo(()=>{const m=new Map<string,number>();classes.forEach(c=>m.set(c.name,c.id));[...baseline,...boxes].forEach(b=>m.has(b.class_name)||m.set(b.class_name,b.class_id));extra.forEach(n=>m.has(n)||m.set(n,Math.max(-1,...m.values())+1));return m},[classes,boxes,baseline,extra]);
 const classList=[...catalog].sort((a,b)=>a[1]-b[1]||a[0].localeCompare(b[0]));const activeClass=drawClass||classList[0]?.[0]||'unlabeled';
 const dirty=useMemo(()=>sig(boxes)!==sig(baseline),[boxes,baseline]);
 useEffect(()=>{onDirty(dirty);if(!dirty)return;const h=(e:BeforeUnloadEvent)=>e.preventDefault();addEventListener('beforeunload',h);return()=>removeEventListener('beforeunload',h)},[dirty]);
 const passing=boxes.filter(b=>(b.confidence??1)>=minConf),shown=passing.filter(b=>!hidden.has(b.class_name)),sel=shown.find(b=>b.key===selected);
 const counts=new Map<string,{id:number;n:number}>();passing.forEach(b=>{const c=counts.get(b.class_name)||{id:catalog.get(b.class_name)??b.class_id,n:0};c.n++;counts.set(b.class_name,c)});
 const [open,setOpenState]=useState<Set<string>>(()=>new Set(openGroupsMemory));
 const setOpen=(o:Set<string>)=>{openGroupsMemory=o;setOpenState(o)};
 const toggleGroup=(name:string)=>{const o=new Set(open);o.has(name)?o.delete(name):o.add(name);setOpen(o)};
 // picking a box on the image opens its class group, then scrolls the row into view
 // opened from an inventory link: select that box and zoom to it once the image's boxes are loaded
 useEffect(()=>{const b=focusKey&&loaded&&hist.present.find(x=>x.key===focusKey);if(b){setSelected(b.key);setTimeout(()=>focus(b),50)}},[loaded,!!space]);
 const scrollTo=useRef<string|null>(null);
 useEffect(()=>{if(!selected)return;const b=boxes.find(x=>x.key===selected);scrollTo.current=selected;if(b&&!open.has(b.class_name))setOpen(new Set(open).add(b.class_name))},[selected]);
 useEffect(()=>{const row=scrollTo.current&&list.current?.querySelector(`[data-key="${scrollTo.current}"]`);if(row){row.scrollIntoView({block:'nearest'});scrollTo.current=null}},[selected,open]);// once the row is rendered

 const push=(next:Box[])=>setHist(h=>({past:[...h.past.slice(-99),h.present],present:next,future:[]}));
 const undo=()=>setHist(h=>h.past.length?{past:h.past.slice(0,-1),present:h.past[h.past.length-1],future:[h.present,...h.future]}:h);
 const redo=()=>setHist(h=>h.future.length?{past:[...h.past,h.present],present:h.future[0],future:h.future.slice(1)}:h);
 const change=(key:string,f:(b:Box)=>Box)=>push(boxes.map(b=>b.key===key?edited(f(b)):b));
 const remove=(key:string)=>{push(boxes.filter(b=>b.key!==key));if(selected===key)setSelected(null)};
 const relabel=(name:string)=>{setDrawClass(name);if(sel)change(sel.key,b=>({...b,class_name:name,class_id:catalog.get(name)??b.class_id,...(vendors?.classes.includes(name)?{}:NO_VENDOR),...NO_CONDITION}))};// a new class invalidates the condition assessment
 const addClass=(e:React.FormEvent)=>{e.preventDefault();const n=newClass.trim();if(!n)return;if(!catalog.has(n))setExtra(x=>[...x,n]);setNewClass('');setDrawClass(n);if(sel)change(sel.key,b=>({...b,class_name:n,class_id:catalog.get(n)??Math.max(-1,...catalog.values())+1}))};
 const nudge=(dx:number,dy:number)=>sel&&space&&change(sel.key,b=>{const x=Math.min(Math.max(dx,-b.x1),space.w-b.x2),y=Math.min(Math.max(dy,-b.y1),space.h-b.y2);return {...b,x1:b.x1+x,x2:b.x2+x,y1:b.y1+y,y2:b.y2+y}});
 const zoomBy=(f:number,at?:{x:number;y:number})=>space&&setView(v=>{const c=at||{x:v.x+v.w/2,y:v.y+v.h/2},w=Math.min(Math.max(v.w/f,space.w/100),space.w*3),r=w/v.w;return {x:c.x-(c.x-v.x)*r,y:c.y-(c.y-v.y)*r,w,h:v.h*r}});
 const fit=()=>space&&setView({x:0,y:0,w:space.w,h:space.h});
 // zoom to a box, capped at 400% so tiny parts keep their context
 const focus=(b:Box)=>{if(!space)return;const w=Math.max((b.x2-b.x1)*3,(b.y2-b.y1)*3*space.w/space.h,size.w/4,size.h/4*space.w/space.h),h=w*space.h/space.w;setView({x:(b.x1+b.x2-w)/2,y:(b.y1+b.y2-h)/2,w,h});setSelected(b.key)};
 const toggleFull=()=>{if(document.fullscreenElement)document.exitFullscreen();else shell.current?.requestFullscreen().catch(()=>setError('Full screen is not available in this browser.'))};
 const zoomRef=useRef(zoomBy);zoomRef.current=zoomBy;
 const toImg=(e:{clientX:number;clientY:number})=>{const m=svg.current?.getScreenCTM();if(!m)return {x:0,y:0};const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(m.inverse());return {x:p.x,y:p.y}};
 const clampPt=(p:{x:number;y:number})=>space?{x:Math.min(Math.max(p.x,0),space.w),y:Math.min(Math.max(p.y,0),space.h)}:p;
 useEffect(()=>{const el=svg.current;if(!el)return;const h=(e:WheelEvent)=>{e.preventDefault();zoomRef.current(Math.pow(1.0015,-e.deltaY*(e.deltaMode===1?30:1)),toImg(e))};el.addEventListener('wheel',h,{passive:false});return()=>el.removeEventListener('wheel',h)},[!!space]);

 function begin(e:React.PointerEvent,mode:Drag['mode'],key?:string,handle?:string){const p=toImg(e);drag.current={mode,sx:e.clientX,sy:e.clientY,px:p.x,py:p.y,view,before:boxes,key,handle,moved:false};svg.current?.setPointerCapture(e.pointerId);if(mode==='pan')setPanning(true)}
 const onStageDown=(e:React.PointerEvent)=>{if(e.button===1||(e.button===0&&(spaceHeld.current||tool==='select'))){e.preventDefault();begin(e,'pan')}else if(e.button===0)begin(e,'draw')};
 const onBoxDown=(e:React.PointerEvent,b:Box)=>{if(e.button!==0||spaceHeld.current)return;e.stopPropagation();setSelected(b.key);begin(e,'move',b.key)};
 const onHandleDown=(e:React.PointerEvent,b:Box,h:string)=>{if(e.button!==0)return;e.stopPropagation();begin(e,'resize',b.key,h)};
 function onMove(e:React.PointerEvent){const p=toImg(e);if(tool==='draw')setCursor(clampPt(p));const d=drag.current;if(!d||!space)return;
  if(!d.moved&&Math.hypot(e.clientX-d.sx,e.clientY-d.sy)<3)return;d.moved=true;
  if(d.mode==='pan'){const kk=Math.max(d.view.w/size.w,d.view.h/size.h);setView({...d.view,x:d.view.x-(e.clientX-d.sx)*kk,y:d.view.y-(e.clientY-d.sy)*kk});return}
  const q=clampPt(p);
  if(d.mode==='draw'){const s=clampPt({x:d.px,y:d.py});setDraft({key:'draft',detection_id:null,class_name:activeClass,class_id:catalog.get(activeClass)??0,confidence:null,source:'manual',...NO_VENDOR,...NO_CONDITION,x1:Math.min(s.x,q.x),y1:Math.min(s.y,q.y),x2:Math.max(s.x,q.x),y2:Math.max(s.y,q.y)});return}
  const o=d.before.find(b=>b.key===d.key);if(!o)return;let n:Box;
  if(d.mode==='move'){const dx=Math.min(Math.max(p.x-d.px,-o.x1),space.w-o.x2),dy=Math.min(Math.max(p.y-d.py,-o.y1),space.h-o.y2);n={...o,x1:o.x1+dx,x2:o.x2+dx,y1:o.y1+dy,y2:o.y2+dy}}
  else{const h=d.handle||'',x1=h.includes('w')?q.x:o.x1,x2=h.includes('e')?q.x:o.x2,y1=h.includes('n')?q.y:o.y1,y2=h.includes('s')?q.y:o.y2;n={...o,x1:Math.min(x1,x2),x2:Math.max(x1,x2),y1:Math.min(y1,y2),y2:Math.max(y1,y2)};if(n.x2-n.x1<2||n.y2-n.y1<2)return}// keep the last valid size
  setHist(hs=>({...hs,present:d.before.map(b=>b.key===d.key?edited(n):b)}))}
 function onUp(e:React.PointerEvent){const d=drag.current;drag.current=null;setPanning(false);if(!d)return;
  if(d.mode==='pan'){if(!d.moved&&e.button===0&&tool==='select')setSelected(null);return}
  if(d.mode==='draw'){const b=draft;setDraft(null);if(b&&(b.x2-b.x1)/k>=4&&(b.y2-b.y1)/k>=4&&b.x2-b.x1>=2&&b.y2-b.y1>=2){const nb={...b,key:'n'+ ++seq.current};push([...boxes,nb]);setSelected(nb.key)}return}
  if(d.moved)setHist(hs=>hs.present===d.before?hs:{past:[...hs.past.slice(-99),d.before],present:hs.present,future:[]})}

 useEffect(()=>{// re-bound every render so the handlers see current state
  const down=(e:KeyboardEvent)=>{if((e.target as HTMLElement).closest?.('input,select,textarea'))return;const key=e.key,mod=e.ctrlKey||e.metaKey,step=e.shiftKey?10:1;
   if(mod&&key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?redo():undo();return}
   if(mod&&key.toLowerCase()==='y'){e.preventDefault();redo();return}
   if(mod||e.altKey)return;
   if(key===' '){spaceHeld.current=true;e.preventDefault();return}
   const act:Record<string,(()=>void)|undefined>={Delete:sel&&(()=>remove(sel.key)),Backspace:sel&&(()=>remove(sel.key)),Escape:()=>{setDraft(null);drag.current=null;setSelected(null)},v:()=>setTool('select'),b:()=>setTool('draw'),f:toggleFull,l:()=>setLabels(x=>!x),'+':()=>zoomBy(1.25),'=':()=>zoomBy(1.25),'-':()=>zoomBy(0.8),'0':fit,'[':nav?.prev,']':nav?.next,PageUp:nav?.prev,PageDown:nav?.next,
    ArrowLeft:sel&&(()=>nudge(-step,0)),ArrowRight:sel&&(()=>nudge(step,0)),ArrowUp:sel&&(()=>nudge(0,-step)),ArrowDown:sel&&(()=>nudge(0,step))};
   const f=act[key.length===1?key.toLowerCase():key];if(f){e.preventDefault();f()}};
  const up=(e:KeyboardEvent)=>{if(e.key===' ')spaceHeld.current=false};
  addEventListener('keydown',down);addEventListener('keyup',up);return()=>{removeEventListener('keydown',down);removeEventListener('keyup',up)}});

 async function approve(andNext:boolean){if(!space)return;setSaving(true);setError('');
  try{const r=await request<MediaAnnotations>(`/api/v1/media/${media.id}/annotations`,json('PUT',{workflow,job_id:jobRef.current,frame_index:0,image_width:Math.round(space.w),image_height:Math.round(space.h),notes,boxes:shown.map(({key,class_id,source,confidence,...b})=>b)}));
   const init=r.annotations.map(fromAnn);setLoaded(r);setHist({past:[],present:init,future:[]});setBaseline(init);setSelected(null);onSaved(r.review!);onDirty(false);if(andNext)nav?.next?.()}
  catch(e){setError(String(e))}finally{setSaving(false)}}

 // manufacturer tagging and local-VLM suggestions
 // condition assessment (all classes): AI screening + second-model check, reviewer can override
 const [assessing,setAssessing]=useState<{done:number;total:number;key:string}|null>(null),cancelAssess=useRef(false),[newDefect,setNewDefect]=useState({type:'rust',severity:'minor',extent:10,location:''});
 const setCond=(key:string,v:Partial<Box>,history=true,keepReviewer=false)=>{const f=(bs:Box[])=>bs.map(b=>b.key===key&&!(keepReviewer&&b.condition_source==='reviewer')?{...b,...v}:b);setHist(h=>history?{past:[...h.past.slice(-99),h.present],present:f(h.present),future:[]}:{...h,present:f(h.present)})};
 const fromCond=(r:ConditionResult):Partial<Box>=>({condition_grade:r.grade,condition_score:r.health_score,condition_defects:r.defects,condition_source:'ai',condition_confidence:r.confidence,condition_verified:r.verified||null,condition_summary:[r.summary,r.recommended_action&&`Action: ${r.recommended_action}`].filter(Boolean).join(' ')});
 const reviewCond=(b:Box,v:Partial<Box>)=>setCond(b.key,{condition_defects:b.condition_defects||[],condition_score:b.condition_score,...v,condition_source:'reviewer',condition_confidence:null,condition_verified:null});
 const askCond=(b:Box)=>request<ConditionResult>(`/api/v1/media/${media.id}/condition-assessment`,json('POST',{class_name:b.class_name,x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2,image_width:Math.round(space!.w),image_height:Math.round(space!.h)}));
 async function assessOne(b:Box){setAssessing({done:0,total:1,key:b.key});setError('');try{const r=await askCond(b);if(alive.current)setCond(b.key,fromCond(r))}catch(e){setError(String(e))}finally{if(alive.current)setAssessing(null)}}
 async function assessAll(todo:Box[]){if(!todo.length)return;cancelAssess.current=false;setError('');setHist(h=>({past:[...h.past.slice(-99),h.present],present:h.present,future:[]}));
  for(let i=0;i<todo.length&&!cancelAssess.current&&alive.current;i++){setAssessing({done:i,total:todo.length,key:todo[i].key});try{const r=await askCond(todo[i]);if(alive.current)setCond(todo[i].key,fromCond(r),false,true)}catch(e){setError(String(e));break}}
  if(alive.current)setAssessing(null)}
 const vendorClass=(b:Box)=>!!vendors?.classes.includes(b.class_name),ai=vendors?.ai,aiReady=ai?.status==='ready';
 const [asking,setAsking]=useState<{done:number;total:number;key:string}|null>(null),cancelAsk=useRef(false),alive=useRef(true);
 useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[]);// set on (re)mount too: StrictMode runs cleanup once in dev
 const setVendor=(key:string,v:Partial<Box>,history=true,keepReviewer=false)=>{const f=(bs:Box[])=>bs.map(b=>b.key===key&&!(keepReviewer&&b.vendor_source==='reviewer')?{...b,...v}:b);setHist(h=>history?{past:[...h.past.slice(-99),h.present],present:f(h.present),future:[]}:{...h,present:f(h.present)})};
 // when the AI can't decide between manufacturers, the reviewer picks one of its candidates (or another name)
 const [cands,setCands]=useState<Record<string,VendorCandidate[]>>({});
 const dropCands=(key:string)=>setCands(c=>{const n={...c};delete n[key];return n});
 const keepCands=(key:string,s:VendorSuggestion)=>s.candidates?.length>1?setCands(c=>({...c,[key]:s.candidates})):dropCands(key);
 const tagVendor=(key:string,field:'manufacturer'|'product_model',value:string)=>{if(field==='manufacturer')dropCands(key);setVendor(key,{[field]:value.trim()||null,vendor_source:'reviewer',vendor_confidence:null,vendor_evidence:null,vendor_basis:null})};
 const choose=(key:string,name:string)=>{const list=(cands[key]||[]).map(c=>`${c.manufacturer} ${Math.round(c.similarity*100)}%`).join(', ');dropCands(key);
  setVendor(key,{manufacturer:name,vendor_source:'reviewer',vendor_confidence:null,vendor_basis:'reviewer_choice',vendor_evidence:`Reviewer chose ${name} from AI candidates: ${list}.`})};
 const fromAI=(s:VendorSuggestion):Partial<Box>=>({manufacturer:s.manufacturer,product_model:s.product_model,vendor_source:'ai',vendor_confidence:s.confidence,vendor_evidence:s.evidence,vendor_basis:s.basis});
 const askVendor=(b:Box)=>request<VendorSuggestion>(`/api/v1/media/${media.id}/vendor-suggestion`,json('POST',{class_name:b.class_name,x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2,image_width:Math.round(space!.w),image_height:Math.round(space!.h)}));
 async function suggestOne(b:Box){setAsking({done:0,total:1,key:b.key});setError('');try{const s=await askVendor(b);if(alive.current){setVendor(b.key,fromAI(s));keepCands(b.key,s)}}catch(e){setError(String(e))}finally{if(alive.current)setAsking(null)}}
 const groups=[...shown.reduce((m,b)=>m.set(b.class_name,[...(m.get(b.class_name)||[]),b]),new Map<string,Box[]>())].map(([n,bs])=>[n,bs.sort((a,b)=>(b.confidence??2)-(a.confidence??2))] as [string,Box[]]).sort((a,b)=>b[1].length-a[1].length||a[0].localeCompare(b[0]));
 // reference library photos for the reviewer to compare against, fetched once per class
 const [refLib,setRefLib]=useState<Record<string,VendorRef[]>>({}),[refMaker,setRefMaker]=useState<string|null>(null);
 const refClass=sel&&vendorClass(sel)?sel.class_name:null;
 useEffect(()=>{if(!refClass||refLib[refClass])return;request<VendorRef[]>(`/api/v1/vendor-references?class_name=${encodeURIComponent(refClass)}`).then(r=>setRefLib(l=>({...l,[refClass]:r}))).catch(()=>{})},[refClass]);
 useEffect(()=>setRefMaker(null),[selected]);
 const refCount=(c?:string)=>c&&vendors?((vendors.folder_references[c]||0)+(vendors.reviewer_references[c]||0)):0;
 const needChoice=shown.filter(b=>(cands[b.key]?.length??0)>1&&b.vendor_source==='ai');
 const untagged=shown.filter(b=>vendorClass(b)&&b.vendor_source!=='reviewer');
 async function suggestAll(){const todo=untagged;if(!todo.length)return;cancelAsk.current=false;setError('');setHist(h=>({past:[...h.past.slice(-99),h.present],present:h.present,future:[]}));// one undo step for the whole run
  for(let i=0;i<todo.length&&!cancelAsk.current&&alive.current;i++){setAsking({done:i,total:todo.length,key:todo[i].key});try{const s=await askVendor(todo[i]);if(alive.current){setVendor(todo[i].key,fromAI(s),false,true);keepCands(todo[i].key,s)}}catch(e){setError(String(e));break}}
  if(alive.current)setAsking(null)}
 const aiTitle=aiReady?`Runs ${ai!.model} locally via Ollama`:`AI vendor model ${ai?.status.replaceAll('_',' ')||'unavailable'}${ai?.status==='model_missing'?` — run: ollama pull ${ai.model}`:ai?.status==='unreachable'?' — start Ollama':''}`;
 // a newer run finished while this image was open: refresh silently only when nothing would be lost
 const loadAI=()=>{push(detections.map(fromDet));jobRef.current=jobId};
 const newerRun=!!loaded&&jobId!=null&&jobId!==jobRef.current&&(dirty||!!review);
 useEffect(()=>{if(!loaded||jobId===jobRef.current||dirty||loaded.review)return;const init=detections.map(fromDet);setHist({past:[],present:init,future:[]});setBaseline(init);jobRef.current=jobId},[jobId,loaded]);
 const ordered=[...shown].sort((a,b)=>(b.x2-b.x1)*(b.y2-b.y1)-(a.x2-a.x1)*(a.y2-a.y1)).sort((a,b)=>+(a.key===selected)-+(b.key===selected));// big first so small parts stay clickable; selection on top
 const fs=12*k,hs=9*k,hiddenCount=boxes.length-shown.length,status=dirty?['fair','Unsaved changes']:review?['good','Approved']:['neutral','Not reviewed'];
 const label=(b:Box,c:string)=><text x={b.x1} y={b.y1>fs*1.3?b.y1-fs*0.35:b.y2+fs} fontSize={fs} fill={c} stroke="#000" strokeWidth={fs/5} paintOrder="stroke" fontWeight={700} pointerEvents="none">{b.class_name}{b.confidence!=null&&b.source==='model'?` ${(b.confidence*100).toFixed(0)}%`:''}{b.condition_grade&&ISSUE.has(b.condition_grade)?` · ${b.condition_grade}`:''}{b.manufacturer&&b.manufacturer!=='Unknown'?` · ${b.manufacturer}`:''}</text>;
 return <>
  <div className="viewer-middle"><div className="viewer-toolbar ann-toolbar">
    <div className="ann-tools ann-title">{nav&&<><T title="Previous image ( [ )" onClick={()=>nav.prev?.()} disabled={!nav.prev}><ChevronLeft size={16}/></T><em>{nav.position}</em><T title="Next image ( ] )" onClick={()=>nav.next?.()} disabled={!nav.next}><ChevronRight size={16}/></T></>}<b title={media.filename}>{media.filename}</b></div>
    <div className="ann-tools">
     <T title="Select / move / resize (V)" on={tool==='select'} onClick={()=>setTool('select')}><MousePointer2 size={15}/></T><T title="Draw box (B)" on={tool==='draw'} onClick={()=>setTool('draw')}><Square size={15}/></T><i className="ann-sep"/>
     <T title="Undo (Ctrl+Z)" onClick={undo} disabled={!hist.past.length}><Undo2 size={15}/></T><T title="Redo (Ctrl+Y)" onClick={redo} disabled={!hist.future.length}><Redo2 size={15}/></T><T title="Delete selected box (Del)" onClick={()=>sel&&remove(sel.key)} disabled={!sel}><Trash2 size={15}/></T><T title="Show all labels (L)" on={labels} onClick={()=>setLabels(!labels)}><Tag size={15}/></T><i className="ann-sep"/>
     <T title="Zoom out (−)" onClick={()=>zoomBy(0.8)} disabled={!space}><ZoomOut size={15}/></T><button className="ann-zoom" title="Fit to screen (0)" onClick={fit} disabled={!space}>{Math.round(100/k)}%</button><T title="Zoom in (+)" onClick={()=>zoomBy(1.25)} disabled={!space}><ZoomIn size={15}/></T><T title="Actual pixels (1:1)" onClick={()=>zoomBy(k)} disabled={!space}>1:1</T>
     <a className={dirty?'warn':''} title={`Download image with ${review?'approved':'AI'} boxes drawn, original metadata kept${dirty?' — unsaved edits are not included':''}`} aria-label="Download annotated image" href={`${API}/api/v1/media/${media.id}/annotated?source=${review?'approved':'model'}&min_conf=${review?0:minConf}`}><Download size={15}/></a>
     <T title={full?'Exit full screen (F)':'Full screen (F)'} on={full} onClick={toggleFull}>{full?<Minimize size={15}/>:<Maximize size={15}/>}</T></div></div>
   <div className="ann-stage" ref={stage}>
    {previewOnly&&<div style={{position:'absolute',left:10,bottom:10,zIndex:3,padding:'5px 9px',borderRadius:7,background:'rgba(255,247,230,.95)',color:'#966506',fontSize:11.5,pointerEvents:'none'}}>Showing the preview: the full-size photo could not be loaded (boxes are unaffected)</div>}
    {!space||!imageUrl?<div className="det-empty"><ScanEye size={40}/><p>{error||'Loading image…'}</p></div>
    :<svg ref={svg} viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} preserveAspectRatio="xMidYMid meet" style={{cursor:panning?'grabbing':tool==='draw'?'crosshair':'grab'}} onPointerDown={onStageDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onPointerLeave={()=>setCursor(null)} onDoubleClick={e=>{if(e.target===e.currentTarget||e.target instanceof SVGImageElement)fit()}}>
     <image href={imageUrl} width={space.w} height={space.h}/>
     {ordered.map(b=>{const c=colorOf(catalog.get(b.class_name)??b.class_id),isSel=b.key===selected,on=isSel||b.key===hover;return <g key={b.key} className="det-box" style={{pointerEvents:tool==='draw'?'none':undefined}} onPointerDown={e=>onBoxDown(e,b)} onDoubleClick={e=>{e.stopPropagation();focus(b)}} onMouseEnter={()=>setHover(b.key)} onMouseLeave={()=>setHover(null)}>
      <rect x={b.x1} y={b.y1} width={b.x2-b.x1} height={b.y2-b.y1} fill={on?c+'40':'transparent'} stroke={c} strokeWidth={isSel?3:on?2.5:1.5} vectorEffect="non-scaling-stroke" style={{cursor:tool==='select'?'move':undefined}}/>
      {(on||labels)&&label(b,c)}
      {isSel&&tool==='select'&&HANDLES.map(h=>{const x=h.includes('w')?b.x1:h.includes('e')?b.x2:(b.x1+b.x2)/2,y=h.includes('n')?b.y1:h.includes('s')?b.y2:(b.y1+b.y2)/2;return <rect key={h} x={x-hs/2} y={y-hs/2} width={hs} height={hs} fill="#fff" stroke={c} strokeWidth={1.5} vectorEffect="non-scaling-stroke" style={{cursor:h+'-resize'}} onPointerDown={e=>onHandleDown(e,b,h)}/>})}</g>})}
     {draft&&<g pointerEvents="none"><rect x={draft.x1} y={draft.y1} width={draft.x2-draft.x1} height={draft.y2-draft.y1} fill={colorOf(draft.class_id)+'30'} stroke={colorOf(draft.class_id)} strokeWidth={2} vectorEffect="non-scaling-stroke"/>{label(draft,colorOf(draft.class_id))}</g>}
     {tool==='draw'&&cursor&&!panning&&<g pointerEvents="none" stroke="#ffffffb0" strokeWidth={1} vectorEffect="non-scaling-stroke"><line x1={0} x2={space.w} y1={cursor.y} y2={cursor.y} vectorEffect="non-scaling-stroke"/><line x1={cursor.x} x2={cursor.x} y1={0} y2={space.h} vectorEffect="non-scaling-stroke"/></g>}
    </svg>}
    {space&&<div className="ann-hint">{tool==='draw'?`Drawing “${activeClass}” · drag to draw · hold Space to pan · V to select`:'Drag empty area to pan · wheel to zoom · double-click a box to zoom to it · B to draw'}</div>}
   </div></div>
  <div className="viewer-right ann-side"><div className="ann-scroll">
   <div className="panel-head"><h2>Annotations</h2><span className={'badge '+status[0]}>{status[1]}</span></div>
   {review&&<p className="muted ann-meta"><CheckCircle2 size={13}/>Approved {new Date(review.reviewed_at).toLocaleString()} via {review.workflow==='manual_qc'?'manual QC':'AI detection'} · {review.box_count} boxes</p>}
   {boxes.some(b=>b.confidence!=null)&&<label className="det-conf"><span>Min confidence <b>{minConf.toFixed(2)}</b></span><input type="range" min={0} max={1} step={0.05} value={minConf} onChange={e=>onMinConf(+e.target.value)}/></label>}
   {[...counts].sort((a,b)=>b[1].n-a[1].n).map(([name,{id,n}])=><button key={name} title={hidden.has(name)?'Show class':'Hide class'} className={'det-class '+(hidden.has(name)?'off':'')} onClick={()=>onToggleClass(name)}><i style={{background:colorOf(id)}}/><span>{name}</span><b>{n}</b></button>)}
   {shown.length>0&&(()=>{const todo=[...shown].filter(b=>!b.condition_grade).sort((a,b)=>(b.x2-b.x1)*(b.y2-b.y1)-(a.x2-a.x1)*(a.y2-a.y1)),issues=shown.filter(b=>b.condition_grade&&ISSUE.has(b.condition_grade)),disputed=shown.filter(b=>b.condition_verified==='disputed');
    return <div className="ann-batch cond">{assessing&&assessing.total>1?<><ScanEye size={14}/><span>Assessing condition… {assessing.done+1}/{assessing.total}</span><button className="text-btn" onClick={()=>{cancelAssess.current=true}}>Cancel</button></>
     :<button className="btn outline" disabled={!aiReady||!!assessing||!todo.length} title={aiReady?'Grade, health score and visible defects for every box not assessed yet; tiny boxes are skipped as too small':aiTitle} onClick={()=>assessAll(todo)}><ScanEye size={14}/>Assess condition ({todo.length})</button>}
     {(issues.length>0||disputed.length>0)&&<button className="text-btn ann-need" onClick={()=>setSelected((disputed[0]||issues[0]).key)}>{issues.length} with defects{disputed.length?` · ${disputed.length} disputed by 2nd model`:''} — review</button>}</div>})()}
   {vendors&&shown.some(vendorClass)&&<div className="ann-batch">{asking&&asking.total>1?<><Sparkles size={14}/><span>Identifying manufacturers… {asking.done+1}/{asking.total}</span><button className="text-btn" onClick={()=>{cancelAsk.current=true}}>Cancel</button></>
    :<button className="btn outline" disabled={!aiReady||!!asking||!untagged.length} title={aiTitle} onClick={suggestAll}><Sparkles size={14}/>Suggest manufacturers ({untagged.length})</button>}{!asking&&needChoice.length>0&&<button className="text-btn ann-need" onClick={()=>setSelected(needChoice[0].key)}>{needChoice.length} box{needChoice.length>1?'es need':' needs'} your choice — review next</button>}{!aiReady&&<small className="muted">{aiTitle}</small>}{aiReady&&<small className="muted">Checks logos/text, then structure against {refCount(shown.find(vendorClass)?.class_name)} reference photo(s){refCount(shown.find(vendorClass)?.class_name)?'':' — tag a few boxes by hand or add photos to the reference folder for structure matching'}.</small>}</div>}
   {!boxes.length&&loaded&&<p className="muted">No boxes on this image yet. Press <b>B</b> or pick the draw tool and drag on the image to add one.</p>}
   <div className="det-selected">
    <div className="tiny-label">{sel?'SELECTED BOX':'NEW BOXES'}</div>
    <label className="ann-field">{sel?'Class':'Class for new boxes'}<select value={sel?sel.class_name:activeClass} onChange={e=>relabel(e.target.value)}>{!catalog.has(activeClass)&&!sel&&<option value={activeClass}>{activeClass}</option>}{classList.map(([n,id])=><option key={n} value={n}>{id} · {n}</option>)}</select></label>
    <form className="ann-add" onSubmit={addClass}><input placeholder="New class name" value={newClass} onChange={e=>setNewClass(e.target.value)} maxLength={100}/><button className="btn outline" disabled={!newClass.trim()}><Plus size={14}/>Add</button></form>
    {sel&&<><div className="info"><small>Source</small><span>{SOURCE[sel.source]||sel.source}</span></div>{sel.confidence!=null&&<div className="info"><small>Model confidence</small><b>{(sel.confidence*100).toFixed(1)}%</b></div>}<div className="info"><small>Box (px)</small><span>{Math.round(sel.x1)}, {Math.round(sel.y1)} → {Math.round(sel.x2)}, {Math.round(sel.y2)}</span></div><div className="info"><small>Size</small><span>{Math.round(sel.x2-sel.x1)} × {Math.round(sel.y2-sel.y1)} px</span></div>
     <div className="ann-row"><button className="btn outline" onClick={()=>focus(sel)}><ZoomIn size={14}/>Zoom to box</button><button className="btn outline danger" onClick={()=>remove(sel.key)}><Trash2 size={14}/>Delete</button></div><p className="muted">Drag the box to move it, drag a corner or edge handle to resize, arrow keys nudge (Shift = 10 px).</p></>}
    {sel&&<div className="ann-cond-panel"><div className="tiny-label">CONDITION</div>
     <div className="ann-cond-head"><select aria-label="Condition grade" value={sel.condition_grade||''} onChange={e=>{const g=e.target.value||null;reviewCond(sel,{condition_grade:g,condition_score:g&&g!=='Unknown'?(sel.condition_score??GRADE_SCORE[g]):null})}}>
       <option value="">Not assessed</option>{GRADES.map(g=><option key={g} value={g}>{g}</option>)}</select>
      {sel.condition_grade&&<span className={'badge '+(sel.condition_grade==='Good'?'good':sel.condition_grade==='Fair'?'fair':sel.condition_grade==='Unknown'?'unknown':'poor')}>{sel.condition_grade}{sel.condition_score!=null?` · ${sel.condition_score}/100`:''}</span>}</div>
     {sel.condition_score!=null&&<div className="ann-health"><div style={{width:`${sel.condition_score}%`}} className={sel.condition_score>=80?'good':sel.condition_score>=55?'fair':'poor'}/></div>}
     {sel.condition_source==='ai'&&<div className={'ann-ai '+(sel.condition_verified==='disputed'?'disputed':'')}><b><ScanEye size={12}/>AI assessment{sel.condition_confidence!=null?` · ${Math.round(sel.condition_confidence*100)}%`:''}{sel.condition_verified==='confirmed'?' · ✓ confirmed by 2nd model':sel.condition_verified==='disputed'?' · ⚠ disputed by 2nd model':''}</b>
      {sel.condition_summary&&<span>{sel.condition_summary}</span>}<button className="text-btn" onClick={()=>reviewCond(sel,{})}><CheckCircle2 size={13}/>Confirm</button></div>}
     {sel.condition_source==='reviewer'&&<small className="muted">Set by reviewer</small>}
     {(sel.condition_defects||[]).length>0&&<div className="ann-defects">{sel.condition_defects!.map((d,i)=><div key={i} className={'ann-defect '+d.severity+(d.confirmed===false?' unconfirmed':'')} title={[d.evidence,d.confirmed===false?'Not seen by the 2nd model':d.confirmed?'Confirmed by the 2nd model':''].filter(Boolean).join(' — ')}>
       <b>{d.type.replaceAll('_',' ')}</b><span>{d.severity} · ~{d.extent_percent}%{d.location?` · ${d.location}`:''}</span><button aria-label="Remove defect" title="Remove defect" onClick={()=>reviewCond(sel,{condition_defects:sel.condition_defects!.filter((_,j)=>j!==i)})}><Trash2 size={12}/></button></div>)}</div>}
     <details className="ann-add-defect"><summary>Add defect</summary><div>
      <select aria-label="Defect type" value={newDefect.type} onChange={e=>setNewDefect({...newDefect,type:e.target.value})}>{DEFECT_TYPES.map(t=><option key={t} value={t}>{t.replaceAll('_',' ')}</option>)}</select>
      <select aria-label="Severity" value={newDefect.severity} onChange={e=>setNewDefect({...newDefect,severity:e.target.value})}>{['minor','moderate','severe'].map(s=><option key={s}>{s}</option>)}</select>
      <label>~<input type="number" min={0} max={100} value={newDefect.extent} onChange={e=>setNewDefect({...newDefect,extent:Math.max(0,Math.min(100,+e.target.value))})}/>%</label>
      <input placeholder="Location (optional)" value={newDefect.location} maxLength={120} onChange={e=>setNewDefect({...newDefect,location:e.target.value})}/>
      <button className="btn outline" onClick={()=>{const d:ConditionDefect={type:newDefect.type,severity:newDefect.severity,extent_percent:newDefect.extent,location:newDefect.location.trim(),evidence:'Added by reviewer',source:'reviewer'};
       const g=sel.condition_grade&&sel.condition_grade!=='Unknown'?sel.condition_grade:newDefect.severity==='severe'?'Poor':'Fair';reviewCond(sel,{condition_defects:[...(sel.condition_defects||[]),d],condition_grade:g,condition_score:sel.condition_score??GRADE_SCORE[g]});setNewDefect({...newDefect,location:''})}}><Plus size={13}/>Add</button></div></details>
     <button className="btn outline ann-suggest" disabled={!aiReady||!!assessing} title={aiReady?'Grade, health score and visible defects from the local vision model':aiTitle} onClick={()=>assessOne(sel)}><ScanEye size={14}/>{assessing?.key===sel.key?'Inspecting…':'Assess condition with AI'}</button></div>}
    {sel&&vendorClass(sel)&&<div className="ann-vendor"><div className="tiny-label">MANUFACTURER</div>
     <datalist id="vendor-options">{(vendors!.manufacturers[sel.class_name]||[]).map(m=><option key={m} value={m}/>)}</datalist>
     <label className="ann-field">Manufacturer<Commit key={`${sel.key}:m:${sel.manufacturer}`} value={sel.manufacturer||''} list="vendor-options" placeholder="Not tagged — pick or type" onCommit={v=>tagVendor(sel.key,'manufacturer',v)}/></label>
     <label className="ann-field">Product model<Commit key={`${sel.key}:p:${sel.product_model}`} value={sel.product_model||''} placeholder="e.g. AIR 6449 (optional)" onCommit={v=>tagVendor(sel.key,'product_model',v)}/></label>
     {(cands[sel.key]?.length??0)>1&&sel.vendor_source==='ai'&&<div className="ann-choice">
      <b>{sel.manufacturer==='Unknown'?'The AI could not decide. Which manufacturer is it?':'The AI also considered:'}</b>
      <div className="ann-choice-opts">{cands[sel.key].map(c=>{const also=c.brands.filter(x=>x!==c.manufacturer);return <button key={c.manufacturer} title={c.matching_features.join(', ')} onClick={()=>choose(sel.key,c.manufacturer)}><span>{c.manufacturer}{also.length>0&&<small> incl. {also.join(', ')}</small>}</span><em>{Math.round(c.similarity*100)}%</em></button>})}
       <button className="alt" onClick={()=>{setRefMaker(null);(document.querySelector('.ann-vendor input') as HTMLInputElement|null)?.focus()}}>Other…</button>
       <button className="alt" onClick={()=>choose(sel.key,'Unknown')}>Keep unknown</button></div>
      <small className="muted">Compare with the reference photos below, then pick. Your choice is saved as a reviewer tag.</small></div>}
     {sel.vendor_source==='ai'&&<div className="ann-ai"><b><Sparkles size={12}/>AI · {BASIS[sel.vendor_basis||'']||'suggestion'}{sel.vendor_confidence!=null?` · ${Math.round(sel.vendor_confidence*100)}%`:''}</b>{sel.vendor_evidence&&<span>{sel.vendor_evidence}</span>}<button className="text-btn" onClick={()=>setVendor(sel.key,{vendor_source:'reviewer',vendor_confidence:null,vendor_evidence:null,vendor_basis:null})}><CheckCircle2 size={13}/>Confirm</button></div>}
     {sel.vendor_source==='reviewer'&&sel.manufacturer&&<small className="muted">Tagged by reviewer</small>}
     <button className="btn outline ann-suggest" disabled={!aiReady||!!asking} title={aiTitle} onClick={()=>suggestOne(sel)}><Sparkles size={14}/>{asking?.key===sel.key?'Looking at the crop…':'Suggest with AI'}</button>
     {(()=>{const lib=refLib[sel.class_name]||[];if(!lib.length)return null;const makers=[...new Set(lib.map(r=>r.manufacturer))];
      const cur=refMaker&&makers.includes(refMaker)?refMaker:makers.find(m=>m.toLowerCase()===(sel.manufacturer||'').toLowerCase())||null;
      return <div className="ann-refs"><div className="tiny-label">REFERENCE PHOTOS</div><div className="ann-ref-makers">{makers.map(m=><button key={m} className={m===cur?'on':''} onClick={()=>setRefMaker(m===cur?'':m)}>{m} <small>{lib.filter(r=>r.manufacturer===m).length}</small></button>)}</div>
       {cur?<div className="ann-ref-grid">{lib.filter(r=>r.manufacturer===cur).map(r=><a key={r.url} href={API+r.url} target="_blank" rel="noreferrer" className={r.match?'':'dim'} title={`${r.manufacturer} ${r.model} · ${r.view.replaceAll('_',' ')}${r.match?'':' — not used by the AI (a drone never sees this side)'}${r.notes?' — '+r.notes:''}`}><img src={API+r.url} loading="lazy" alt={`${r.manufacturer} ${r.model} ${r.view}`}/><span>{r.model}</span><small>{r.view.replaceAll('_',' ')}{r.match?'':' · not used by AI'}</small></a>)}</div>
        :<small className="muted">Pick a manufacturer to compare its catalog photos with this box.</small>}</div>})()}
     </div>}
   </div>
   {shown.length>0&&<><div className="ann-list-head"><h4 className="det-sub">Boxes ({shown.length})</h4><button className="text-btn" onClick={()=>setOpen(groups.every(([n])=>open.has(n))?new Set():new Set(groups.map(([n])=>n)))}>{groups.every(([n])=>open.has(n))?'Collapse all':'Expand all'}</button></div>
   <div className="ann-list" ref={list}>{groups.map(([name,items])=>{const isOpen=open.has(name),issues=items.filter(b=>b.condition_grade&&ISSUE.has(b.condition_grade)).length,tagged=items.filter(b=>b.manufacturer&&b.manufacturer!=='Unknown').length;return <div key={name} className="ann-group">
    <button className={'ann-group-head '+(isOpen?'open':'')} aria-expanded={isOpen} onClick={()=>toggleGroup(name)}><ChevronRight size={14} className="chev"/><i style={{background:colorOf(catalog.get(name)??items[0].class_id)}}/><span>{name}</span>{issues>0&&<em className="ann-cond-tag poor" title="Boxes graded Fair, Poor or Critical">{issues} with defects</em>}{tagged>0&&<em className="ann-tag" title="Boxes with a manufacturer">{tagged} tagged</em>}<b>{items.length}</b></button>
    {isOpen&&items.map(b=><div key={b.key} data-key={b.key} className={'ann-item '+(b.key===selected?'on':'')} onClick={()=>setSelected(b.key)} onDoubleClick={()=>focus(b)} onMouseEnter={()=>setHover(b.key)} onMouseLeave={()=>setHover(null)}><i style={{background:colorOf(catalog.get(b.class_name)??b.class_id)}}/><span>{b.class_name}</span>{(cands[b.key]?.length??0)>1&&b.vendor_source==='ai'?<em className="ann-tag choose" title="AI found several candidates — pick one">choose</em>:b.manufacturer&&<em className={'ann-tag '+(b.vendor_source==='ai'?'ai':'')} title={b.vendor_source==='ai'?`AI suggestion · ${BASIS[b.vendor_basis||'']||''}`:'Reviewer tag'}>{b.manufacturer}</em>}{b.condition_grade&&ISSUE.has(b.condition_grade)&&<em className={'ann-cond-tag '+(b.condition_grade==='Fair'?'fair':'poor')} title={b.condition_summary||''}>{b.condition_grade}{b.condition_verified==='disputed'?' ?':''}</em>}<small>{b.source==='model'&&b.confidence!=null?`${(b.confidence*100).toFixed(0)}%`:SOURCE[b.source]}</small><button title="Delete box" aria-label="Delete box" onClick={e=>{e.stopPropagation();remove(b.key)}}><Trash2 size={13}/></button></div>)}</div>})}</div></>}
   {notice}
  </div>
  <div className="ann-foot">
   {newerRun&&<div className="ann-newer"><span>Newer AI detections are available for this image.</span><button className="text-btn" onClick={loadAI}><ScanEye size={13}/>Load them{dirty?' (your edits can be undone)':''}</button></div>}
   {hiddenCount>0&&<small className="ann-warn">{hiddenCount} box{hiddenCount>1?'es':''} hidden by the filters above won't be saved.</small>}
   {error&&<small className="det-error">{error}</small>}
   <input aria-label="Review note" placeholder="Review note (optional)" value={notes} onChange={e=>setNotes(e.target.value)}/>
   <div className="ann-row"><button className="btn primary" onClick={()=>approve(false)} disabled={saving||!space||!imageUrl||!loaded}><CheckCircle2 size={15}/>{saving?'Saving…':`${review?'Re-approve':'Approve'} · ${shown.length} box${shown.length===1?'':'es'}`}</button>{nav?.next&&<button className="btn outline" title="Approve and open the next image" onClick={()=>approve(true)} disabled={saving||!space||!imageUrl||!loaded}>& next<ChevronRight size={14}/></button>}</div>
   <div className="ann-row ann-links">{dirty&&<button className="text-btn" onClick={()=>push(baseline)}><RotateCcw size={13}/>Discard changes</button>}{review&&detections.length>0&&!newerRun&&<button className="text-btn" onClick={loadAI}><ScanEye size={13}/>Load AI output</button>}{boxes.length>0&&<button className="text-btn" onClick={()=>{push([]);setSelected(null)}}><Eraser size={13}/>Clear all</button>}</div>
  </div></div></>;
}
