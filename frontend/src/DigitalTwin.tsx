import {useEffect,useMemo,useRef,useState} from 'react';
import {AlertTriangle,Box,Camera,CheckCircle2,ChevronDown,Eye,EyeOff,Layers,Move3d,RadioTower,RotateCw,Wrench,XCircle,Maximize,Minimize,Ruler,Pentagon,Undo2,Trash2} from 'lucide-react';
import {Asset,Inspection,RegistrationCandidate,request,json,Site} from './api';
import Tower3D from './Tower3D';
import './measure-scale.css';
import './twin-classes.css';
import {MeasurementMode,MeasurementPoint,Model3D,ModelScale,ModelView,classColor} from './Reconstruction';

const insCode=(id:number)=>'INS-'+String(id).padStart(4,'0');
const fmt=(value?:number|null)=>value==null?'—':value.toLocaleString();
const reviewLabel=(status:string)=>status==='approved'?'Approved':status==='rejected'?'Rejected':status==='needs_adjustment'?'Needs adjustment':'Needs review';
const reviewBadge=(status:string)=>status==='approved'?'good':status==='rejected'?'poor':status==='needs_adjustment'?'fair':'neutral';
type MeasureUnit='mm'|'cm'|'m'|'km'|'in'|'ft'|'yd'|'mi';
const MEASURE_UNITS:{value:MeasureUnit;label:string;perMetre:number}[]=[{value:'mm',label:'Millimetres',perMetre:1000},{value:'cm',label:'Centimetres',perMetre:100},{value:'m',label:'Metres',perMetre:1},{value:'km',label:'Kilometres',perMetre:.001},{value:'in',label:'Inches',perMetre:39.3700787},{value:'ft',label:'Feet',perMetre:3.2808399},{value:'yd',label:'Yards',perMetre:1.0936133},{value:'mi',label:'Miles',perMetre:.000621371}];
const segment=(a:MeasurementPoint,b:MeasurementPoint)=>Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]);
const triangleArea=(a:MeasurementPoint,b:MeasurementPoint,c:MeasurementPoint)=>{const u=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],v=[c[0]-a[0],c[1]-a[1],c[2]-a[2]];return .5*Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])};
const prettyMeasure=(n:number)=>n.toLocaleString(undefined,{maximumFractionDigits:n>=1000?2:n>=1?3:5});

function MeasurementPanel({mode,points,unit,scale,scaleInfo,scaleBusy,scaleError,onDetect,onCalibrate,onMode,onUnit,onScale,onUndo,onClear,onClose}:{mode:MeasurementMode;points:MeasurementPoint[];unit:MeasureUnit;scale:number;scaleInfo:ModelScale|null;scaleBusy:string;scaleError:string;onDetect:()=>void;onCalibrate:(metresPerUnit:number,referenceMetres:number,modelUnits:number)=>void;onMode:(mode:MeasurementMode)=>void;onUnit:(unit:MeasureUnit)=>void;onScale:(scale:number)=>void;onUndo:()=>void;onClear:()=>void;onClose:()=>void}){
 const [known,setKnown]=useState('');
 const needed=mode==='distance'?2:4,complete=points.length===needed,raw=complete?(mode==='distance'?segment(points[0],points[1]):triangleArea(points[0],points[1],points[2])+triangleArea(points[0],points[2],points[3])):null;
 // without a known scale the model's units are not metres: show them as model units instead of mislabelling them
 const unscaled=!scaleInfo,selected=MEASURE_UNITS.find(u=>u.value===unit)!;
 const converted=raw==null?null:unscaled?raw:mode==='distance'?raw*scale*selected.perMetre:raw*scale*scale*selected.perMetre*selected.perMetre;
 const suffix=unscaled?(mode==='distance'?'model units':'model units²'):mode==='distance'?unit:`${unit}²`;
 return <div className="measurement-panel"><div className="measurement-head"><span><Ruler size={15}/><b>3D measurement</b></span><button type="button" onClick={onClose} title="Close measurement"><XCircle size={15}/></button></div>
  <div className="measurement-body"><div className="measurement-modes"><button className={mode==='distance'?'on':''} onClick={()=>onMode('distance')}><Ruler size={14}/>Distance</button><button className={mode==='area'?'on':''} onClick={()=>onMode('area')}><Pentagon size={14}/>4-point area</button></div>
   <div className={'measurement-result '+(complete?'complete':'')}><small>{complete?(mode==='distance'?'DISTANCE':'SURFACE AREA'):`SELECT POINTS · ${points.length}/${needed}`}</small><strong>{converted==null?`${needed-points.length} point${needed-points.length===1?'':'s'} remaining`:`${prettyMeasure(converted)} ${suffix}`}</strong>{converted!=null&&unscaled&&<em className="measurement-unscaled">Not to scale: detect or calibrate the scale to get real lengths</em>}<span>{mode==='distance'?'Click two points on the model.':'Click four corners in perimeter order; any quadrilateral shape is supported.'}</span></div>
   <div className="measurement-settings"><label><span>Display unit</span><select value={unit} disabled={unscaled} title={unscaled?'Available once the model has a real-world scale':undefined} onChange={e=>onUnit(e.target.value as MeasureUnit)}>{MEASURE_UNITS.map(u=><option key={u.value} value={u.value}>{u.label} ({u.value}{mode==='area'?'²':''})</option>)}</select></label><label><span>Scale calibration</span><div><em>1 model unit =</em><input type="number" min="0.000001" step="any" value={scale} onChange={e=>{const n=+e.target.value;if(Number.isFinite(n)&&n>0)onScale(n)}}/><b>m</b></div></label></div>
   {scaleInfo?<div className="measurement-scale-ok"><CheckCircle2 size={13}/><span>{scaleInfo.method==='manual'?`Scale calibrated by a reviewer${scaleInfo.reference_m?` from a known ${prettyMeasure(scaleInfo.reference_m)} m length`:''}.`
     :`Scale from the drone's ${scaleInfo.method==='gps_rtk'?'RTK ':''}GPS · ${scaleInfo.cameras??'?'} photos · cameras within ${((scaleInfo.residual_median_m??0)*100).toFixed(0)} cm${scaleInfo.uncertainty_pct!=null?` · ±${scaleInfo.uncertainty_pct<.1?'0.1':scaleInfo.uncertainty_pct.toFixed(1)}%`:''}.`}</span></div>
    :<div className="measurement-warning"><AlertTriangle size={13}/><span>{scaleBusy?scaleBusy:<>No real-world scale yet{scaleError?` (${scaleError})`:''}, so results are in model units. <button type="button" className="measurement-link" onClick={onDetect}>Detect from drone GPS</button> or measure a known length and calibrate.</>}</span></div>}
   {mode==='distance'&&complete&&raw!=null&&raw>0&&<form className="measurement-calibrate" onSubmit={e=>{e.preventDefault();const k=+known;if(k>0){const metres=k/selected.perMetre;onCalibrate(metres/raw,metres,raw);setKnown('')}}}>
    <span>Known real length of this distance?</span><input type="number" min="0" step="any" value={known} onChange={e=>setKnown(e.target.value)} placeholder={`e.g. 2.6`}/><b>{unit}</b><button type="submit" disabled={!(+known>0)}>Calibrate</button></form>}
   <div className="measurement-actions"><button disabled={!points.length} onClick={onUndo}><Undo2 size={13}/>Undo point</button><button disabled={!points.length} onClick={onClear}><Trash2 size={13}/>Clear</button></div>
  </div></div>;
}

type Props={site:Site|null;inspections:Inspection[];assets:Asset[];selected:Asset|null;onPick:(id:number)=>void;onOpenAssets:()=>void;readOnly?:boolean};

export default function DigitalTwin({site,inspections,assets,selected,onPick,onOpenAssets,readOnly=false}:Props){
 const [models,setModels]=useState<Model3D[]>([]),[selectedModel,setSelectedModel]=useState<number|null>(null);
 const [mode,setMode]=useState<'points'|'mesh'>('mesh'),[pointSize,setPointSize]=useState(.035),[showCams,setShowCams]=useState(false);
 const [loading,setLoading]=useState(false),[error,setError]=useState('');
 const [registrations,setRegistrations]=useState<RegistrationCandidate[]>([]),[selectedRegistration,setSelectedRegistration]=useState<number|null>(null),[focusRequest,setFocusRequest]=useState(0),[registering,setRegistering]=useState(false),[reviewing,setReviewing]=useState(false);
 const [editingRegistration,setEditingRegistration]=useState<number|null>(null),[editOriginal,setEditOriginal]=useState<{x:number;y:number;z:number;yaw:number|null}|null>(null),[editMode,setEditMode]=useState<'translate'|'rotate'>('translate');
 const [measurementMode,setMeasurementMode]=useState<MeasurementMode|null>(null),[measurementPoints,setMeasurementPoints]=useState<MeasurementPoint[]>([]),[measureUnit,setMeasureUnit]=useState<MeasureUnit>('m'),[metresPerModelUnit,setMetresPerModelUnit]=useState(1);
 const [scaleInfo,setScaleInfo]=useState<ModelScale|null>(null),[scaleBusy,setScaleBusy]=useState(''),[scaleError,setScaleError]=useState(''),autoDetected=useRef(new Set<number>());
 const siteInspections=useMemo(()=>inspections.filter(i=>i.site_id===site?.id),[inspections,site?.id]);

 useEffect(()=>{let live=true;setModels([]);setSelectedModel(null);setInsFilter(null);setError('');if(!siteInspections.length)return;
  setLoading(true);Promise.all(siteInspections.map(i=>request<Model3D[]>(`/api/v1/inspections/${i.id}/models`).catch(()=>[])))
   .then(groups=>{if(!live)return;const completed=groups.flat().filter(m=>m.status==='completed').sort((a,b)=>b.id-a.id);setModels(completed);setSelectedModel(completed[0]?.id??null)})
   .catch(e=>live&&setError(String(e))).finally(()=>live&&setLoading(false));return()=>{live=false}},
  [site?.id,siteInspections.map(i=>i.id).join(',')]);

 const [insFilter,setInsFilter]=useState<number|null>(null);  // an inspection picked in the viewer header: its latest model
 const model=insFilter!=null?(models.find(m=>m.id===selectedModel&&m.inspection_id===insFilter)||models.find(m=>m.inspection_id===insFilter))
  :models.find(m=>m.id===selectedModel)||models[0];
 // the model's saved real-world scale (GPS/RTK or a reviewer's calibration) drives every measurement; 1 = unscaled model units
 useEffect(()=>{const s=model?.quality?.scale??null;setScaleInfo(s);setMetresPerModelUnit(s?.m_per_unit||1);setScaleBusy('');setScaleError('')},[model?.id]);
 async function saveScale(body:{m_per_unit:number;reference_m?:number;reference_units?:number;note?:string}){if(!model)return;
  try{const s=await request<ModelScale>(`/api/v1/models/${model.id}/scale`,json('PUT',body));setScaleInfo(s);setMetresPerModelUnit(s.m_per_unit);model.quality.scale=s}catch(e){setError(String(e))}}
 async function detectScale(){if(!model)return;setScaleBusy('Detecting the scale from the drone photos’ GPS…');setScaleError('');
  try{const s=await request<ModelScale>(`/api/v1/models/${model.id}/scale/detect`,{method:'POST'});setScaleInfo(s);setMetresPerModelUnit(s.m_per_unit);model.quality.scale=s;setScaleBusy('')}
  catch(e){let m=String((e as Error).message||e);try{m=JSON.parse(m).detail}catch{}setScaleBusy('');setScaleError(String(m).replace(/^Cannot detect the scale: /,''))}}
 // opening Measure on an unscaled model tries the drone GPS once, so results come out in real units without an extra step
 useEffect(()=>{if(!measurementMode||!model||scaleInfo||autoDetected.current.has(model.id))return;autoDetected.current.add(model.id);detectScale()},[measurementMode,model?.id,scaleInfo]);
 const manualScale=useRef<number|undefined>(undefined);  // typed values are saved once typing pauses
 useEffect(()=>{if(manualScale.current==null||!model)return;const v=manualScale.current;const t=setTimeout(()=>{manualScale.current=undefined;saveScale({m_per_unit:v,note:'typed in the measurement panel'})},900);return()=>clearTimeout(t)},[metresPerModelUnit]);
 const inspection=siteInspections.find(i=>i.id===model?.inspection_id);
 const quality=model?.quality||{};
 const coverage=quality.registered_images!=null&&quality.images_total?quality.registered_images/quality.images_total:null;
 const realModel=!!model;
 useEffect(()=>{if(model)setMode(model.urls.web_mesh?'mesh':'points')},[model?.id]);
 useEffect(()=>{let live=true;setRegistrations([]);setSelectedRegistration(null);setEditingRegistration(null);setEditOriginal(null);setEditMode('translate');setMeasurementMode(null);setMeasurementPoints([]);if(!model)return;request<RegistrationCandidate[]>(`/api/v1/models/${model.id}/registrations`).then(r=>live&&setRegistrations(r)).catch(()=>{});return()=>{live=false}},[model?.id]);
 const candidate=registrations.find(r=>r.id===selectedRegistration)||null;
 // show/hide classes; the list is grouped by class
 const [hiddenClasses,setHiddenClasses]=useState<Set<string>>(new Set());
 // every unique class detected in the photos, with how many of each are placed in 3D (a class can be found in photos and not placed yet)
 const [detClasses,setDetClasses]=useState<{class_name:string;detections:number;photos:number}[]>([]);
 const [boxState,setBoxState]=useState<{state:string;error:string|null;boxes:number}|null>(null);
 useEffect(()=>{let live=true;setDetClasses([]);if(!model)return;request<{class_name:string;detections:number;photos:number}[]>(`/api/v1/inspections/${model.inspection_id}/detection-classes`).then(d=>live&&setDetClasses(d)).catch(()=>{});return()=>{live=false}},[model?.id]);
 const [pollKick,setPollKick]=useState(0);  // the button restarts the polling
 useEffect(()=>{setBoxState(null)},[model?.id]);
 useEffect(()=>{let live=true,timer=0,was='';if(!model)return;
  const look=()=>request<{state:string;error:string|null;boxes:number}>(`/api/v1/models/${model.id}/registrations/status`).then(s=>{if(!live)return;setBoxState(s);
   if(['queued','running'].includes(s.state)){was=s.state;timer=window.setTimeout(look,8000)}
   else if(was&&s.state==='done')request<RegistrationCandidate[]>(`/api/v1/models/${model.id}/registrations`).then(r=>live&&setRegistrations(r)).catch(()=>{})}).catch(()=>{});
  look();return()=>{live=false;clearTimeout(timer)}},[model?.id,pollKick]);
 const placing=registering||['queued','running'].includes(boxState?.state||'');
 const classCounts=useMemo(()=>{const c=new Map<string,number>();registrations.forEach(r=>c.set(r.class_name,(c.get(r.class_name)||0)+1));return [...c].sort((a,b)=>b[1]-a[1])},[registrations]);
 const classRows=useMemo(()=>{const boxes=new Map(classCounts),rows=detClasses.map(d=>({name:d.class_name,boxes:boxes.get(d.class_name)||0,detections:d.detections,photos:d.photos}));
  for(const [n,k] of classCounts)if(!detClasses.some(d=>d.class_name===n))rows.push({name:n,boxes:k,detections:0,photos:0});
  return rows.sort((a,b)=>b.boxes-a.boxes||b.detections-a.detections)},[detClasses,classCounts]);
 const visibleRegs=registrations.filter(r=>!hiddenClasses.has(r.class_name));
 const visibleMarkers=visibleRegs.filter(r=>r.status!=='rejected'),reviewableVisible=visibleRegs.filter(r=>!['approved','rejected'].includes(r.status));
 const approvedCount=registrations.filter(r=>r.status==='approved').length;
 const toggleClass=(n:string)=>setHiddenClasses(h=>{const s=new Set(h);s.has(n)?s.delete(n):s.add(n);return s});
 const focusRegistration=(id:number)=>{if(editingRegistration!=null&&editingRegistration!==id)cancelAdjustment();setSelectedRegistration(id);setFocusRequest(v=>v+1)};
 async function register(){if(!model)return;setRegistering(true);setError('');try{const r=await request<{candidates?:RegistrationCandidate[];status?:string}>(`/api/v1/models/${model.id}/register-assets?background=true`,{method:'POST'});
  if(r.candidates){setRegistrations(r.candidates);setSelectedRegistration(r.candidates[0]?.id??null)}else{setBoxState(s=>({state:'queued',error:null,boxes:s?.boxes??0}));setPollKick(k=>k+1)}}catch(e){setError(String(e))}finally{setRegistering(false)}}
 async function review3d(candidateIds:number[],status:'approved'|'rejected'|'needs_adjustment'|'needs_review'){if(!model||!candidateIds.length)return false;setReviewing(true);setError('');try{const r=await request<{candidates:RegistrationCandidate[]}>(`/api/v1/models/${model.id}/registrations/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_ids:candidateIds,status})});setRegistrations(r.candidates);if(status!=='needs_adjustment'&&editingRegistration!=null&&candidateIds.includes(editingRegistration)){setEditingRegistration(null);setEditOriginal(null)}return true}catch(e){setError(String(e));return false}finally{setReviewing(false)}}
 async function beginAdjustment(row:RegistrationCandidate){if(!await review3d([row.id],'needs_adjustment'))return;setMeasurementMode(null);setMeasurementPoints([]);setEditOriginal({x:row.x,y:row.y,z:row.z,yaw:row.yaw??0});setEditMode('translate');setEditingRegistration(row.id);setSelectedRegistration(row.id);setFocusRequest(v=>v+1)}
 function updateDraft(id:number,values:Partial<Pick<RegistrationCandidate,'x'|'y'|'z'|'yaw'>>){setRegistrations(rows=>rows.map(row=>row.id===id?{...row,...values}:row))}
 function transformDraft(id:number,position:[number,number,number],yaw:number){updateDraft(id,{x:position[0],z:position[1],y:position[2],yaw})}
 function cancelAdjustment(){if(editingRegistration!=null&&editOriginal)updateDraft(editingRegistration,editOriginal);setEditingRegistration(null);setEditOriginal(null)}
 async function saveAdjustment(){if(!candidate||candidate.id!==editingRegistration)return;setReviewing(true);setError('');try{const saved=await request<RegistrationCandidate>(`/api/v1/registrations/${candidate.id}/geometry`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({x:candidate.x,y:candidate.y,z:candidate.z,yaw:candidate.yaw??0})});setRegistrations(rows=>rows.map(row=>row.id===saved.id?saved:row));setEditingRegistration(null);setEditOriginal(null)}catch(e){setError(String(e))}finally{setReviewing(false)}}
 const chooseMeasurementMode=(next:MeasurementMode)=>{setShowCams(false);setMeasurementMode(next);setMeasurementPoints([])};
 const addMeasurementPoint=(point:MeasurementPoint)=>setMeasurementPoints(current=>{const maximum=measurementMode==='area'?4:2;return current.length>=maximum?[point]:[...current,point]});

 // full screen covers the whole workspace (explorer, 3D view, inspector); Esc or the button leaves it
 const shell=useRef<HTMLDivElement>(null),[full,setFull]=useState(false);
 useEffect(()=>{const h=()=>setFull(document.fullscreenElement===shell.current);h();document.addEventListener('fullscreenchange',h);return()=>document.removeEventListener('fullscreenchange',h)},[]);
 const toggleFull=()=>{if(document.fullscreenElement)document.exitFullscreen();else shell.current?.requestFullscreen().catch(()=>setError('Full screen is not available in this browser.'))};

 return <div ref={shell} className="viewer-shell recon-shell twin-shell">
  <div className="viewer-left twin-explorer">
   <div className="twin-explorer-head"><div className="twin-explorer-mark"><RadioTower size={17}/></div><div><span>SCENE EXPLORER</span><h3>{site?.code||'Select a site'}</h3><small>{site?.name||'No tower selected'}</small></div>{realModel&&<i title="Model connected"/>}</div>

   <section className="twin-scene-card">
    <div className="twin-layer"><span><Layers size={14}/></span><div><b>Geometry</b><small>{realModel?'Reconstructed model':'Tower structure'}</small></div><em>Visible</em></div>
    <div className="twin-layer"><span><Box size={14}/></span><div><b>Surface</b><small>{realModel?(model.dense?'Dense mesh':'Sparse points'):'Demonstration'}</small></div><em>On</em></div>
    {realModel&&<button className="twin-layer" onClick={()=>setShowCams(!showCams)}><span><Camera size={14}/></span><div><b>Drone cameras</b><small>Source positions</small></div><em className={showCams?'active':''}>{showCams?'On':'Off'}</em></button>}
   </section>

   <details className="twin-explorer-section">
    <summary><span>Inspection model</span><b>{models.length}</b><ChevronDown size={14}/></summary>
    <div className="twin-section-body">
     {models.length>0&&<label className="twin-model-select"><span>Active reconstruction</span><select value={model?.id??''} onChange={e=>{const id=+e.target.value;setSelectedModel(id);setInsFilter(models.find(m=>m.id===id)?.inspection_id??null)}}>{models.map(m=>{const i=siteInspections.find(x=>x.id===m.inspection_id);return <option key={m.id} value={m.id}>Model #{m.id} · {m.dense?'Dense':'Sparse'} · {i?.label||insCode(m.inspection_id)}</option>})}</select></label>}
     {model&&<div className="twin-model-meta"><span><CheckCircle2 size={13}/>Ready</span><small>{insCode(model.inspection_id)} · {quality.registered_images??'—'}/{quality.images_total??'—'} images</small></div>}
     {loading&&<p className="muted">Loading reconstruction…</p>}
     {!loading&&!realModel&&<p className="muted">No completed reconstruction for this tower.</p>}
     {error&&<p className="det-error">{error}</p>}
    </div>
   </details>

   {assets.length>0&&<details className="twin-explorer-section">
    <summary><span>Asset register</span><b>{assets.length}</b><ChevronDown size={14}/></summary>
    <div className="twin-section-body twin-scroll-list">{assets.map(a=><button key={a.id} className={'tree-asset '+(selected?.id===a.id?'chosen':'')} onClick={()=>onPick(a.id)}><i className={'dot '+a.condition.toLowerCase()}/><div><b>{a.code}</b><small>{a.category}</small></div></button>)}</div>
   </details>}

   {realModel&&<details className="twin-explorer-section twin-equipment-section" open>
    <summary><span>3D equipment</span><b>{visibleRegs.length}/{registrations.length}</b><ChevronDown size={14}/></summary>
    <div className="twin-section-body">
     <div className="twin-equipment-summary"><div><strong>{registrations.length}</strong><small>Detected</small></div><div><strong>{approvedCount}</strong><small>Approved</small></div><div><strong>{registrations.length-approvedCount}</strong><small>To review</small></div></div>
     {placing?<div className="twin-placing"><span className="twin-spinner"/>Placing the photo detections in 3D…<small>{detClasses.reduce((a,d)=>a+d.detections,0).toLocaleString()} detections · this takes a few minutes for a large flight</small></div>
      :boxState?.state==='failed'?<div className="det-error">Placing in 3D failed: {boxState.error}</div>
      :!registrations.length&&detClasses.length>0&&<div className="twin-placing idle">{detClasses.length} classes were detected in the photos and are not placed in 3D yet.</div>}
     {!readOnly&&detClasses.length>0&&!placing&&<button className="btn primary wide" onClick={register}><Box size={14}/>{registrations.length?'Rebuild 3D boxes from the detections':'Show the detections in 3D'}</button>}
     {classRows.length>0&&<><div className="twin-classes-head"><span>Detected classes · {classRows.length}</span>{registrations.length>0&&<button onClick={()=>setHiddenClasses(h=>h.size?new Set():new Set(classRows.map(r=>r.name)))}>{hiddenClasses.size?'Show all':'Hide all'}</button>}</div>
      <div className="twin-classes">{classRows.map(r=><button key={r.name} className={'det-class '+(hiddenClasses.has(r.name)?'off':'')+(r.boxes?'':' unplaced')} disabled={!r.boxes} title={!r.boxes?'Found in the photos, not placed in 3D yet':hiddenClasses.has(r.name)?'Show class':'Hide class'} onClick={()=>toggleClass(r.name)}>
       <i style={{background:classColor(r.name)}}/><span>{r.name.replaceAll('_',' ')}<small>{r.boxes?`${r.boxes} in 3D`:'not in 3D'}{r.detections?` · ${r.detections.toLocaleString()} in ${r.photos.toLocaleString()} photos`:''}</small></span><b>{r.boxes}</b>{r.boxes?(hiddenClasses.has(r.name)?<EyeOff size={12}/>:<Eye size={12}/>):null}</button>)}</div></>}
     {!readOnly&&reviewableVisible.length>0&&<button className="btn primary wide registration-approve-all" disabled={reviewing} onClick={()=>review3d(reviewableVisible.map(r=>r.id),'approved')}><CheckCircle2 size={14}/>{reviewing?'Saving…':`Approve visible (${reviewableVisible.length})`}</button>}
     {visibleRegs.length>0&&<details className="twin-box-browser"><summary><span>Review individual boxes</span><b>{visibleRegs.length}</b><ChevronDown size={13}/></summary><div className="twin-scroll-list">{[...visibleRegs].sort((a,b)=>a.class_name.localeCompare(b.class_name)||b.confidence-a.confidence).map(r=><button key={r.id} className={'tree-asset '+(candidate?.id===r.id?'chosen ':'')+(r.status==='rejected'?'rejected':'')} onClick={()=>focusRegistration(r.id)}><i className="registration-dot" style={{background:classColor(r.class_name)}}/><div><b>{r.class_name.replaceAll('_',' ')}</b><small>{r.media_count} images · {Math.round(r.confidence*100)}% · {reviewLabel(r.status)}</small></div></button>)}</div></details>}
    </div>
   </details>}

   {!assets.length&&<div className="twin-explorer-foot"><span>Asset register</span><small>No canonical assets yet</small></div>}
  </div>

  <div className="viewer-middle"><div className="viewer-toolbar ann-toolbar"><div className="ann-tools ann-title"><span className="live-dot"/><b>{site?.code||'3D tower workspace'}</b>
    {siteInspections.length>0&&<select aria-label="Inspection" title="Inspection" value={insFilter??model?.inspection_id??''} onChange={e=>{const id=+e.target.value;setInsFilter(id);setSelectedModel(models.find(m=>m.inspection_id===id)?.id??null)}} style={{background:'transparent',color:'inherit',border:'1px solid #2d5878',borderRadius:6,padding:'2px 6px',fontSize:12,fontWeight:600,maxWidth:260}}>
     {siteInspections.map(i=><option key={i.id} value={i.id} style={{color:'#1d2f40'}}>{insCode(i.id)} · {i.label}{models.some(m=>m.inspection_id===i.id)?'':' (no 3D model)'}</option>)}</select>}
    {realModel&&<b>Model #{model.id}</b>}</div>
   {realModel?<div className="ann-tools"><button className={mode==='points'?'on':''} onClick={()=>setMode('points')}><Layers size={15}/>Points</button><button className={mode==='mesh'?'on':''} onClick={()=>setMode('mesh')} disabled={!model.urls.web_mesh}><Box size={15}/>Mesh</button><i className="ann-sep"/><label className="recon-size">size<input type="range" min={.005} max={.12} step={.005} value={pointSize} onChange={e=>setPointSize(+e.target.value)}/></label><button className={showCams?'on':''} onClick={()=>setShowCams(!showCams)} title="Drone camera positions"><Camera size={15}/></button><i className="ann-sep"/><button className={measurementMode?'on':''} disabled={editingRegistration!=null} onClick={()=>measurementMode?(setMeasurementMode(null),setMeasurementPoints([])):chooseMeasurementMode('distance')} title="Measure distance or area on the 3D model"><Ruler size={15}/>Measure</button></div>:<span>Orbit · Pan · Zoom</span>}<div className="ann-tools"><button className={full?'on':''} title={full?'Exit full screen (Esc)':'Full screen'} onClick={toggleFull}>{full?<Minimize size={15}/>:<Maximize size={15}/>}</button></div>
  </div>
   {realModel?<div className={'recon-stage '+(measurementMode?'measurement-active':'')}><ModelView key={model.id} model={model} mode={mode} pointSize={pointSize} showCams={showCams} markers={visibleMarkers} selectedMarker={selectedRegistration} editingMarker={editingRegistration} markerEditMode={editMode} focusRequest={focusRequest} onMarkerPick={focusRegistration} onMarkerTransform={transformDraft} measurementMode={measurementMode} measurementPoints={measurementPoints} onMeasurementPoint={addMeasurementPoint}/>{measurementMode&&<MeasurementPanel mode={measurementMode} points={measurementPoints} unit={measureUnit} scale={metresPerModelUnit} scaleInfo={scaleInfo} scaleBusy={scaleBusy} scaleError={scaleError} onDetect={detectScale} onCalibrate={(mpu,ref,units)=>saveScale({m_per_unit:mpu,reference_m:ref,reference_units:units,note:'calibrated from a known length'})} onMode={chooseMeasurementMode} onUnit={setMeasureUnit} onScale={n=>{manualScale.current=n;setMetresPerModelUnit(n)}} onUndo={()=>setMeasurementPoints(p=>p.slice(0,-1))} onClear={()=>setMeasurementPoints([])} onClose={()=>{setMeasurementMode(null);setMeasurementPoints([])}}/>}<div className="ann-hint">{measurementMode?`Measurement active · click ${Math.max(0,(measurementMode==='distance'?2:4)-measurementPoints.length)} more surface point${Math.max(0,(measurementMode==='distance'?2:4)-measurementPoints.length)===1?'':'s'} · drag empty space to orbit`:editingRegistration!=null?(editMode==='translate'?'Drag an axis to move the selected box · switch to Rotate to change its facing':'Drag a yellow outside corner handle to rotate the box · switch to Move to reposition it'):'Click a list item or coloured box to focus · drag to orbit · right-drag to pan · wheel to zoom'}{showCams&&!measurementMode?' · click a yellow camera point to view its image':''}</div></div>:<Tower3D assets={assets} selected={selected?.id||null} onPick={onPick}/>}</div>

  <div className="viewer-right"><div className="panel-head"><h2>{candidate?'Registration Candidate':selected?'Asset Inspector':'Digital Twin Model'}</h2>{candidate?<span className={'badge '+reviewBadge(candidate.status)}>{reviewLabel(candidate.status)}</span>:realModel&&!selected&&<span className="badge good">linked</span>}</div>
   {candidate?<>
    <div className="twin-candidate-icon"><RadioTower size={38}/></div><div className="tiny-label">3D REGISTRATION · {reviewLabel(candidate.status).toUpperCase()}</div><h2>{candidate.class_name.replaceAll('_',' ')}</h2><p>Candidate #{candidate.id} from COLMAP feature tracks inside YOLO boxes.</p>
    <div className="info"><small>Confidence</small><b>{Math.round(candidate.confidence*100)}%</b></div><div className="info"><small>Supporting images</small><b>{candidate.media_count}</b></div><div className="info"><small>2D observations</small><b>{candidate.evidence_count}</b></div><div className="info"><small>3D feature points</small><b>{fmt(candidate.point_count)}</b></div>
    {editingRegistration===candidate.id?<div className="registration-coordinate-editor"><div className="registration-transform-mode"><button className={editMode==='translate'?'on':''} onClick={()=>setEditMode('translate')}><Move3d size={13}/>Move</button><button className={editMode==='rotate'?'on':''} onClick={()=>setEditMode('rotate')}><RotateCw size={13}/>Rotate</button></div>{editMode==='translate'?<><small>Drag the 3D axis handles or enter coordinates</small><div>{(['x','y','z'] as const).map(axis=><label key={axis}>{axis.toUpperCase()}<input type="number" step="0.01" value={candidate[axis]} onChange={e=>{const value=Number(e.target.value);if(Number.isFinite(value))updateDraft(candidate.id,{[axis]:value})}}/></label>)}</div></>:<><small>Drag any yellow handle outside the box edges or enter the vertical rotation</small><div className="registration-rotation-input"><label>Rotation<input type="number" step="1" value={((candidate.yaw??0)*180/Math.PI).toFixed(1)} onChange={e=>{const degrees=Number(e.target.value);if(Number.isFinite(degrees))updateDraft(candidate.id,{yaw:degrees*Math.PI/180})}}/></label><span>degrees</span></div></>}</div>:<><div className="info"><small>Position X / Y / Z</small><b>{candidate.x.toFixed(2)} / {candidate.y.toFixed(2)} / {candidate.z.toFixed(2)}</b></div><div className="info"><small>Vertical rotation</small><b>{Math.round((candidate.yaw??0)*180/Math.PI)}°</b></div></>}
    {candidate.size_w!=null&&<div className="info"><small>Box W × H × D</small><b>{candidate.size_w.toFixed(2)} × {candidate.size_h?.toFixed(2)} × {candidate.size_d?.toFixed(2)}</b></div>}{candidate.fit_iou!=null&&<div className="info"><small>Projection agreement</small><b>{Math.round(candidate.fit_iou*100)}%</b></div>}{candidate.box_method&&<div className="info"><small>Box fitted from</small><b>{candidate.box_method==='dense'?`${fmt(candidate.box_points)} dense points`:'photo estimate'}</b></div>}
    {!readOnly&&(editingRegistration===candidate.id?<div className="registration-review-actions"><button className="btn primary" disabled={reviewing} onClick={saveAdjustment}><CheckCircle2 size={14}/>Save position</button><button className="btn outline" disabled={reviewing} onClick={cancelAdjustment}><XCircle size={14}/>Cancel</button></div>:<div className="registration-review-actions"><button className="btn primary" disabled={reviewing||candidate.status==='approved'} onClick={()=>review3d([candidate.id],'approved')}><CheckCircle2 size={14}/>Approve</button><button className="btn outline" disabled={reviewing} onClick={()=>beginAdjustment(candidate)}><Wrench size={14}/>{candidate.status==='needs_adjustment'?'Move box':'Needs adjustment'}</button><button className="btn outline danger" disabled={reviewing||candidate.status==='rejected'} onClick={()=>review3d([candidate.id],'rejected')}><XCircle size={14}/>Reject</button></div>)}
    {readOnly&&<div className="notice"><Eye size={15}/>Read-only access. An Inspector can approve or adjust 3D boxes.</div>}
    <div className="notice"><AlertTriangle size={15}/>{editingRegistration===candidate.id?(editMode==='translate'?'Move the box with the red X, green Y and blue Z handles, then save.':'Rotate the box by dragging a yellow handle on the outside frame, then save.'):candidate.status==='approved'?'Approved boxes are locked and preserved during 3D rebuilds.':candidate.status==='rejected'?'Rejected boxes are hidden from the 3D scene and will not return during rebuilds.':candidate.status==='needs_adjustment'?'This box is locked for manual adjustment. Click Move box to reposition or rotate it, then approve it after verification.':'Verify the box against the mesh, then approve it or mark it for adjustment.'}</div></>
   :selected?<><div className="asset-big-icon"><RadioTower size={50}/></div><div className="tiny-label">CANONICAL ASSET</div><h2>{selected.code}</h2><p>{selected.category}</p><div className="info"><small>Condition</small><span className={'badge '+selected.condition.toLowerCase()}>{selected.condition}</span></div><div className="info"><small>Identity status</small><b>{selected.identity_status}</b></div><div className="info"><small>3D localization</small><b>{selected.localization_status}</b></div><div className="info"><small>Tower-relative height</small><b>{selected.z??'—'} m</b></div><button className="btn primary wide" onClick={onOpenAssets}>Open asset register</button></>
   :realModel?<><div className="twin-model-icon"><Box size={42}/></div><div className="tiny-label">RECONSTRUCTED DIGITAL TWIN</div><h2>{site?.code} · {site?.name}</h2><p>{inspection?`${insCode(inspection.id)} · ${inspection.label}`:'Linked inspection'}</p>
    <div className="info"><small>Source images</small><b>{quality.images_total??'—'}</b></div><div className="info"><small>Images registered</small><b>{quality.registered_images??'—'} / {quality.images_total??'—'}</b></div><div className="info"><small>Coverage</small><b>{coverage==null?'—':`${Math.round(coverage*100)}%`}</b></div><div className="info"><small>Model type</small><b>{model.dense?'Dense mesh':'Sparse point cloud'}</b></div><div className="info"><small>Dense points</small><b>{fmt(quality.dense_points)}</b></div><div className="info"><small>Mesh faces</small><b>{fmt(quality.mesh_faces)}</b></div><div className="info"><small>Reprojection error</small><b>{quality.mean_reprojection_error_px==null?'—':`${quality.mean_reprojection_error_px.toFixed(2)} px`}</b></div><div className="info"><small>Canonical assets</small><b>{assets.length}</b></div>
    {!readOnly&&<button className="btn primary wide" onClick={register} disabled={registering}>{registering?'Registering equipment…':registrations.length?'Rebuild 3D boxes':'Generate 3D boxes'}</button>}<div className="notice"><AlertTriangle size={15}/>{readOnly?'Read-only access. An Inspector can generate or rebuild equipment boxes.':'Geometry is linked but not scaled or georeferenced. Registration output remains review-only.'}</div></>:<div className="empty"><Box size={38}/><p>Build a model in 3D Reconstruction to connect it to this Digital Twin.</p></div>}
  </div>
 </div>;
}
