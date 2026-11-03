import {useEffect,useMemo,useRef,useState} from 'react';
import {AlertTriangle,Archive,ArchiveRestore,ArrowRight,Box,Calendar,ClipboardCheck,Edit3,ExternalLink,Image as ImageIcon,Layers,MapPin,Plus,RadioTower,ScanEye,Search,ShieldAlert,Trash2,UploadCloud,Video,X} from 'lucide-react';
import {Site,request,json} from './api';
import './sites.css';

type Stats={inspections:number;last_inspection_at:string|null;photos:number;videos:number;images:number;approved_images:number;items:number;items_with_defects:number;poor_or_critical:number;models:number;dense_models:number;boxes_3d:number;jobs_running:number};
type SiteRow=Site&{stats:Stats};
type InspectionRow={id:number;label:string;status:string;created_at:string;photos:number;videos:number;images:number;approved_images:number;detections:number;items_with_defects:number;models:{id:number;dense:boolean|null;status:string}[];latest_job:{id:number;kind:string;status:string;progress:number;detail:string}|null};
type Form={code:string;name:string;region:string;operator:string;tower_type:string;height_m:string;latitude:string;longitude:string;address:string;notes:string;status:string};

const STRUCTURES=['Lattice','Monopole','Guyed mast','Rooftop','Camouflaged','Other'];
const STATUSES=[{id:'active',label:'Active'},{id:'maintenance',label:'Maintenance'},{id:'decommissioned',label:'Decommissioned'},{id:'archived',label:'Archived'}];
const statusLabel=(s:string)=>STATUSES.find(x=>x.id===s)?.label||s;
const fmtDate=(d?:string|null)=>d?new Date(d).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'}):'—';
const insCode=(id:number)=>`INS-${String(id).padStart(4,'0')}`;
/** API errors arrive as the response text: show FastAPI's "detail" (or the first validation message) instead of raw JSON. */
function errText(e:unknown){const m=String((e as Error)?.message??e);try{const d=JSON.parse(m).detail;return Array.isArray(d)?d.map((x:{loc?:string[];msg:string})=>`${x.loc?.slice(-1)[0]??''}: ${x.msg}`).join('; '):String(d)}catch{return m}}

const emptyForm:Form={code:'',name:'',region:'',operator:'',tower_type:'Lattice',height_m:'',latitude:'',longitude:'',address:'',notes:'',status:'active'};
const toForm=(s:Site):Form=>({code:s.code,name:s.name,region:s.region==='Unknown'?'':s.region,operator:s.operator||'',tower_type:s.tower_type,height_m:s.height_m?.toString()??'',
 latitude:s.latitude?.toString()??'',longitude:s.longitude?.toString()??'',address:s.address||'',notes:s.notes||'',status:s.status});

/** Field-level checks before anything is sent; the server repeats them (and checks the ID is unique). */
function validate(f:Form){const e:Partial<Record<keyof Form,string>>={};const num=(v:string)=>v.trim()===''?null:Number(v);
 if(f.code.trim().length<2)e.code='Enter a tower ID, e.g. TWR-004';
 if(f.name.trim().length<2)e.name='Enter the site name';
 const h=num(f.height_m),lat=num(f.latitude),lon=num(f.longitude);
 if(h!==null&&(!isFinite(h)||h<=0||h>1000))e.height_m='Height in metres, 1–1000';
 if(lat!==null&&(!isFinite(lat)||lat<-90||lat>90))e.latitude='Latitude between −90 and 90';
 if(lon!==null&&(!isFinite(lon)||lon<-180||lon>180))e.longitude='Longitude between −180 and 180';
 if((lat===null)!==(lon===null))e[lat===null?'latitude':'longitude']='Enter both coordinates, or neither';
 return e}
const payload=(f:Form,edit:boolean)=>{const n=(v:string)=>v.trim()===''?null:Number(v);
 return {code:f.code.trim(),name:f.name.trim(),region:f.region.trim()||'Unknown',operator:f.operator.trim()||null,tower_type:f.tower_type,height_m:n(f.height_m),
  latitude:n(f.latitude),longitude:n(f.longitude),address:f.address.trim()||null,notes:f.notes.trim()||null,...(edit?{status:f.status}:{})}};

function TowerForm({site,onClose,onSaved}:{site:Site|null;onClose:()=>void;onSaved:(s:Site)=>void}){
 const [f,setF]=useState<Form>(site?toForm(site):emptyForm),[touched,setTouched]=useState(false),[saving,setSaving]=useState(false),[error,setError]=useState('');
 const errors=validate(f),bad=Object.keys(errors).length>0;
 const set=(k:keyof Form)=>(e:React.ChangeEvent<HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement>)=>setF({...f,[k]:e.target.value});
 const field=(k:keyof Form,label:string,props:React.InputHTMLAttributes<HTMLInputElement>={})=><label className={'st-field'+(touched&&errors[k]?' bad':'')}><span>{label}</span><input value={f[k]} onChange={set(k)} {...props}/>{touched&&errors[k]&&<small>{errors[k]}</small>}</label>;
 async function save(){setTouched(true);if(bad)return;setSaving(true);setError('');
  try{const saved=site?await request<Site>(`/api/v1/sites/${site.id}`,json('PATCH',payload(f,true))):await request<Site>('/api/v1/sites',json('POST',payload(f,false)));onSaved(saved)}
  catch(e){setError(errText(e))}finally{setSaving(false)}}
 useEffect(()=>{const k=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};window.addEventListener('keydown',k);return()=>window.removeEventListener('keydown',k)},[onClose]);
 return <div className="st-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)onClose()}}><form className="st-modal" onSubmit={e=>{e.preventDefault();save()}}>
  <div className="st-modal-head"><div className="st-modal-icon"><RadioTower size={18}/></div><div><h2>{site?`Edit ${site.code}`:'Add tower'}</h2><small>{site?'Update the site record. Inspections and their data are not affected.':'Register a tower site; inspections, uploads and 3D models are added to it afterwards.'}</small></div><button type="button" className="st-icon-btn" onClick={onClose} aria-label="Close"><X size={18}/></button></div>
  <div className="st-modal-body">
   <div className="st-section-title">Identification</div>
   <div className="st-grid">{field('code','Tower ID *',{placeholder:'TWR-004',autoFocus:!site,style:{textTransform:'uppercase'}})}{field('name','Site name *',{placeholder:'North Ridge'})}
    {field('region','Region / city',{placeholder:'Hyderabad'})}{field('operator','Operator / owner',{placeholder:'e.g. Indus Towers'})}</div>
   <div className="st-section-title">Structure</div>
   <div className="st-grid"><label className="st-field"><span>Structure type</span><select value={f.tower_type} onChange={set('tower_type')}>{[...new Set([...STRUCTURES,f.tower_type])].map(s=><option key={s}>{s}</option>)}</select></label>
    {field('height_m','Height (m)',{inputMode:'decimal',placeholder:'45'})}
    {site&&<label className="st-field"><span>Status</span><select value={f.status} onChange={set('status')}>{STATUSES.map(s=><option key={s.id} value={s.id}>{s.label}</option>)}</select></label>}</div>
   <div className="st-section-title">Location</div>
   <div className="st-grid">{field('latitude','Latitude',{inputMode:'decimal',placeholder:'17.3850'})}{field('longitude','Longitude',{inputMode:'decimal',placeholder:'78.4867'})}
    <label className="st-field st-span"><span>Address</span><input value={f.address} onChange={set('address')} placeholder="Street, area, PIN"/></label>
    <label className="st-field st-span"><span>Notes</span><textarea rows={3} value={f.notes} onChange={set('notes')} placeholder="Access, permits, landlord contact…"/></label></div>
   {error&&<div className="error st-form-error"><AlertTriangle size={16}/>{error}</div>}
  </div>
  <div className="st-modal-foot"><button type="button" className="btn outline" onClick={onClose}>Cancel</button><button type="submit" className="btn primary" disabled={saving||(touched&&bad)}>{saving?'Saving…':site?'Save changes':<><Plus size={15}/>Add tower</>}</button></div>
 </form></div>;
}

function Confirm({title,body,action,danger,onCancel,onConfirm}:{title:string;body:React.ReactNode;action:string;danger?:boolean;onCancel:()=>void;onConfirm:()=>Promise<void>}){
 const [busy,setBusy]=useState(false),[error,setError]=useState('');
 return <div className="st-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)onCancel()}}><div className="st-modal st-confirm">
  <div className="st-modal-head"><div className={'st-modal-icon'+(danger?' danger':'')}>{danger?<Trash2 size={18}/>:<Archive size={18}/>}</div><div><h2>{title}</h2></div></div>
  <div className="st-modal-body"><p>{body}</p>{error&&<div className="error st-form-error"><AlertTriangle size={16}/>{error}</div>}</div>
  <div className="st-modal-foot"><button className="btn outline" onClick={onCancel}>Cancel</button><button className={'btn primary'+(danger?' st-danger':'')} disabled={busy} onClick={async()=>{setBusy(true);setError('');try{await onConfirm()}catch(e){setError(errText(e));setBusy(false)}}}>{busy?'Working…':action}</button></div>
 </div></div>;
}

function Progress({done,total}:{done:number;total:number}){const pct=total?Math.round(done*100/total):0;
 return <div className="st-progress" title={`${done} of ${total} images approved in QC`}><div><i style={{width:pct+'%'}}/></div><small>{total?`${done}/${total}`:'—'}</small></div>}

function SiteDrawer({site,onClose,onEdit,onArchive,onDelete,go,select,reload}:{site:SiteRow;onClose:()=>void;onEdit:()=>void;onArchive:()=>void;onDelete:()=>void;go:(p:string)=>void;select:(id:number)=>void;reload:()=>void}){
 const [rows,setRows]=useState<InspectionRow[]|null>(null),[error,setError]=useState(''),[label,setLabel]=useState(''),[adding,setAdding]=useState(false);
 const load=()=>request<InspectionRow[]>(`/api/v1/sites/${site.id}/inspections-summary`).then(setRows).catch(e=>setError(errText(e)));
 useEffect(()=>{setRows(null);setError('');setAdding(false);load()},[site.id]);
 useEffect(()=>{const k=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};window.addEventListener('keydown',k);return()=>window.removeEventListener('keydown',k)},[onClose]);
 const open=(page:string)=>{select(site.id);go(page)};
 async function addInspection(){if(label.trim().length<2)return;try{await request('/api/v1/inspections',json('POST',{site_id:site.id,label:label.trim()}));setAdding(false);setLabel('');await load();reload()}catch(e){setError(errText(e))}}
 const s=site.stats,archived=site.status==='archived';
 const maps=site.latitude!=null&&site.longitude!=null?`https://www.openstreetmap.org/?mlat=${site.latitude}&mlon=${site.longitude}#map=17/${site.latitude}/${site.longitude}`:null;
 return <div className="st-overlay st-drawer-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)onClose()}}><aside className="st-drawer">
  <div className="st-drawer-head"><div className="st-modal-icon"><RadioTower size={18}/></div><div><h2>{site.code}</h2><small>{site.name}</small></div><span className={'st-status '+site.status}>{statusLabel(site.status)}</span><button className="st-icon-btn" onClick={onClose} aria-label="Close"><X size={18}/></button></div>
  <div className="st-drawer-actions">
   <button className="btn primary" onClick={()=>{setAdding(true);setLabel('Drone survey '+new Date().toLocaleDateString())}} disabled={archived}><Plus size={14}/>New inspection</button>
   <button className="btn outline" onClick={()=>open('upload')}><UploadCloud size={14}/>Upload media</button>
   <button className="btn outline" onClick={()=>open('twin')}><Layers size={14}/>Digital twin</button>
   <button className="btn outline" onClick={onEdit}><Edit3 size={14}/>Edit</button>
  </div>
  {adding&&<form className="st-inline-form" onSubmit={e=>{e.preventDefault();addInspection()}}><input autoFocus value={label} onChange={e=>setLabel(e.target.value)} placeholder="Inspection label"/><button className="btn primary" disabled={label.trim().length<2}>Create</button><button type="button" className="btn outline" onClick={()=>setAdding(false)}>Cancel</button></form>}
  <div className="st-drawer-body">
   <div className="st-kpis"><div><strong>{s.inspections}</strong><small>Inspections</small></div><div><strong>{s.photos}</strong><small>Photos{s.videos?` · ${s.videos} video`:''}</small></div><div><strong>{s.models}</strong><small>3D models</small></div><div className={s.items_with_defects?'warn':''}><strong>{s.items_with_defects}</strong><small>With defects</small></div></div>
   <div className="st-details">
    <div><span>Region</span><b>{site.region||'—'}</b></div><div><span>Operator</span><b>{site.operator||'—'}</b></div>
    <div><span>Structure</span><b>{site.tower_type}{site.height_m?` · ${site.height_m} m`:''}</b></div>
    <div><span>Coordinates</span><b>{maps?<a href={maps} target="_blank" rel="noreferrer">{site.latitude!.toFixed(5)}, {site.longitude!.toFixed(5)} <ExternalLink size={11}/></a>:'—'}</b></div>
    {site.address&&<div className="st-span"><span>Address</span><b>{site.address}</b></div>}
    {site.notes&&<div className="st-span"><span>Notes</span><b className="st-notes">{site.notes}</b></div>}
   </div>
   <div className="st-section-title">Inspections</div>
   {error&&<div className="error"><AlertTriangle size={16}/>{error}</div>}
   {!rows?<p className="muted">Loading…</p>:!rows.length?<div className="st-empty-small"><Calendar size={22}/><p>No inspections yet. Create one, then upload drone photos or video to it.</p></div>
   :<div className="st-inspections">{rows.map(r=><div key={r.id} className="st-inspection">
     <div className="st-ins-head"><b>{r.label}</b><small>{insCode(r.id)} · {fmtDate(r.created_at)}</small></div>
     <div className="st-ins-facts"><span><ImageIcon size={12}/>{r.photos} photos</span>{r.videos>0&&<span><Video size={12}/>{r.videos} videos</span>}<span><ScanEye size={12}/>{r.detections.toLocaleString()} detections</span>
      {r.models.filter(m=>m.status==='completed').length>0&&<span><Box size={12}/>{r.models.filter(m=>m.status==='completed').map(m=>m.dense?'dense':'sparse').join(' + ')} 3D</span>}
      {r.items_with_defects>0&&<span className="warn"><ShieldAlert size={12}/>{r.items_with_defects} defects</span>}</div>
     <Progress done={r.approved_images} total={r.images}/>
     {r.latest_job&&['queued','running','failed'].includes(r.latest_job.status)&&<small className={'st-job '+r.latest_job.status}>{r.latest_job.kind==='reconstruction'?'3D build':'Detection'} {r.latest_job.status}{r.latest_job.status==='running'?` · ${r.latest_job.progress}%`:''}</small>}
     <div className="st-ins-links"><button className="text-btn" onClick={()=>open('detection')}>Detections</button><button className="text-btn" onClick={()=>open('review')}><ClipboardCheck size={12}/>QC review</button><button className="text-btn" onClick={()=>open('reconstruction')}><Box size={12}/>3D</button><button className="text-btn" onClick={()=>open('assets')}>Inventory <ArrowRight size={12}/></button></div>
    </div>)}</div>}
  </div>
  <div className="st-drawer-foot">
   <button className="btn outline" onClick={onArchive}>{archived?<><ArchiveRestore size={14}/>Restore</>:<><Archive size={14}/>Archive</>}</button>
   <button className="btn outline st-danger-outline" onClick={onDelete} disabled={s.inspections>0} title={s.inspections?'Towers with inspections can be archived, not deleted':'Delete this empty tower'}><Trash2 size={14}/>Delete</button>
   {site.created_at&&<small className="muted">Added {fmtDate(site.created_at)}</small>}
  </div>
 </aside></div>;
}

type Props={go:(page:string)=>void;selectSite:(id:number)=>void;onChanged:()=>void;addSignal:number};
export default function SitesPage({go,selectSite,onChanged,addSignal}:Props){
 const [rows,setRows]=useState<SiteRow[]|null>(null),[error,setError]=useState('');
 const [query,setQuery]=useState(''),[status,setStatus]=useState('current'),[region,setRegion]=useState(''),[structure,setStructure]=useState(''),[sort,setSort]=useState('code');
 const [form,setForm]=useState<{site:Site|null}|null>(null),[openId,setOpenId]=useState<number|null>(null),[confirm,setConfirm]=useState<null|{kind:'archive'|'delete';site:SiteRow}>(null);
 const load=()=>request<SiteRow[]>('/api/v1/sites-overview').then(r=>{setRows(r);setError('')}).catch(e=>setError(errText(e)));
 useEffect(()=>{load();const t=setInterval(load,30000);return()=>clearInterval(t)},[]);
 const seenSignal=useRef(addSignal);  // the header's Add Tower button: react to new clicks only, not to the count when the page opens
 useEffect(()=>{if(addSignal!==seenSignal.current){seenSignal.current=addSignal;setForm({site:null})}},[addSignal]);
 const changed=async()=>{await load();onChanged()};
 const all=rows||[];
 const regions=useMemo(()=>[...new Set(all.map(s=>s.region).filter(Boolean))].sort(),[rows]),structures=useMemo(()=>[...new Set(all.map(s=>s.tower_type))].sort(),[rows]);
 const counts=useMemo(()=>Object.fromEntries(STATUSES.map(s=>[s.id,all.filter(x=>x.status===s.id).length])),[rows]);
 const q=query.trim().toLowerCase();
 const shown=all.filter(s=>(status==='all'||(status==='current'?s.status!=='archived':s.status===status))&&(!region||s.region===region)&&(!structure||s.tower_type===structure)
  &&(!q||[s.code,s.name,s.region,s.operator,s.address,s.tower_type].some(v=>v?.toLowerCase().includes(q))))
  .sort((a,b)=>sort==='name'?a.name.localeCompare(b.name):sort==='recent'?(b.stats.last_inspection_at||'').localeCompare(a.stats.last_inspection_at||''):sort==='defects'?b.stats.items_with_defects-a.stats.items_with_defects:a.code.localeCompare(b.code,undefined,{numeric:true}));
 const live=all.filter(s=>s.status!=='archived'),sum=(k:keyof Stats)=>live.reduce((t,s)=>t+(Number(s.stats[k])||0),0);
 const cards=[{label:'Tower sites',value:live.length,sub:`${counts.active||0} active · ${counts.maintenance||0} in maintenance`,icon:RadioTower},
  {label:'Inspections',value:sum('inspections'),sub:`${sum('photos').toLocaleString()} photos · ${sum('videos')} videos`,icon:Calendar},
  {label:'QC approved images',value:`${sum('approved_images').toLocaleString()}`,sub:`of ${sum('images').toLocaleString()} images`,icon:ClipboardCheck},
  {label:'3D models',value:sum('models'),sub:`${sum('boxes_3d')} equipment boxes`,icon:Box},
  {label:'Items with defects',value:sum('items_with_defects'),sub:`${sum('poor_or_critical')} poor or critical`,icon:ShieldAlert,warn:sum('poor_or_critical')>0}];
 const current=all.find(s=>s.id===openId)||null,filtered=query||region||structure||status!=='current';
 return <>
  <div className="st-cards">{cards.map(c=><div key={c.label} className={'stat-card st-card'+(c.warn?' warn':'')}><div className="st-card-icon"><c.icon size={18}/></div><div><small>{c.label}</small><strong>{rows?c.value:'—'}</strong><span>{rows?c.sub:''}</span></div></div>)}</div>
  <section className="panel">
   <div className="panel-head"><h2>Tower sites</h2><span className="muted">{shown.length} of {all.length} sites</span></div>
   <div className="st-toolbar">
    <div className="search"><Search size={16}/><input placeholder="Search ID, name, region, operator, address" value={query} onChange={e=>setQuery(e.target.value)}/>{query&&<button className="st-icon-btn" onClick={()=>setQuery('')} aria-label="Clear search"><X size={14}/></button>}</div>
    <div className="st-seg">{[{id:'current',label:'Current'},...STATUSES,{id:'all',label:'All'}].map(s=><button key={s.id} className={status===s.id?'on':''} onClick={()=>setStatus(s.id)}>{s.label}{s.id in counts?<b>{counts[s.id]}</b>:null}</button>)}</div>
    <select aria-label="Region" value={region} onChange={e=>setRegion(e.target.value)}><option value="">All regions</option>{regions.map(r=><option key={r}>{r}</option>)}</select>
    <select aria-label="Structure" value={structure} onChange={e=>setStructure(e.target.value)}><option value="">All structures</option>{structures.map(r=><option key={r}>{r}</option>)}</select>
    <select aria-label="Sort" value={sort} onChange={e=>setSort(e.target.value)}><option value="code">Sort: Tower ID</option><option value="name">Sort: Name</option><option value="recent">Sort: Last inspection</option><option value="defects">Sort: Most defects</option></select>
    {filtered&&<button className="text-btn" onClick={()=>{setQuery('');setRegion('');setStructure('');setStatus('current')}}>Reset</button>}
   </div>
   {error&&<div className="error"><AlertTriangle size={16}/>{error}<button onClick={load}>Retry</button></div>}
   {!rows&&!error?<p className="muted">Loading sites…</p>:!shown.length?<div className="st-empty"><RadioTower size={34}/><p>{all.length?'No sites match these filters.':'No tower sites yet. Add the first tower to start uploading inspections.'}</p>{all.length?<button className="btn outline" onClick={()=>{setQuery('');setRegion('');setStructure('');setStatus('current')}}>Clear filters</button>:<button className="btn primary" onClick={()=>setForm({site:null})}><Plus size={15}/>Add tower</button>}</div>
   :<div className="st-table-wrap"><table className="st-table"><thead><tr><th>Tower</th><th>Region · operator</th><th>Structure</th><th>Location</th><th>Inspections</th><th>QC progress</th><th>3D</th><th>Condition</th><th>Status</th><th></th></tr></thead>
    <tbody>{shown.map(s=><tr key={s.id} className={openId===s.id?'on':''} onClick={()=>setOpenId(s.id)}>
     <td><b>{s.code}</b><small>{s.name}</small></td>
     <td>{s.region||'—'}<small>{s.operator||'—'}</small></td>
     <td>{s.tower_type}<small>{s.height_m?`${s.height_m} m`:'height —'}</small></td>
     <td>{s.latitude!=null&&s.longitude!=null?<a className="st-coords" href={`https://www.openstreetmap.org/?mlat=${s.latitude}&mlon=${s.longitude}#map=17/${s.latitude}/${s.longitude}`} target="_blank" rel="noreferrer" onClick={e=>e.stopPropagation()}><MapPin size={12}/>{s.latitude.toFixed(3)}, {s.longitude.toFixed(3)}</a>:<span className="muted">—</span>}</td>
     <td>{s.stats.inspections}<small>{s.stats.last_inspection_at?`last ${fmtDate(s.stats.last_inspection_at)}`:'none yet'}</small></td>
     <td><Progress done={s.stats.approved_images} total={s.stats.images}/>{s.stats.jobs_running>0&&<small className="st-job running">processing…</small>}</td>
     <td>{s.stats.models?<>{s.stats.models} model{s.stats.models>1?'s':''}<small>{s.stats.boxes_3d} boxes{s.stats.dense_models?' · dense':''}</small></>:<span className="muted">—</span>}</td>
     <td>{s.stats.items_with_defects?<span className={'badge '+(s.stats.poor_or_critical?'poor':'fair')}>{s.stats.items_with_defects} defects</span>:s.stats.items?<span className="badge good">No defects</span>:<span className="muted">—</span>}</td>
     <td><span className={'st-status '+s.status}>{statusLabel(s.status)}</span></td>
     <td><button className="text-btn" onClick={e=>{e.stopPropagation();setOpenId(s.id)}}>Details <ArrowRight size={13}/></button></td>
    </tr>)}</tbody></table></div>}
  </section>
  {current&&<SiteDrawer site={current} onClose={()=>setOpenId(null)} go={go} select={selectSite} reload={changed} onEdit={()=>setForm({site:current})}
    onArchive={()=>current.status==='archived'?request(`/api/v1/sites/${current.id}`,json('PATCH',{status:'active'})).then(changed).catch(e=>setError(errText(e))):setConfirm({kind:'archive',site:current})}
    onDelete={()=>setConfirm({kind:'delete',site:current})}/>}
  {form&&<TowerForm site={form.site} onClose={()=>setForm(null)} onSaved={async s=>{setForm(null);await changed();setOpenId(s.id)}}/>}
  {confirm?.kind==='archive'&&<Confirm title={`Archive ${confirm.site.code}?`} action="Archive" onCancel={()=>setConfirm(null)}
    body={<>The tower is hidden from the current list and its inspections, photos, 3D models and findings are kept. You can restore it at any time from <b>Archived</b>.</>}
    onConfirm={async()=>{await request(`/api/v1/sites/${confirm.site.id}`,json('PATCH',{status:'archived'}));setConfirm(null);await changed()}}/>}
  {confirm?.kind==='delete'&&<Confirm danger title={`Delete ${confirm.site.code}?`} action="Delete permanently" onCancel={()=>setConfirm(null)}
    body={<>This removes the site record <b>{confirm.site.name}</b>. Only towers without inspections can be deleted.</>}
    onConfirm={async()=>{await request(`/api/v1/sites/${confirm.site.id}`,{method:'DELETE'});setConfirm(null);setOpenId(null);await changed()}}/>}
 </>;
}
