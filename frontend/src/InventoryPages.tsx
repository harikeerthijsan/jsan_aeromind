import React,{useEffect,useMemo,useRef,useState} from 'react';import {useNavigate} from 'react-router-dom';
import {Database,ShieldAlert,Image as ImageIcon,CheckCircle2,AlertTriangle,Download,ExternalLink,ChevronDown,ChevronRight,Factory,Layers,ClipboardCheck,RotateCcw} from 'lucide-react';
import {API,request,Site,Inspection,Annotation} from './api';import {colorOf,fmtTime} from './AnnotationEditor';

// Observed inventory and condition from QC-approved images (GET /api/v1/sites/{id}/inventory)
type InvImage={video:string|null;frame_time:number|null;media_id:number;inspection_id:number;filename:string;url:string;status:string;reviewed_at:string|null;workflow:string|null;notes:string;image_width:number|null;image_height:number|null;items:Annotation[]};
type ClassRow={class_name:string;class_id:number;count:number;manufacturers:Record<string,number>;grades:Record<string,number>;defects:number};
type Inventory={site:Site;inspections:Inspection[];images:InvImage[];summary:{images_total:number;images_approved:number;items:number;items_with_defects:number;grades:Record<string,number>;manufacturers:Record<string,number>;defect_types:Record<string,number>;verified:Record<string,number>;by_class:ClassRow[]}};
const GRADES=['Critical','Poor','Fair','Good','Unknown','Not assessed'],ISSUES=['Critical','Poor','Fair'];
const GRADE_COLOR:Record<string,string>={Critical:'#c6463b',Poor:'#e07a2e',Fair:'#d39b14',Good:'#16835a',Unknown:'#8fa0b2','Not assessed':'#d5dde6'};
const gradeOf=(a:Annotation)=>a.condition_grade||'Not assessed';
const insCode=(id:number)=>'INS-'+String(id).padStart(4,'0');
const fmtDate=(s:string|null)=>s?new Date(s).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'}):'';
const pretty=(s:string)=>s.replaceAll('_',' ');

function useInventory(siteId?:number,inspectionId?:number|null){
 const [data,setData]=useState<Inventory|null>(null),[error,setError]=useState(''),[tick,setTick]=useState(0);
 useEffect(()=>{if(!siteId)return;let live=true;setError('');request<Inventory>(`/api/v1/sites/${siteId}/inventory${inspectionId?`?inspection_id=${inspectionId}`:''}`).then(d=>live&&setData(d)).catch(e=>live&&setError(String(e)));return()=>{live=false}},[siteId,inspectionId,tick]);
 return {data:data&&data.site.id===siteId?data:null,error,reload:()=>setTick(t=>t+1)};
}
export function GradeBadge({grade,score}:{grade:string|null;score?:number|null}){const g=grade||'Not assessed';return <span className={'inv-grade'+(g==='Not assessed'?' na':'')} style={{'--g':GRADE_COLOR[g]} as React.CSSProperties}>{g}{score!=null&&g!=='Unknown'?` · ${score}`:''}</span>}
function GradeBar({grades,total}:{grades:Record<string,number>;total:number}){
 return <div className="inv-gradebar" title={GRADES.filter(g=>grades[g]).map(g=>`${g}: ${grades[g]}`).join(' · ')}>{GRADES.filter(g=>grades[g]).map(g=><div key={g} style={{width:`${grades[g]/Math.max(1,total)*100}%`,background:GRADE_COLOR[g]}}/>)}</div>}
/** The photos' stored previews (1280 px, ~250 KB) instead of the 12-16 MB originals: a page of equipment asked for
 * dozens of originals at once, and the browser gave up on some (net::ERR_FAILED). */
const PREVIEW=1280;
const preview=(img:InvImage)=>`${API}/api/v1/media/${img.media_id}/thumb?w=${PREVIEW}`;
/** Loads only once on screen (a long list of items must not fetch every photo up front). */
function useOnScreen<T extends Element>(){const ref=useRef<T>(null),[on,setOn]=useState(false);
 useEffect(()=>{const el=ref.current;if(!el||on)return;const o=new IntersectionObserver(e=>{if(e.some(x=>x.isIntersecting)){setOn(true);o.disconnect()}},{rootMargin:'200px'});o.observe(el);return()=>o.disconnect()},[on]);
 return [ref,on] as const}
/** The equipment cut out of the photo with CSS: from the preview, or from the original for parts too small to be sharp in it. */
function ItemThumb({img,item,size=64}:{img:InvImage;item:Annotation;size?:number}){
 const pad=0.12,w=item.x2-item.x1,h=item.y2-item.y1,cw=w*(1+2*pad),ch=h*(1+2*pad),k=size/Math.max(cw,ch),W=item.image_width,H=item.image_height,[ref,on]=useOnScreen<HTMLDivElement>();
 const src=W*k<=2*PREVIEW?preview(img):API+img.url;
 return <div ref={ref} className="inv-thumb" style={{width:size,height:size,backgroundImage:on?`url(${src})`:undefined,backgroundSize:`${W*k}px ${H*k}px`,backgroundPosition:`${-(item.x1-w*pad)*k+(size-cw*k)/2}px ${-(item.y1-h*pad)*k+(size-ch*k)/2}px`}}/>}
function ImageOverlay({img,hover,onHover}:{img:InvImage;hover:number|null;onHover:(id:number|null)=>void}){
 const [ref,on]=useOnScreen<SVGSVGElement>();
 if(!img.image_width||!img.image_height)return <img className="inv-photo" src={preview(img)} alt={img.filename} loading="lazy"/>;
 const sw=Math.max(img.image_width,img.image_height)/400;
 return <svg ref={ref} className="inv-photo" viewBox={`0 0 ${img.image_width} ${img.image_height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${img.filename} with ${img.items.length} approved boxes`}>
  {on&&<image href={preview(img)} width={img.image_width} height={img.image_height}/>}
  {[...img.items].sort((a,b)=>(b.x2-b.x1)*(b.y2-b.y1)-(a.x2-a.x1)*(a.y2-a.y1)).map(a=>{const g=gradeOf(a),c=g==='Not assessed'?colorOf(a.class_id):GRADE_COLOR[g],on=hover===a.id;
   return <rect key={a.id} x={a.x1} y={a.y1} width={a.x2-a.x1} height={a.y2-a.y1} fill={on?c+'55':'transparent'} stroke={c} strokeWidth={(on?3:ISSUES.includes(g)?2.2:1.2)*sw} onMouseEnter={()=>onHover(a.id)} onMouseLeave={()=>onHover(null)}><title>{a.class_name}{a.manufacturer?` · ${a.manufacturer}`:''} · {g}</title></rect>})}
 </svg>}
function Toolbar({data,inspectionId,setInspectionId,reload,extra}:{data:Inventory|null;inspectionId:number|null;setInspectionId:(v:number|null)=>void;reload:()=>void;extra?:React.ReactNode}){
 const withApproved=new Set(data?.images.filter(i=>i.status==='approved').map(i=>i.inspection_id));
 return <div className="toolbar inv-toolbar"><select aria-label="Inspection" value={inspectionId??''} onChange={e=>setInspectionId(e.target.value?+e.target.value:null)}><option value="">All inspections</option>{data?.inspections.map(i=><option key={i.id} value={i.id}>{insCode(i.id)} · {i.label}</option>)}</select>
  {extra}<span className="inv-spacer"/>{data?.inspections.filter(i=>withApproved.has(i.id)).map(i=><a key={i.id} className="btn outline" href={`${API}/api/v1/inspections/${i.id}/annotations.csv`} title="One row per approved box: class, manufacturer, condition, defects"><Download size={14}/>{insCode(i.id)} CSV</a>)}
  <button className="btn outline" onClick={reload} title="Reload"><RotateCcw size={14}/></button></div>}
const openInQC=(nav:(to:string)=>void,img:InvImage,item?:Annotation)=>nav(`/review?inspection=${img.inspection_id}&media=${img.media_id}${item?`&item=${item.id}`:''}`);

export function AssetInventory({site,legacy}:{site?:Site;legacy:React.ReactNode}){
 const nav=useNavigate();const [inspectionId,setInspectionId]=useState<number|null>(null),[open,setOpen]=useState<Set<number>>(new Set()),[hover,setHover]=useState<number|null>(null),[cls,setCls]=useState('');
 const {data,error,reload}=useInventory(site?.id,inspectionId);const s=data?.summary;
 const approved=(data?.images||[]).filter(i=>i.status==='approved'),pending=(data?.images||[]).filter(i=>i.status!=='approved');
 const toggle=(id:number)=>setOpen(o=>{const n=new Set(o);n.has(id)?n.delete(id):n.add(id);return n});
 const cards=[{label:'Approved images',value:s?`${s.images_approved} / ${s.images_total}`:'—',icon:ImageIcon},{label:'Equipment observed',value:s?.items??'—',icon:Database},{label:'Equipment classes',value:s?.by_class.length??'—',icon:Layers},{label:'Manufacturers tagged',value:s?Object.keys(s.manufacturers).filter(m=>m!=='Unknown').length:'—',icon:Factory},{label:'Items with defects',value:s?.items_with_defects??'—',icon:ShieldAlert}];
 return <>
  <div className="stats inv-stats">{cards.map(c=><div className="stat-card" key={c.label}><div className="stat-icon"><c.icon size={20}/></div><div><small>{c.label}</small><strong>{c.value}</strong></div></div>)}</div>
  <section className="panel"><div className="panel-head"><h2>Observed inventory · {site?.code}</h2><span className="muted">From QC-approved images</span></div>
   <Toolbar data={data} inspectionId={inspectionId} setInspectionId={setInspectionId} reload={reload}/>
   {error&&<div className="error"><AlertTriangle size={16}/>{error}</div>}
   {s&&s.items===0?<div className="inv-empty"><ClipboardCheck size={34}/><p>No approved equipment yet. Review images in <b>AI Detection</b> or <b>QC Review</b> and press <b>Approve</b>; they appear here.</p><button className="btn primary" onClick={()=>nav('/review')}>Open QC Review</button></div>
   :s&&<table className="inv-table"><thead><tr><th>Equipment class</th><th>Count</th><th>Manufacturers</th><th>Condition</th><th>Defects</th><th></th></tr></thead><tbody>{s.by_class.map(c=><tr key={c.class_name} className={cls===c.class_name?'on':''}>
     <td><span className="inv-dot" style={{background:colorOf(c.class_id)}}/><b>{pretty(c.class_name)}</b></td><td><b>{c.count}</b></td>
     <td><div className="inv-chips">{Object.entries(c.manufacturers).map(([m,n])=><span key={m} className={'inv-chip'+(m==='Unknown'?' dim':'')}>{m} <small>{n}</small></span>)}{!Object.keys(c.manufacturers).length&&<span className="muted">—</span>}</div></td>
     <td className="inv-gradecell"><GradeBar grades={c.grades} total={c.count}/><small>{GRADES.filter(g=>c.grades[g]).map(g=>`${c.grades[g]} ${g.toLowerCase()}`).join(' · ')}</small></td>
     <td>{c.defects||'—'}</td><td><button className="text-btn" onClick={()=>setCls(cls===c.class_name?'':c.class_name)}>{cls===c.class_name?'Show all':'Filter images'}</button></td></tr>)}</tbody></table>}
  </section>
  {approved.length>0&&<section className="panel"><div className="panel-head"><h2>Per-image inventory</h2><span className="muted">{approved.length} approved image{approved.length===1?'':'s'}{cls?` containing ${pretty(cls)}`:''}</span></div>
   <div className="inv-grid">{approved.filter(i=>!cls||i.items.some(a=>a.class_name===cls)).map(img=>{const counts=new Map<string,{id:number;n:number}>();img.items.forEach(a=>{const c=counts.get(a.class_name)||{id:a.class_id,n:0};c.n++;counts.set(a.class_name,c)});
    const grades:Record<string,number>={};img.items.forEach(a=>{grades[gradeOf(a)]=(grades[gradeOf(a)]||0)+1});const issues=img.items.filter(a=>ISSUES.includes(gradeOf(a))).length,isOpen=open.has(img.media_id);
    return <article key={img.media_id} className="inv-card">
     <ImageOverlay img={img} hover={hover} onHover={setHover}/>
     <div className="inv-card-body"><div className="inv-card-head"><div><b>{img.filename}</b><small>{img.video?`frame at ${fmtTime(img.frame_time||0)} of ${img.video} · `:''}{insCode(img.inspection_id)} · approved {fmtDate(img.reviewed_at)} · {img.workflow==='manual_qc'?'manual QC':'AI detection'}</small></div>{issues>0?<span className="badge poor">{issues} with defects</span>:<span className="badge good">Approved</span>}</div>
      <div className="inv-chips">{[...counts].sort((a,b)=>b[1].n-a[1].n).map(([n,{id,n:k}])=><span key={n} className="inv-chip"><i style={{background:colorOf(id)}}/>{pretty(n)} <small>{k}</small></span>)}</div>
      <GradeBar grades={grades} total={img.items.length}/>
      {img.notes&&<p className="muted inv-note">“{img.notes}”</p>}
      <div className="inv-actions"><button className="text-btn" onClick={()=>toggle(img.media_id)}>{isOpen?<ChevronDown size={14}/>:<ChevronRight size={14}/>}{img.items.length} items</button><span className="inv-spacer"/>
       <button className="text-btn" onClick={()=>openInQC(nav,img)}><ExternalLink size={13}/>Open in QC</button><a className="text-btn" href={`${API}/api/v1/media/${img.media_id}/annotated?source=approved`}><Download size={13}/>Annotated image</a></div>
      {isOpen&&<div className="inv-items"><table><thead><tr><th></th><th>Equipment</th><th>Manufacturer</th><th>Condition</th><th>Defects</th></tr></thead><tbody>
       {[...img.items].sort((a,b)=>GRADES.indexOf(gradeOf(a))-GRADES.indexOf(gradeOf(b))||a.class_name.localeCompare(b.class_name)).map(a=><tr key={a.id} className={hover===a.id?'on':''} onMouseEnter={()=>setHover(a.id)} onMouseLeave={()=>setHover(null)} onClick={()=>openInQC(nav,img,a)} title="Open this box in QC">
        <td><ItemThumb img={img} item={a} size={44}/></td><td><b>{pretty(a.class_name)}</b><small>{a.source==='manual'?'drawn by reviewer':a.source==='model_edited'?'AI box, edited':'AI box'}</small></td>
        <td>{a.manufacturer?<>{a.manufacturer}{a.product_model&&<small>{a.product_model}</small>}<small>{a.vendor_source==='ai'?'AI suggestion':a.vendor_basis==='reviewer_choice'?'reviewer pick':'reviewer'}</small></>:<span className="muted">—</span>}</td>
        <td><GradeBadge grade={a.condition_grade} score={a.condition_score}/>{a.condition_verified==='disputed'&&<small className="inv-warn">disputed</small>}</td>
        <td>{(a.condition_defects||[]).map((d,i)=><small key={i} className="inv-defect">{pretty(d.type)} · {d.severity} · ~{d.extent_percent}%</small>)}{!(a.condition_defects||[]).length&&<span className="muted">—</span>}</td></tr>)}</tbody></table></div>}
     </div></article>})}</div></section>}
  {pending.length>0&&<section className="panel"><div className="panel-head"><h2>Awaiting approval</h2><span className="muted">{pending.length} image{pending.length===1?'':'s'} not in the inventory yet</span></div>
   <div className="inv-chips">{pending.map(img=><button key={img.media_id} className="inv-chip pending" onClick={()=>openInQC(nav,img)}><ImageIcon size={12}/>{img.filename} <small>{insCode(img.inspection_id)}</small></button>)}</div></section>}
  <details className="panel inv-legacy"><summary>Canonical asset register (demo records)</summary>{legacy}</details>
 </>;
}

export function ConditionFindings({site,legacy}:{site?:Site;legacy:React.ReactNode}){
 const nav=useNavigate();const [inspectionId,setInspectionId]=useState<number|null>(null),[grade,setGrade]=useState('assessed'),[cls,setCls]=useState(''),[check,setCheck]=useState('');
 const {data,error,reload}=useInventory(site?.id,inspectionId);const s=data?.summary;
 const rows=useMemo(()=>(data?.images||[]).filter(i=>i.status==='approved').flatMap(img=>img.items.map(item=>({img,item}))),[data]);
 const shown=rows.filter(({item})=>{const g=gradeOf(item);return (grade==='assessed'?g!=='Not assessed':grade==='issues'?ISSUES.includes(g):grade==='all'||g===grade)&&(!cls||item.class_name===cls)&&(!check||(check==='reviewer'?item.condition_source==='reviewer':item.condition_verified===check))})
  .sort((a,b)=>GRADES.indexOf(gradeOf(a.item))-GRADES.indexOf(gradeOf(b.item))||(a.item.condition_score??101)-(b.item.condition_score??101));
 const classes=[...new Set(rows.map(r=>r.item.class_name))].sort();
 return <>
  <div className="stats inv-stats">{['Critical','Poor','Fair','Good','Not assessed'].map(g=><button key={g} className={'stat-card inv-gradecard'+(grade===g?' on':'')} style={{'--g':GRADE_COLOR[g]} as React.CSSProperties} onClick={()=>setGrade(grade===g?'assessed':g)}><div><small>{g}</small><strong>{s?.grades[g]||0}</strong></div></button>)}</div>
  <section className="panel"><div className="panel-head"><h2>Condition findings · {site?.code}</h2><span className="muted">From QC-approved images · {s?.verified.confirmed||0} confirmed · {s?.verified.disputed||0} disputed by 2nd model</span></div>
   <Toolbar data={data} inspectionId={inspectionId} setInspectionId={setInspectionId} reload={reload} extra={<>
    <select aria-label="Condition" value={grade} onChange={e=>setGrade(e.target.value)}><option value="assessed">All assessed (Good to Critical)</option><option value="issues">Fair, Poor & Critical</option><option value="all">All items incl. not assessed</option>{GRADES.map(g=><option key={g} value={g}>{g}</option>)}</select>
    <select aria-label="Equipment class" value={cls} onChange={e=>setCls(e.target.value)}><option value="">All classes</option>{classes.map(c=><option key={c} value={c}>{pretty(c)}</option>)}</select>
    <select aria-label="Verification" value={check} onChange={e=>setCheck(e.target.value)}><option value="">Any source</option><option value="confirmed">AI · confirmed</option><option value="disputed">AI · disputed</option><option value="reviewer">Set by reviewer</option></select></>}/>
   {s&&Object.keys(s.defect_types).length>0&&<div className="inv-chips inv-defect-types">{Object.entries(s.defect_types).map(([t,n])=><span key={t} className="inv-chip">{pretty(t)} <small>{n}</small></span>)}</div>}
   {error&&<div className="error"><AlertTriangle size={16}/>{error}</div>}
   {!shown.length?<div className="inv-empty"><ShieldAlert size={34}/><p>{rows.length?'No approved items match these filters.':'No approved items yet.'} Run <b>Assess condition</b> in QC Review, check the results and press <b>Approve</b>; findings appear here.</p><button className="btn primary" onClick={()=>nav('/review')}>Open QC Review</button></div>
   :<div className="inv-findings">{shown.map(({img,item})=>{const g=gradeOf(item);return <div key={item.id} className="inv-finding" style={{'--g':GRADE_COLOR[g]} as React.CSSProperties}>
     <ItemThumb img={img} item={item} size={92}/>
     <div className="inv-finding-main"><div className="inv-finding-head"><b>{pretty(item.class_name)}{item.manufacturer&&item.manufacturer!=='Unknown'?` · ${item.manufacturer}`:''}</b><GradeBadge grade={item.condition_grade} score={item.condition_score}/>
       {item.condition_source==='reviewer'?<span className="inv-src">set by reviewer</span>:item.condition_verified==='confirmed'?<span className="inv-src ok"><CheckCircle2 size={12}/>confirmed by 2nd model</span>:item.condition_verified==='disputed'?<span className="inv-src warn"><AlertTriangle size={12}/>disputed by 2nd model</span>:item.condition_source==='ai'?<span className="inv-src">AI</span>:null}</div>
      <small className="muted">{img.video?`${img.video} @ ${fmtTime(img.frame_time||0)}`:img.filename} · {insCode(img.inspection_id)} · approved {fmtDate(img.reviewed_at)}</small>
      {(item.condition_defects||[]).length>0&&<div className="inv-chips">{item.condition_defects!.map((d,i)=><span key={i} className={'inv-chip sev-'+d.severity+(d.confirmed===false?' dim':'')} title={d.evidence}>{pretty(d.type)} · {d.severity} · ~{d.extent_percent}%{d.location?` · ${d.location}`:''}</span>)}</div>}
      {item.condition_summary&&<p>{item.condition_summary}</p>}</div>
     <button className="btn outline" onClick={()=>openInQC(nav,img,item)}><ExternalLink size={13}/>Open in QC</button></div>})}</div>}
   <div className="notice"><AlertTriangle size={17}/>Grades and percentages are visual estimates from drone photos by a local vision model, checked by reviewers. Confirm critical findings on site before acting.</div>
  </section>
  <details className="panel inv-legacy"><summary>Defect records (demo)</summary>{legacy}</details>
 </>;
}
