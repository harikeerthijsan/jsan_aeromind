import {useEffect,useMemo,useRef,useState} from 'react';
import {Area,AreaChart,CartesianGrid,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import {Box,Check,ChevronLeft,ChevronRight,Clock,Cpu,Layers,RadioTower,Search,ShieldAlert} from 'lucide-react';
import {API,request,Site} from './api';
import './overview.css';

/** Overview dashboard (layout inspired by the DNX admin template: dark highlight card, activity chart, card carousel,
 * progress bars with a knob, soft status pills, a week calendar and a "today" card with numbered steps). Real data from
 * GET /api/v1/overview. */
type Stats={inspections:number;last_inspection_at:string|null;photos:number;images:number;approved_images:number;items_with_defects:number;models:number;boxes_3d:number;jobs_running:number};
type OvSite=Site&{stats:Stats;cover_media_id:number|null;last_inspection:{id:number;label:string}|null};
type Step={key:string;label:string;page:string;done:number;detail:string};
type Overview={totals:{sites:number;inspections:number;images:number;approved_images:number;items:number;items_with_defects:number;poor_or_critical:number;models:number;boxes_3d:number;jobs_running:number};
 activity:{date:string;approved:number;inspections:number;models:number}[];
 latest:{id:number;label:string;created_at:string;site:{id:number;code:string;name:string}|null;cover_media_id:number|null;steps:Step[];progress:number}|null;
 sites:OvSite[];jobs:{id:number;kind:string;status:string;stage:string;progress:number;detail:string;inspection:{id:number;label:string};site:{id:number;code:string}}[]};

const thumb=(id:number|null|undefined,w=480)=>id?`${API}/api/v1/media/${id}/thumb?w=${w}`:'';
const ago=(d?:string|null)=>{if(!d)return 'never';const days=Math.floor((Date.now()-new Date(d).getTime())/864e5);return days<=0?'today':days===1?'yesterday':days<31?`${days} days ago`:new Date(d).toLocaleDateString(undefined,{day:'numeric',month:'short'})};
const pct=(a:number,b:number)=>b?Math.round(a*100/b):0;
const JOB_STATUS:Record<string,{label:string;cls:string}>={completed:{label:'Completed',cls:'ok'},running:{label:'In progress',cls:'run'},queued:{label:'Queued',cls:'wait'},failed:{label:'Failed',cls:'bad'},awaiting_worker:{label:'Waiting',cls:'wait'}};

function Ring({value,size=86}:{value:number;size?:number}){const r=size/2-6,c=2*Math.PI*r;
 return <svg className="ov-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`}><circle cx={size/2} cy={size/2} r={r} className="track"/>
  <circle cx={size/2} cy={size/2} r={r} className="value" strokeDasharray={c} strokeDashoffset={c*(1-Math.min(100,value)/100)} transform={`rotate(-90 ${size/2} ${size/2})`}/>
  <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle">{value}%</text></svg>}
export function KnobBar({value}:{value:number}){const v=Math.max(0,Math.min(100,value));
 return <div className="ov-bar"><i style={{width:v+'%'}}/>{v>0&&<b style={{left:`calc(${v}% - 6px)`}}/>}</div>}

function Activity({data}:{data:Overview['activity']}){
 const [range,setRange]=useState(7);
 const rows=useMemo(()=>data.slice(-range).map(d=>({...d,label:new Date(d.date+'T00:00').toLocaleDateString(undefined,range<=7?{weekday:'short'}:{day:'numeric',month:'short'})})),[data,range]);
 const total=rows.reduce((t,r)=>t+r.approved,0);
 return <section className="ov-card ov-activity"><div className="ov-card-head"><div><h3>Activity</h3><small>{total.toLocaleString()} images approved · {rows.reduce((t,r)=>t+r.models,0)} 3D models · {rows.reduce((t,r)=>t+r.inspections,0)} inspections</small></div>
  <select value={range} onChange={e=>setRange(+e.target.value)} aria-label="Period"><option value={7}>This week</option><option value={30}>Last 30 days</option><option value={90}>Last 90 days</option></select></div>
  <div className="ov-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{top:10,right:8,left:-6,bottom:0}}>
   <defs><linearGradient id="ovFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1769aa" stopOpacity={.28}/><stop offset="100%" stopColor="#1769aa" stopOpacity={0}/></linearGradient></defs>
   <CartesianGrid vertical={false} stroke="#edf1f5"/><XAxis dataKey="label" tickLine={false} axisLine={false} tick={{fontSize:11,fill:'#8b9aaa'}} interval="preserveStartEnd" minTickGap={14}/>
   <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{fontSize:11,fill:'#8b9aaa'}} width={44}/>
   <Tooltip contentStyle={{borderRadius:10,border:'1px solid #e1e9f0',fontSize:12}} formatter={(v,n)=>[v,n==='approved'?'Images approved':String(n)]} labelFormatter={(_,p)=>p?.[0]?`${new Date(p[0].payload.date+'T00:00').toLocaleDateString(undefined,{weekday:'long',day:'numeric',month:'short'})} · ${p[0].payload.inspections} inspection(s) · ${p[0].payload.models} 3D model(s)`:''}/>
   <Area type="monotone" dataKey="approved" stroke="#1769aa" strokeWidth={2.5} fill="url(#ovFill)" dot={false} activeDot={{r:5,strokeWidth:2,stroke:'#fff'}}/>
  </AreaChart></ResponsiveContainer></div></section>;
}

function WeekCalendar({data}:{data:Overview['activity']}){
 const byDate=useMemo(()=>Object.fromEntries(data.map(d=>[d.date,d])),[data]);
 const [offset,setOffset]=useState(0),[picked,setPicked]=useState<string|null>(null);
 const today=new Date();today.setHours(0,0,0,0);const start=new Date(today);start.setDate(today.getDate()-today.getDay()+offset*7);
 const days=[...Array(7)].map((_,i)=>{const d=new Date(start);d.setDate(start.getDate()+i);return d});
 const key=(d:Date)=>`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
 const sel=picked?byDate[picked]:null,busy=(d:Date)=>{const a=byDate[key(d)];return a&&(a.approved||a.inspections||a.models)};
 return <section className="ov-card ov-cal"><div className="ov-cal-head"><button onClick={()=>setOffset(offset-1)} aria-label="Previous week"><ChevronLeft size={16}/></button>
  <b>{days[0].toLocaleDateString(undefined,{month:'long',year:'numeric'})}</b><button onClick={()=>setOffset(Math.min(0,offset+1))} disabled={offset>=0} aria-label="Next week"><ChevronRight size={16}/></button></div>
  <div className="ov-week">{days.map(d=>{const k=key(d),isToday=+d===+today;return <button key={k} className={(isToday?'today ':'')+(picked===k?'on ':'')+(d>today?'future':'')} disabled={d>today} onClick={()=>setPicked(picked===k?null:k)}>
   <small>{d.toLocaleDateString(undefined,{weekday:'narrow'})}</small><b>{d.getDate()}</b>{busy(d)?<i/>:null}</button>})}</div>
  {picked&&<p className="ov-cal-note">{new Date(picked+'T00:00').toLocaleDateString(undefined,{weekday:'long',day:'numeric',month:'short'})}: {sel&&(sel.approved||sel.inspections||sel.models)?[sel.approved&&`${sel.approved} images approved`,sel.inspections&&`${sel.inspections} inspection(s) created`,sel.models&&`${sel.models} 3D model(s)`].filter(Boolean).join(' · '):'no activity'}</p>}
 </section>;
}

function LatestInspection({latest,go,selectSite}:{latest:NonNullable<Overview['latest']>;go:(p:string)=>void;selectSite:(id:number)=>void}){
 const open=(page:string)=>{if(latest.site)selectSite(latest.site.id);go(page)};
 return <section className="ov-card ov-latest"><div className="ov-card-head"><h3>Latest inspection</h3><small>{ago(latest.created_at)}</small></div>
  {latest.cover_media_id?<img className="ov-cover" src={thumb(latest.cover_media_id,720)} alt="" loading="lazy"/>:<div className="ov-cover ov-cover-empty"><RadioTower size={40}/></div>}
  <h4>{latest.label}</h4><small className="ov-muted">{latest.site?`${latest.site.code} · ${latest.site.name}`:''}</small>
  <div className="ov-progress-head"><span>Workflow</span><b>{latest.progress}%</b></div><KnobBar value={latest.progress}/>
  <div className="ov-steps-head"><h3>Next steps</h3><small>{latest.steps.filter(s=>s.done>=1).length}/{latest.steps.length} done</small></div>
  <ol className="ov-steps">{latest.steps.map((s,i)=><li key={s.key}><button onClick={()=>open(s.page)} className={s.done>=1?'done':s.done>0?'part':''}>
   <span className="ov-step-no">{s.done>=1?<Check size={14}/>:i+1}</span><span><b>{s.label}</b><small>{s.detail}</small></span>{s.done>0&&s.done<1&&<em>{Math.round(s.done*100)}%</em>}</button></li>)}</ol>
 </section>;
}

function TowerCards({sites,go,selectSite}:{sites:OvSite[];go:(p:string)=>void;selectSite:(id:number)=>void}){
 const row=useRef<HTMLDivElement>(null),scroll=(dir:number)=>row.current?.scrollBy({left:dir*(row.current.clientWidth*.8),behavior:'smooth'});
 return <section className="ov-section"><div className="ov-section-head"><h2>Tower sites</h2><div className="ov-arrows"><button onClick={()=>scroll(-1)} aria-label="Scroll left"><ChevronLeft size={18}/></button><button onClick={()=>scroll(1)} aria-label="Scroll right"><ChevronRight size={18}/></button></div></div>
  <div className="ov-tower-row" ref={row}>{sites.filter(s=>s.status!=='archived').map(s=>{const p=pct(s.stats.approved_images,s.stats.images);return <button key={s.id} className="ov-card ov-tower" onClick={()=>{selectSite(s.id);go('twin')}}>
   {s.cover_media_id?<img src={thumb(s.cover_media_id)} alt="" loading="lazy"/>:<div className="ov-tower-empty"><RadioTower size={34}/><small>No photos yet</small></div>}
   <h4>{s.code} · {s.name}</h4><small className="ov-muted">{s.region||'—'}{s.tower_type?` · ${s.tower_type}`:''}{s.height_m?` · ${s.height_m} m`:''}</small>
   <div className="ov-progress-head"><span>QC approved</span><b>{s.stats.images?`${p}%`:'—'}</b></div><KnobBar value={p}/>
   <div className="ov-tower-foot"><span><Clock size={14}/>{s.stats.last_inspection_at?`Inspected ${ago(s.stats.last_inspection_at)}`:'Not inspected yet'}</span>
    <span className="ov-chips">{s.stats.models>0&&<i title="3D model"><Box size={12}/>{s.stats.boxes_3d}</i>}{s.stats.items_with_defects>0&&<i className="warn" title="Items with defects"><ShieldAlert size={12}/>{s.stats.items_with_defects}</i>}</span></div>
  </button>})}</div></section>;
}

function JobsTable({jobs,go}:{jobs:Overview['jobs'];go:(p:string)=>void}){
 const [q,setQ]=useState(''),[page,setPage]=useState(0),per=5;
 const rows=jobs.filter(j=>!q||`${j.kind} ${j.inspection.label} ${j.site.code} ${j.status}`.toLowerCase().includes(q.toLowerCase()));
 const pages=Math.max(1,Math.ceil(rows.length/per)),shown=rows.slice(page*per,page*per+per);
 useEffect(()=>setPage(0),[q]);
 return <section className="ov-section"><div className="ov-section-head"><h2>Processing jobs</h2><div className="ov-search"><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search jobs"/><Search size={15}/></div></div>
  <div className="ov-card ov-table-card"><table className="ov-table"><thead><tr><th>Job</th><th>Tower</th><th>Progress</th><th>Status</th><th>Stage</th></tr></thead>
   <tbody>{shown.map(j=>{const st=JOB_STATUS[j.status]||{label:j.status.replaceAll('_',' '),cls:'wait'};return <tr key={j.id} onClick={()=>go(j.kind==='reconstruction'?'reconstruction':'processing')} title={j.detail}>
    <td><b>{j.kind==='reconstruction'?'3D reconstruction':j.kind==='detection'?'AI detection':j.kind.replaceAll('_',' ')}</b><small>{j.inspection.label}</small></td><td>{j.site.code}</td>
    <td><div className="ov-table-progress"><span>{j.progress}%</span><KnobBar value={j.progress}/></div></td><td><span className={'ov-pill '+st.cls}>{st.label}</span></td><td className="ov-muted">{j.stage.replaceAll('_',' ')}</td></tr>})}
    {!shown.length&&<tr><td colSpan={5} className="ov-muted">{jobs.length?'No jobs match.':'No processing jobs yet.'}</td></tr>}</tbody></table>
   <div className="ov-pager"><span>{rows.length?`${page*per+1}–${Math.min(rows.length,page*per+per)} of ${rows.length}`:'0'}</span><div>
    <button disabled={page===0} onClick={()=>setPage(page-1)} aria-label="Previous page"><ChevronLeft size={15}/></button>
    {[...Array(pages)].map((_,i)=><button key={i} className={i===page?'on':''} onClick={()=>setPage(i)}>{i+1}</button>).slice(Math.max(0,page-2),page+3)}
    <button disabled={page>=pages-1} onClick={()=>setPage(page+1)} aria-label="Next page"><ChevronRight size={15}/></button></div></div></div></section>;
}

/** Sidebar status card: where the GPU work runs and whether it is ready (/health). */
export function WorkerCard({collapsed}:{collapsed:boolean}){
 type H={ai_inference:string;model:string|null;gpu_worker:string;worker?:{online:boolean;gpu:string|null;seen_at:string}|null;storage:string};
 const [h,setH]=useState<H|null>(null);
 useEffect(()=>{let live=true;const load=()=>request<H>('/health').then(x=>live&&setH(x)).catch(()=>live&&setH(null));load();const t=setInterval(load,30000);return()=>{live=false;clearInterval(t)}},[]);
 const remote=h?.gpu_worker==='remote',online=remote?!!h?.worker?.online:h?.ai_inference==='ready';
 if(collapsed)return <div className={'ov-worker mini '+(online?'on':'off')} title={online?'GPU ready':remote?'GPU worker offline':'AI not ready'}><Cpu size={16}/></div>;
 return <div className={'ov-worker '+(online?'on':'off')}><div className="ov-worker-icon"><Cpu size={18}/></div>
  <b>{online?'GPU ready':remote?'GPU worker offline':'AI not ready'}</b>
  <small>{!h?'Checking…':remote?(online?`${h.worker?.gpu||'GPU'} · YOLO ${h.ai_inference==='ready'?'ready':h.ai_inference}`:'Start backend\\run-worker.ps1 on the GPU PC to run detection, 3D and AI checks.'):(online?`Local GPU · ${h.model||'YOLO'}`:`YOLO ${h.ai_inference.replaceAll('_',' ')}`)}</small>
  <span><i/>{remote?'Railway + local GPU worker':'Local server'}</span></div>;
}

export default function OverviewPage({go,selectSite}:{go:(page:string)=>void;selectSite:(id:number)=>void}){
 const [data,setData]=useState<Overview|null>(null),[error,setError]=useState('');
 useEffect(()=>{let live=true;const load=()=>request<Overview>('/api/v1/overview').then(d=>{if(live){setData(d);setError('')}}).catch(e=>live&&setError(String(e)));load();const t=setInterval(load,60000);return()=>{live=false;clearInterval(t)}},[]);
 if(error&&!data)return <div className="error">{error}</div>;
 if(!data)return <div className="ov-loading">Loading overview…</div>;
 const t=data.totals,qc=pct(t.approved_images,t.images);
 return <div className="ov-grid">
  <div className="ov-main">
   <div className="ov-top">
    <section className="ov-card ov-hero"><h3>QC approved images</h3><strong>{t.approved_images.toLocaleString()}</strong>
     <div className="ov-hero-row"><Ring value={qc}/><div><b>{t.images.toLocaleString()}</b><small>images in {t.inspections} inspection{t.inspections===1?'':'s'}</small></div></div>
     <div className="ov-hero-foot"><span><RadioTower size={13}/>{t.sites} towers</span><span><Layers size={13}/>{t.models} 3D models</span><span className={t.items_with_defects?'warn':''}><ShieldAlert size={13}/>{t.items_with_defects} defects</span></div></section>
    <Activity data={data.activity}/>
   </div>
   <TowerCards sites={data.sites} go={go} selectSite={selectSite}/>
   <JobsTable jobs={data.jobs} go={go}/>
  </div>
  <aside className="ov-rail"><WeekCalendar data={data.activity}/>{data.latest?<LatestInspection latest={data.latest} go={go} selectSite={selectSite}/>
   :<section className="ov-card ov-latest"><h3>Latest inspection</h3><p className="ov-muted">No inspections with photos yet. Create one under Inspections and upload drone media.</p></section>}</aside>
 </div>;
}
