import {useState} from 'react';
import {ArrowRight,CalendarDays,CheckCircle2,Clock3,Cpu,RadioTower,Search,UploadCloud} from 'lucide-react';
import {Inspection,Job,Site} from './api';
import './inspections.css';

type Props={inspections:Inspection[];sites:Site[];jobs:Job[];go:(page:string)=>void;selectSite:(id:number)=>void};
const inspectionCode=(id:number)=>`INS-${String(id).padStart(4,'0')}`;

export default function InspectionsPage({inspections,sites,jobs,go,selectSite}:Props){
 const [query,setQuery]=useState(''),[siteId,setSiteId]=useState('all');
 const siteOf=(inspection:Inspection)=>sites.find(site=>site.id===inspection.site_id);
 const jobOf=(inspection:Inspection)=>jobs.filter(job=>job.inspection_id===inspection.id).sort((a,b)=>b.id-a.id)[0]||null;
 const active=jobs.filter(job=>['queued','running','awaiting_worker'].includes(job.status)).length;
 const completed=new Set(jobs.filter(job=>job.status==='completed').map(job=>job.inspection_id)).size;
 const rows=[...inspections].sort((a,b)=>new Date(b.created_at).getTime()-new Date(a.created_at).getTime()).filter(inspection=>{const site=siteOf(inspection),text=`${inspectionCode(inspection.id)} ${inspection.label} ${site?.code||''} ${site?.name||''}`.toLowerCase();return(!query||text.includes(query.toLowerCase()))&&(siteId==='all'||inspection.site_id===+siteId)});
 const open=(inspection:Inspection,page:string)=>{selectSite(inspection.site_id);go(page)};
 const jobLabel=(job:Job|null)=>!job?'Not started':job.status==='awaiting_worker'?'Waiting for worker':job.status.replaceAll('_',' ');
 return <div className="is-page">
  <section className="is-summary">
   <div><span className="is-summary-icon blue"><CalendarDays size={18}/></span><span><small>Total inspections</small><b>{inspections.length}</b></span></div>
   <div><span className="is-summary-icon amber"><Cpu size={18}/></span><span><small>Active jobs</small><b>{active}</b></span></div>
   <div><span className="is-summary-icon green"><CheckCircle2 size={18}/></span><span><small>Processed</small><b>{completed}</b></span></div>
  </section>

  <section className="is-panel">
   <div className="is-panel-head"><div><h2>Inspection campaigns</h2><p>Survey records across all tower sites</p></div><span>{rows.length} shown</span></div>
   <div className="is-toolbar"><label><Search size={15}/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder="Search inspections"/></label><select value={siteId} onChange={event=>setSiteId(event.target.value)} aria-label="Filter by tower site"><option value="all">All tower sites</option>{sites.map(site=><option value={site.id} key={site.id}>{site.code} · {site.name}</option>)}</select></div>
   <div className="is-table-wrap"><table className="is-table"><thead><tr><th>Inspection</th><th>Tower site</th><th>Created</th><th>Workflow</th><th></th></tr></thead><tbody>{rows.map(inspection=>{const site=siteOf(inspection),job=jobOf(inspection),live=!!job&&['queued','running','awaiting_worker'].includes(job.status),done=job?.status==='completed';return <tr key={inspection.id}>
    <td><div className="is-inspection"><span><UploadCloud size={16}/></span><div><b>{inspection.label}</b><small>{inspectionCode(inspection.id)}</small></div></div></td>
    <td><b className="is-site"><RadioTower size={13}/>{site?.code||`Site ${inspection.site_id}`}</b><small>{site?.name||'Unknown tower'}{site?.region?` · ${site.region}`:''}</small></td>
    <td><span className="is-date">{new Date(inspection.created_at).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'})}</span></td>
    <td><span className={'is-status '+(live?'live':done?'done':job?.status==='failed'?'failed':'draft')}>{live?<Cpu size={12}/>:done?<CheckCircle2 size={12}/>:<Clock3 size={12}/>} {jobLabel(job)}</span>{job&&<small>{job.stage.replaceAll('_',' ')}{live?` · ${job.progress}%`:''}</small>}</td>
    <td><div className="is-actions"><button onClick={()=>open(inspection,'upload')}>Media</button><button className="is-open" onClick={()=>open(inspection,job?.kind==='reconstruction'?'reconstruction':done?'detection':'processing')} aria-label={`Open ${inspection.label}`}><ArrowRight size={15}/></button></div></td>
   </tr>})}</tbody></table>
   {!rows.length&&<div className="is-empty"><Search size={25}/><b>{inspections.length?'No matching inspections':'No inspections yet'}</b><span>{inspections.length?'Try another search or tower site.':'Use New Inspection above to create the first campaign.'}</span>{inspections.length&&<button onClick={()=>{setQuery('');setSiteId('all')}}>Clear filters</button>}</div>}</div>
  </section>
 </div>;
}
