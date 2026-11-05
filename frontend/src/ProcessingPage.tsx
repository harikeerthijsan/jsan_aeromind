import {useState} from 'react';
import {AlertTriangle,ArrowRight,Box,CheckCircle2,Clock3,Cpu,RefreshCw,Search} from 'lucide-react';
import {Inspection,Job,Site} from './api';
import './processing.css';

type Filter='all'|'active'|'completed'|'failed';
type Props={jobs:Job[];inspections:Inspection[];sites:Site[];onRefresh:()=>void;go:(page:string)=>void;selectSite:(id:number)=>void};
const jobName=(job:Job)=>job.kind==='reconstruction'?'3D reconstruction':job.kind==='detection'?'AI detection':(job.kind||'Processing job').replaceAll('_',' ');

export default function ProcessingPage({jobs,inspections,sites,onRefresh,go,selectSite}:Props){
 const [query,setQuery]=useState(''),[filter,setFilter]=useState<Filter>('all');
 const inspectionOf=(job:Job)=>inspections.find(inspection=>inspection.id===job.inspection_id);
 const siteOf=(job:Job)=>{const inspection=inspectionOf(job);return sites.find(site=>site.id===inspection?.site_id)};
 const active=jobs.filter(job=>['queued','running','awaiting_worker'].includes(job.status)).length,completed=jobs.filter(job=>job.status==='completed').length,failed=jobs.filter(job=>job.status==='failed').length;
 const rows=[...jobs].sort((a,b)=>b.id-a.id).filter(job=>{const inspection=inspectionOf(job),site=siteOf(job),text=`${job.id} ${jobName(job)} ${inspection?.label||''} ${site?.code||''} ${job.stage} ${job.detail}`.toLowerCase();const matches=filter==='all'||filter==='active'&&['queued','running','awaiting_worker'].includes(job.status)||job.status===filter;return(!query||text.includes(query.toLowerCase()))&&matches});
 const open=(job:Job)=>{const inspection=inspectionOf(job);if(inspection)selectSite(inspection.site_id);go(job.kind==='reconstruction'?'reconstruction':job.kind==='detection'?'detection':'processing')};
 return <div className="pc-page">
  <section className="pc-summary">
   <button className={filter==='all'?'on':''} onClick={()=>setFilter('all')}><span className="all"><Cpu size={18}/></span><div><small>All jobs</small><b>{jobs.length}</b></div></button>
   <button className={filter==='active'?'on':''} onClick={()=>setFilter(filter==='active'?'all':'active')}><span className="active"><Clock3 size={18}/></span><div><small>Active</small><b>{active}</b></div></button>
   <button className={filter==='completed'?'on':''} onClick={()=>setFilter(filter==='completed'?'all':'completed')}><span className="complete"><CheckCircle2 size={18}/></span><div><small>Completed</small><b>{completed}</b></div></button>
   <button className={filter==='failed'?'on':''} onClick={()=>setFilter(filter==='failed'?'all':'failed')}><span className="failed"><AlertTriangle size={18}/></span><div><small>Failed</small><b>{failed}</b></div></button>
  </section>

  <section className="pc-panel">
   <div className="pc-panel-head"><div><h2>Processing jobs</h2><p>Detection and reconstruction work across inspections</p></div><button onClick={onRefresh}><RefreshCw size={14}/>Refresh</button></div>
   <div className="pc-toolbar"><label><Search size={15}/><input value={query} onChange={event=>setQuery(event.target.value)} placeholder="Search jobs, inspections or towers"/></label><div>{(['all','active','completed','failed'] as Filter[]).map(value=><button key={value} className={filter===value?'on':''} onClick={()=>setFilter(value)}>{value[0].toUpperCase()+value.slice(1)}</button>)}</div></div>
   <div className="pc-list">{rows.map(job=>{const inspection=inspectionOf(job),site=siteOf(job),isActive=['queued','running','awaiting_worker'].includes(job.status),status=job.status==='awaiting_worker'?'waiting':job.status.replaceAll('_',' ');return <article key={job.id} className={'pc-job '+(isActive?'active':job.status)}>
    <span className="pc-job-icon">{job.kind==='reconstruction'?<Box size={19}/>:<Cpu size={19}/>}</span>
    <div className="pc-job-main"><div><b>{jobName(job)}</b><span>JOB-{String(job.id).padStart(4,'0')}</span></div><p>{inspection?.label||`Inspection ${job.inspection_id}`}{site&&<> · {site.code} · {site.name}</>}</p><small>{job.detail||'No processing detail available.'}</small></div>
    <div className="pc-job-stage"><span>{job.stage.replaceAll('_',' ')}</span><div><i style={{width:`${Math.max(job.progress,isActive?3:0)}%`}}/></div><small>{job.progress}%</small></div>
    <span className={'pc-status '+(isActive?'active':job.status)}><i/>{status}</span>
    <button className="pc-open" onClick={()=>open(job)} aria-label={`Open ${jobName(job)}`}><ArrowRight size={15}/></button>
   </article>})}
   {!rows.length&&<div className="pc-empty"><Cpu size={29}/><b>{jobs.length?'No jobs match these filters':'No processing jobs yet'}</b><span>{jobs.length?'Try another search or status.':'Upload inspection media, then start detection or reconstruction.'}</span>{jobs.length?<button onClick={()=>{setQuery('');setFilter('all')}}>Clear filters</button>:<button onClick={()=>go('upload')}>Open Media Upload</button>}</div>}</div>
  </section>

  <section className="pc-flow"><div><h3>Workflow stages</h3><p>Jobs move through only the stages required by their processing type.</p></div><ol>{['Validation','Frame extraction','AI detection','Human QA','3D reconstruction','Registration'].map((stage,index)=><li key={stage}><span>{index+1}</span>{stage}</li>)}</ol></section>
 </div>;
}
