import {ArrowRight,Box,CalendarDays,Database,MapPin,Plus,RadioTower,ShieldAlert,UploadCloud} from 'lucide-react';
import {Asset,Inspection,Site} from './api';
import './tower-detail.css';

type Props={site:Site|undefined;inspections:Inspection[];assets:Asset[];canInspect:boolean;canManage:boolean;go:(page:string)=>void;onCreateInspection:()=>void};
const inspectionCode=(id:number)=>`INS-${String(id).padStart(4,'0')}`;

export default function TowerDetailPage({site,inspections,assets,canInspect,canManage,go,onCreateInspection}:Props){
 if(!site)return <div className="td-empty"><RadioTower size={36}/><h2>No tower site selected</h2><p>Create or select a site to view its details.</p><button onClick={()=>go('projects')}>Open Projects &amp; Sites</button></div>;
 const siteInspections=inspections.filter(inspection=>inspection.site_id===site.id).sort((a,b)=>new Date(b.created_at).getTime()-new Date(a.created_at).getTime());
 const flagged=assets.filter(asset=>['Poor','Critical'].includes(asset.condition)).length;
 const last=siteInspections[0];
 const location=site.address||site.region||'Location not added';
 return <div className="td-page">
  <section className="td-identity">
   <div className="td-tower-icon"><RadioTower size={31}/></div>
   <div className="td-title"><span>{site.code}</span><h2>{site.name}</h2><p><MapPin size={13}/>{location}</p></div>
   <div className="td-status"><i className={site.status}/><span><small>SITE STATUS</small><b>{site.status.replaceAll('_',' ')}</b></span></div>
  </section>

  <section className="td-stats">
   <div><small>Known height</small><b>{site.height_m??'—'}{site.height_m!=null&&<em>m</em>}</b></div>
   <div><small>Observed equipment</small><b>{assets.length}</b></div>
   <div><small>Inspection campaigns</small><b>{siteInspections.length}</b></div>
   <div className={flagged?'warn':''}><small>Flagged equipment</small><b>{flagged}</b></div>
  </section>

  <div className="td-grid">
   <section className="td-panel td-information"><div className="td-panel-head"><div><h2>Site information</h2><p>Core tower record and location details</p></div></div><dl>
    <div><dt>Tower type</dt><dd>{site.tower_type||'Not specified'}</dd></div><div><dt>Operator</dt><dd>{site.operator||'Not specified'}</dd></div><div><dt>Region</dt><dd>{site.region||'Not specified'}</dd></div><div><dt>Coordinates</dt><dd>{site.latitude!=null&&site.longitude!=null?`${site.latitude.toFixed(6)}, ${site.longitude.toFixed(6)}`:'Not mapped'}</dd></div><div className="wide"><dt>Address</dt><dd>{site.address||'No address added'}</dd></div><div className="wide"><dt>Notes</dt><dd>{site.notes||'No site notes'}</dd></div>
   </dl></section>

   <section className="td-panel td-shortcuts"><div className="td-panel-head"><div><h2>Quick access</h2><p>Continue working with this tower</p></div></div><div>
    <button onClick={()=>go('twin')}><span className="blue"><Box size={18}/></span><span><b>3D Digital Twin</b><small>Open tower model</small></span><ArrowRight size={15}/></button>
    <button onClick={()=>go('assets')}><span className="violet"><Database size={18}/></span><span><b>Asset inventory</b><small>{assets.length} observed items</small></span><ArrowRight size={15}/></button>
    <button onClick={()=>go('defects')}><span className="orange"><ShieldAlert size={18}/></span><span><b>Condition &amp; defects</b><small>{flagged} flagged items</small></span><ArrowRight size={15}/></button>
    {canInspect&&<button onClick={()=>go('upload')}><span className="green"><UploadCloud size={18}/></span><span><b>Upload survey</b><small>Add inspection media</small></span><ArrowRight size={15}/></button>}
   </div></section>
  </div>

  <section className="td-panel td-inspections"><div className="td-panel-head"><div><h2>Recent inspections</h2><p>{last?`Last campaign created ${new Date(last.created_at).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'})}`:'No campaigns created for this tower'}</p></div>{canManage&&<button onClick={onCreateInspection}><Plus size={15}/>New inspection</button>}</div>
   {siteInspections.length?<div className="td-inspection-list">{siteInspections.slice(0,5).map(inspection=><button key={inspection.id} onClick={()=>go('inspections')}><span className="td-inspection-date"><CalendarDays size={15}/></span><span><b>{inspection.label}</b><small>{inspectionCode(inspection.id)} · {new Date(inspection.created_at).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'})}</small></span><em>{inspection.status.replaceAll('_',' ')}</em><ArrowRight size={15}/></button>)}</div>
   :<div className="td-no-inspections"><CalendarDays size={26}/><span><b>No inspection campaigns</b><small>Create a campaign before uploading drone media.</small></span></div>}
   {siteInspections.length>5&&<button className="td-view-all" onClick={()=>go('inspections')}>View all {siteInspections.length} inspections <ArrowRight size={14}/></button>}
  </section>
 </div>;
}
