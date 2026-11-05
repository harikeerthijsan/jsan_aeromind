import {FormEvent,useMemo,useState} from 'react';
import {ArrowUpRight,Check,ClipboardCheck,Edit3,KeyRound,LockKeyhole,Mail,MapPin,Monitor,RadioTower,Save,ShieldCheck,UserRound,Users,Video,X} from 'lucide-react';
import {AuthRole,AuthUser,Inspection,Job,Site,json,request} from './api';
import {roleAtLeast,roleLabel} from './Auth';
import './profile.css';

type Props={user:AuthUser;sites:Site[];inspections:Inspection[];jobs:Job[];go:(page:string)=>void;onUserChange:(user:AuthUser)=>void};
type AccessStep={role:AuthRole;label:string;detail:string};
const access:AccessStep[]=[
 {role:'viewer',label:'Observe',detail:'Sites, digital twins, inventory and reports'},
 {role:'inspector',label:'Verify',detail:'Uploads, AI detection, QC and defect decisions'},
 {role:'manager',label:'Coordinate',detail:'Tower records and inspection campaigns'},
 {role:'admin',label:'Govern',detail:'People, models, integrations and workspace policy'}
];
const initials=(name:string)=>name.split(/\s+/).filter(Boolean).map(part=>part[0]).join('').slice(0,2).toUpperCase();
const date=(value:string|null,withTime=false)=>value?new Date(value).toLocaleString(undefined,withTime?{dateStyle:'medium',timeStyle:'short'}:{dateStyle:'medium'}):'Not recorded';
const browser=()=>{const ua=navigator.userAgent;return /Edg\//.test(ua)?'Microsoft Edge':/Firefox\//.test(ua)?'Firefox':/Chrome\//.test(ua)?'Chrome':/Safari\//.test(ua)?'Safari':'Web browser'};

export default function ProfilePage({user,sites,inspections,jobs,go,onUserChange}:Props){
 const [editing,setEditing]=useState(false),[name,setName]=useState(user.name),[saving,setSaving]=useState(false),[error,setError]=useState(''),[copied,setCopied]=useState(false);
 const activeJobs=jobs.filter(job=>['queued','running','awaiting_worker'].includes(job.status)).length;
 const located=sites.filter(site=>site.latitude!=null&&site.longitude!=null).length;
 const device=useMemo(()=>({browser:browser(),zone:Intl.DateTimeFormat().resolvedOptions().timeZone||'Local time'}),[]);
 const paths=roleAtLeast(user.role,'admin')?[{label:'Manage workspace users',page:'users',icon:Users},{label:'Review model readiness',page:'models',icon:ShieldCheck}]
  :roleAtLeast(user.role,'manager')?[{label:'Manage tower sites',page:'projects',icon:RadioTower},{label:'Create an inspection',page:'inspections',icon:Video}]
  :roleAtLeast(user.role,'inspector')?[{label:'Continue QC review',page:'review',icon:ClipboardCheck},{label:'Open AI detection',page:'detection',icon:ShieldCheck}]
  :[{label:'Explore tower sites',page:'projects',icon:RadioTower},{label:'Open asset inventory',page:'assets',icon:ClipboardCheck}];
 async function save(event:FormEvent){event.preventDefault();const next=name.trim();if(next.length<2)return;setSaving(true);setError('');try{const updated=await request<AuthUser>('/api/v1/auth/me',json('PATCH',{name:next}));onUserChange(updated);setName(updated.name);setEditing(false)}catch(e){setError(e instanceof Error?e.message:String(e))}finally{setSaving(false)}}
 async function copyEmail(){try{await navigator.clipboard.writeText(user.email);setCopied(true);setTimeout(()=>setCopied(false),1600)}catch{setError('Could not copy the email address.')}}
 return <div className="pf-shell">
  <aside className="pf-spine">
   <div className="pf-pass">
    <div className="pf-pass-cap"><span>AEROMIND / IDENTITY</span><b>AM-{String(user.id).padStart(6,'0')}</b></div>
    <div className="pf-monogram" aria-hidden="true"><span>{initials(user.name)}</span><i/><i/><i/><i/></div>
    <div className="pf-state"><i/>ACCOUNT {user.active?'ACTIVE':'INACTIVE'}</div>
    {editing?<form className="pf-name-form" onSubmit={save}><label>Display name<input value={name} onChange={e=>setName(e.target.value)} minLength={2} maxLength={160} autoFocus/></label><div><button type="button" onClick={()=>{setEditing(false);setName(user.name);setError('')}}><X size={14}/>Cancel</button><button className="save" disabled={saving||name.trim().length<2}><Save size={14}/>{saving?'Saving':'Save'}</button></div></form>
    :<div className="pf-name"><small>OPERATOR RECORD</small><h2>{user.name}</h2><button onClick={()=>setEditing(true)}><Edit3 size={14}/>Edit display name</button></div>}
    {error&&<div className="pf-error">{error}</div>}
    <div className="pf-contact"><span><Mail size={14}/><span><small>WORK EMAIL</small><b>{user.email}</b></span></span><button onClick={copyEmail}>{copied?<><Check size={13}/>Copied</>:'Copy'}</button></div>
    <div className="pf-role-stamp"><small>CLEARANCE</small><strong>{roleLabel(user.role)}</strong><span>{access.find(step=>step.role===user.role)?.label} authority</span></div>
    <div className="pf-pass-foot"><span>ISSUED<br/><b>{date(user.created_at)}</b></span><RadioTower size={35}/><em>TV/{new Date(user.created_at).getFullYear()}/{String(user.id).padStart(4,'0')}</em></div>
   </div>
   <p className="pf-spine-note"><LockKeyhole size={14}/>Identity fields come from the authenticated workspace record. Access changes require an administrator.</p>
  </aside>

  <div className="pf-record">
   <header className="pf-lede"><div><span className="pf-index">01 / OPERATOR FOLIO</span><h2>One identity.<br/><em>A precise field of action.</em></h2><p>This page describes what your account can reach now—without simulated activity, vanity scores or invented achievements.</p></div><div className="pf-seal"><ShieldCheck/><span>SERVER<br/>VERIFIED</span><small>{user.active?'IN FORCE':'SUSPENDED'}</small></div></header>

   <section className="pf-reach"><div className="pf-section-label"><span>02</span><div><b>Workspace reach</b><small>Live scope, not personal performance</small></div></div><div className="pf-tallies">
    <div><strong>{sites.length}</strong><span>Tower sites</span><small><MapPin size={12}/>{located} geolocated</small></div>
    <div><strong>{inspections.length}</strong><span>Inspections</span><small><Video size={12}/>Visible to this workspace</small></div>
    <div><strong>{activeJobs}</strong><span>Live jobs</span><small><Monitor size={12}/>{activeJobs?'Needs awareness':'Queue is clear'}</small></div>
   </div></section>

   <section className="pf-access"><div className="pf-section-label"><span>03</span><div><b>Access ladder</b><small>Your current boundary is marked in ink</small></div></div><div className="pf-access-track">{access.map((step,index)=>{const enabled=roleAtLeast(user.role,step.role),current=user.role===step.role;return <div key={step.role} className={(enabled?'enabled ':'')+(current?'current':'')}><span>{String(index+1).padStart(2,'0')}</span><i>{enabled?<Check size={13}/>:null}</i><div><b>{step.label}</b><small>{step.detail}</small></div>{current&&<em>YOU ARE HERE</em>}</div>})}</div></section>

   <div className="pf-bottom">
    <section className="pf-facts"><div className="pf-section-label"><span>04</span><div><b>Account ledger</b><small>Facts attached to this session</small></div></div><dl>
     <div><dt>Last sign-in</dt><dd>{date(user.last_login,true)}</dd></div><div><dt>Account opened</dt><dd>{date(user.created_at)}</dd></div><div><dt>Current client</dt><dd>{device.browser}</dd></div><div><dt>Local time zone</dt><dd>{device.zone}</dd></div><div><dt>Account state</dt><dd><i className={user.active?'ok':''}/>{user.active?'Active':'Inactive'}</dd></div><div><dt>Role changes</dt><dd>Administrator controlled</dd></div>
    </dl></section>
    <section className="pf-paths"><div className="pf-section-label"><span>05</span><div><b>Next stations</b><small>Shortcuts shaped by your role</small></div></div>{paths.map(path=><button key={path.page} onClick={()=>go(path.page)}><path.icon size={18}/><span>{path.label}<small>Open workspace module</small></span><ArrowUpRight size={16}/></button>)}<div className="pf-security"><KeyRound size={16}/><span><b>Protected session</b><small>Bearer authentication with a revocable server session.</small></span></div></section>
   </div>
  </div>
 </div>;
}
