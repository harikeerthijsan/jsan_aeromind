import React,{useEffect,useState} from 'react';
import {RadioTower,Eye,EyeOff,ShieldCheck,ArrowRight,CheckCircle2,Users,UserPlus,X,KeyRound} from 'lucide-react';
import {AuthRole,AuthUser,json,request} from './api';

export const ROLE_LEVEL:Record<AuthRole,number>={viewer:0,inspector:1,manager:2,admin:3};
export const roleAtLeast=(role:AuthRole,required:AuthRole)=>ROLE_LEVEL[role]>=ROLE_LEVEL[required];
export const roleLabel=(role:AuthRole)=>role.charAt(0).toUpperCase()+role.slice(1);

type AuthConfig={demo:boolean;accounts:{email:string;name:string;role:AuthRole}[];demo_password:string|null};
type LoginResult={token:string;expires_at:string;user:AuthUser};

export function LoginPage({onLogin}:{onLogin:(user:AuthUser,token:string)=>void}){
 const [email,setEmail]=useState(''),[password,setPassword]=useState(''),[remember,setRemember]=useState(true),[show,setShow]=useState(false),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const [config,setConfig]=useState<AuthConfig|null>(null);
 useEffect(()=>{request<AuthConfig>('/api/v1/auth/config').then(setConfig).catch(()=>setConfig({demo:false,accounts:[],demo_password:null}))},[]);
 async function submit(event:React.FormEvent){event.preventDefault();setBusy(true);setError('');try{const result=await request<LoginResult>('/api/v1/auth/login',json('POST',{email,password,remember}));onLogin(result.user,result.token)}catch(e){setError(e instanceof Error?e.message:'Unable to sign in')}finally{setBusy(false)}}
 function chooseDemo(account:AuthConfig['accounts'][number]){setEmail(account.email);setPassword(config?.demo_password||'');setError('')}
 return <main className="auth-shell">
  <section className="auth-visual">
   <div className="auth-grid"/><div className="auth-orbit orbit-one"/><div className="auth-orbit orbit-two"/>
   <div className="auth-brand"><span><RadioTower size={25}/></span><strong>AERO<em>MIND</em></strong></div>
   <div className="auth-story"><span className="auth-kicker">INFRASTRUCTURE INTELLIGENCE</span><h1>Every tower.<br/>One trusted view.</h1><p>Secure inspection operations, AI-assisted review and digital-twin intelligence—connected in one workspace.</p><div className="auth-benefits"><span><CheckCircle2/>Role-controlled access</span><span><CheckCircle2/>Traceable inspection decisions</span><span><CheckCircle2/>Protected tower data</span></div></div>
   <div className="auth-trust"><ShieldCheck size={18}/><span>Enterprise-ready access control</span></div>
  </section>
  <section className="auth-form-side"><div className="auth-form-wrap">
   <div className="auth-mobile-brand"><RadioTower/><strong>AERO<em>MIND</em></strong></div>
   <div className="auth-heading"><span>WELCOME BACK</span><h2>Sign in to your workspace</h2><p>Use your AeroMind account to continue.</p></div>
   <form className="auth-form" onSubmit={submit}>
    <label>Work email<input type="email" autoComplete="username" value={email} onChange={e=>setEmail(e.target.value)} placeholder="name@company.com" required autoFocus/></label>
    <label>Password<div className="password-field"><input type={show?'text':'password'} autoComplete="current-password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Enter your password" required/><button type="button" onClick={()=>setShow(!show)} aria-label={show?'Hide password':'Show password'}>{show?<EyeOff/>:<Eye/>}</button></div></label>
    <label className="remember-row"><input type="checkbox" checked={remember} onChange={e=>setRemember(e.target.checked)}/><span>Keep me signed in for 7 days</span></label>
    {error&&<div className="auth-error">{error}</div>}
    <button className="auth-submit" disabled={busy}>{busy?'Signing in…':<>Sign in <ArrowRight size={18}/></>}</button>
   </form>
   {config?.demo&&<div className="demo-access"><div><b>Demo role access</b><small>Select an account to test its permissions.</small></div><div className="demo-grid">{config.accounts.map(a=><button type="button" key={a.email} onClick={()=>chooseDemo(a)}><span>{roleLabel(a.role)}</span><small>{a.name}</small></button>)}</div><p>Demo password: <code>{config.demo_password}</code></p></div>}
   <p className="auth-footer">Protected by encrypted, revocable sessions. Contact an administrator if you need access.</p>
  </div></section>
 </main>
}

const roleDescriptions:Record<AuthRole,string>={viewer:'Read-only access to sites, twins, inventory and reports.',inspector:'Viewer access plus uploads, AI review, QC and defect decisions.',manager:'Inspector access plus site and inspection management.',admin:'Full access, configuration and user administration.'};

export function UsersPage({current}:{current:AuthUser}){
 const [users,setUsers]=useState<AuthUser[]>([]),[loading,setLoading]=useState(true),[error,setError]=useState(''),[showCreate,setShowCreate]=useState(false);
 const [draft,setDraft]=useState({name:'',email:'',role:'viewer' as AuthRole,password:''}),[saving,setSaving]=useState(false);
 const load=()=>{setLoading(true);request<AuthUser[]>('/api/v1/users').then(setUsers).catch(e=>setError(String(e))).finally(()=>setLoading(false))};
 useEffect(load,[]);
 async function create(event:React.FormEvent){event.preventDefault();setSaving(true);setError('');try{await request('/api/v1/users',json('POST',draft));setDraft({name:'',email:'',role:'viewer',password:''});setShowCreate(false);load()}catch(e){setError(e instanceof Error?e.message:String(e))}finally{setSaving(false)}}
 async function update(user:AuthUser,changes:Partial<{name:string;role:AuthRole;active:boolean;password:string}>){setError('');try{const updated=await request<AuthUser>(`/api/v1/users/${user.id}`,json('PATCH',changes));setUsers(list=>list.map(item=>item.id===updated.id?updated:item))}catch(e){setError(e instanceof Error?e.message:String(e))}}
 function resetPassword(user:AuthUser){const password=prompt(`Set a new password for ${user.name} (minimum 10 characters)`);if(password)update(user,{password})}
 return <div className="users-page">
  <section className="role-summary">{(['viewer','inspector','manager','admin'] as AuthRole[]).map(role=><div key={role} className={`role-card role-${role}`}><span>{roleLabel(role)}</span><b>{users.filter(u=>u.active&&u.role===role).length}</b><p>{roleDescriptions[role]}</p></div>)}</section>
  <section className="panel users-panel"><div className="panel-head"><div><h2>Workspace users</h2><p className="muted">Manage access without sharing a common password.</p></div><button className="btn primary" onClick={()=>setShowCreate(true)}><UserPlus size={16}/> Add user</button></div>
   {error&&<div className="auth-error users-error">{error}</div>}
   {loading?<div className="users-loading">Loading users…</div>:<div className="users-table-wrap"><table className="users-table"><thead><tr><th>User</th><th>Role</th><th>Status</th><th>Last sign in</th><th>Actions</th></tr></thead><tbody>{users.map(user=><tr key={user.id}><td><div className="user-cell"><span>{user.name.split(/\s+/).map(v=>v[0]).join('').slice(0,2).toUpperCase()}</span><div><b>{user.name}{user.id===current.id&&<small className="you-label">You</small>}</b><small>{user.email}</small></div></div></td><td><select value={user.role} disabled={user.id===current.id} onChange={e=>update(user,{role:e.target.value as AuthRole})}>{(['viewer','inspector','manager','admin'] as AuthRole[]).map(role=><option value={role} key={role}>{roleLabel(role)}</option>)}</select></td><td><button className={`status-toggle ${user.active?'active':''}`} disabled={user.id===current.id} onClick={()=>update(user,{active:!user.active})}><i/>{user.active?'Active':'Inactive'}</button></td><td>{user.last_login?new Date(user.last_login).toLocaleString():'Never'}</td><td><button className="icon-text-btn" onClick={()=>resetPassword(user)}><KeyRound size={15}/> Reset password</button></td></tr>)}</tbody></table></div>}
  </section>
  {showCreate&&<div className="modal-backdrop" onMouseDown={()=>setShowCreate(false)}><form className="user-modal" onSubmit={create} onMouseDown={e=>e.stopPropagation()}><div className="modal-title"><div><Users/><span><b>Add workspace user</b><small>Assign the minimum role they need.</small></span></div><button type="button" onClick={()=>setShowCreate(false)}><X/></button></div><label>Full name<input value={draft.name} onChange={e=>setDraft({...draft,name:e.target.value})} required minLength={2}/></label><label>Work email<input type="email" value={draft.email} onChange={e=>setDraft({...draft,email:e.target.value})} required/></label><label>Role<select value={draft.role} onChange={e=>setDraft({...draft,role:e.target.value as AuthRole})}>{(['viewer','inspector','manager','admin'] as AuthRole[]).map(role=><option value={role} key={role}>{roleLabel(role)} — {roleDescriptions[role]}</option>)}</select></label><label>Temporary password<input type="password" value={draft.password} onChange={e=>setDraft({...draft,password:e.target.value})} required minLength={10}/><small>Minimum 10 characters. Share it securely.</small></label><div className="modal-actions"><button type="button" className="btn outline" onClick={()=>setShowCreate(false)}>Cancel</button><button className="btn primary" disabled={saving}>{saving?'Creating…':'Create user'}</button></div></form></div>}
 </div>
}
