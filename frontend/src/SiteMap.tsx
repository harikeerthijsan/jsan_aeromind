import {useEffect,useRef,useState} from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {Crosshair,Loader2,MapPin,Maximize2,Search,X} from 'lucide-react';

/** Leaflet maps for Projects & Sites: OpenStreetMap tiles (no key; attribution required), div-based markers styled in sites.css. */
export type MapSite={id:number;code:string;name:string;status:string;latitude?:number|null;longitude?:number|null};
const TILES='https://tile.openstreetmap.org/{z}/{x}/{y}.png',ATTRIBUTION='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors';
const HOME:L.LatLngTuple=[22.5,79.0],HOME_ZOOM=5;  // India, until there are towers to fit
const located=(s:MapSite)=>s.latitude!=null&&s.longitude!=null;
const round6=(v:number)=>Math.round(v*1e6)/1e6;
const esc=(t:string)=>t.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));

/** One Leaflet map bound to a div for the component's lifetime, kept sized to its container (modals and panels resize after mount). */
function useMap(opts:L.MapOptions={}){
 const el=useRef<HTMLDivElement>(null),map=useRef<L.Map|null>(null),[ready,setReady]=useState(false);
 useEffect(()=>{if(!el.current)return;
  const m=L.map(el.current,{center:HOME,zoom:HOME_ZOOM,worldCopyJump:true,...opts});
  L.tileLayer(TILES,{maxZoom:19,attribution:ATTRIBUTION}).addTo(m);map.current=m;setReady(true);
  const ro=new ResizeObserver(()=>m.invalidateSize());ro.observe(el.current);
  return()=>{ro.disconnect();m.remove();map.current=null;setReady(false)}},[]);
 return {el,map,ready};
}

const towerSvg=`<svg viewBox="0 0 44 58" aria-hidden="true">
 <ellipse class="st-tower-shadow" cx="22" cy="53" rx="17" ry="3"/>
 <path class="st-tower-pad" d="M5 50.5 15 46.5 39 49.5 29 54.5Z"/>
 <g class="st-tower-back"><path d="M15 48 22 5 35 49M22 5 9 52"/><path d="m17 38 15 2-13-12 11 1-9-10 7 1M15 47l18-7M17 38l-8 14M19 28 35 49M21 19 30 29"/></g>
 <g class="st-tower-front"><path d="M9 52 22 5 29 54M9 52h20"/><path d="m12 42 15 2-12-12 10 2-8-12 6 2M12 42l17 12M15 32 9 22M17 22l8 12M20 12l3 12"/></g>
 <g class="st-tower-depth"><path d="M9 52 15 48M29 54l6-5M12 42l5-4M27 44l5-4M15 32l4-4M25 34l5-5M17 22l4-3M23 24l5-4"/></g>
 <path class="st-tower-platform" d="M15 18.5 20 16l10 1.5-5 3Z"/><path class="st-tower-mast" d="M21.3 6V1.5h1.5V6"/><circle class="st-tower-beacon" cx="22.05" cy="1.7" r="1.8"/>
 </svg>`;
const pinIcon=(s:MapSite,on:boolean)=>L.divIcon({className:'st-pin st-tower-pin'+(on?' on':''),iconSize:[44,58],iconAnchor:[22,53],
 html:`<span class="st-tower ${esc(s.status)}">${towerSvg}</span><b>${esc(s.code)}</b>`});

/** Every tower with coordinates; clicking one calls onSelect (the page opens its detail panel). */
export function SitesMap({sites,selectedId,onSelect}:{sites:MapSite[];selectedId:number|null;onSelect:(id:number)=>void}){
 const {el,map,ready}=useMap(),layer=useRef<L.LayerGroup|null>(null),fitted=useRef('');
 const fit=()=>{const m=map.current,pts=sites.filter(located).map(s=>[s.latitude!,s.longitude!] as L.LatLngTuple);if(!m)return;
  if(pts.length>1)m.fitBounds(pts,{padding:[50,50],maxZoom:14});else if(pts.length===1)m.setView(pts[0],13);else m.setView(HOME,HOME_ZOOM)};
 useEffect(()=>{const m=map.current;if(!m||!ready)return;
  layer.current?.remove();const g=L.layerGroup().addTo(m);layer.current=g;
  for(const s of sites.filter(located))
   L.marker([s.latitude!,s.longitude!],{icon:pinIcon(s,s.id===selectedId),riseOnHover:true,zIndexOffset:s.id===selectedId?1000:0,keyboard:true,title:`${s.code} · ${s.name}`})
    .bindTooltip(`<b>${esc(s.code)}</b> · ${esc(s.name)}`,{direction:'top',offset:[0,-45]}).on('click',()=>onSelect(s.id)).addTo(g);
  const key=sites.filter(located).map(s=>`${s.id}:${s.latitude},${s.longitude}`).join('|');
  if(key!==fitted.current){fitted.current=key;fit()}  // re-fit only when the set of locations changes, not on selection
  declutter();
 },[sites,selectedId,ready]);
 // nearby towers' ID labels would overlap at low zoom: hide a label that collides with one already shown (selected first);
 // the pin stays and its name shows on hover. Zooming in brings the labels back.
 const declutter=()=>{const el0=el.current;if(!el0)return;const shown:DOMRect[]=[];
  const labels=[...el0.querySelectorAll<HTMLElement>('.st-pin b')].sort((a,b)=>Number(b.parentElement!.classList.contains('on'))-Number(a.parentElement!.classList.contains('on')));
  for(const b of labels){b.style.visibility='';const r=b.getBoundingClientRect();
   const hit=shown.some(o=>r.left<o.right+4&&r.right>o.left-4&&r.top<o.bottom+2&&r.bottom>o.top-2);
   if(hit)b.style.visibility='hidden';else shown.push(r)}};
 useEffect(()=>{const m=map.current;if(!m||!ready)return;m.on('zoomend moveend',declutter);return()=>{m.off('zoomend moveend',declutter)}},[ready]);
 return <div className="st-map-wrap"><div ref={el} className="st-map"/>
  <button type="button" className="st-map-btn" title="Show all towers" onClick={fit}><Maximize2 size={15}/></button>
  <div className="st-map-legend">{['active','maintenance','decommissioned','archived'].map(s=><span key={s}><i className={s}/>{s[0].toUpperCase()+s.slice(1)}</span>)}</div>
 </div>;
}

/** Place search with OpenStreetMap's Nominatim. Its public-server policy allows at most 1 request per second and no
 * search-as-you-type, so this searches on Enter / the button only, spaces requests out and caches answers.
 * https://operations.osmfoundation.org/policies/nominatim/ */
type Place={display_name:string;lat:string;lon:string;type?:string;category?:string;boundingbox?:[string,string,string,string]};
const placeCache=new Map<string,Place[]>();let lastSearch=0;
async function searchPlaces(q:string,near?:L.LatLngBounds):Promise<Place[]>{
 const key=q.trim().toLowerCase();if(placeCache.has(key))return placeCache.get(key)!;
 const wait=lastSearch+1100-Date.now();if(wait>0)await new Promise(r=>setTimeout(r,wait));lastSearch=Date.now();
 const u=new URLSearchParams({q:q.trim(),format:'jsonv2',limit:'6','accept-language':navigator.language||'en'});
 if(near)u.set('viewbox',near.toBBoxString());  // prefer places near the current map view (a boost, not a filter)
 const r=await fetch('https://nominatim.openstreetmap.org/search?'+u.toString());
 if(!r.ok)throw Error(`place search failed (${r.status})`);
 const out=await r.json() as Place[];placeCache.set(key,out);return out}
/** "17.385, 78.4867" / "17.385 78.4867" -> [lat, lon] */
function parseCoords(q:string):[number,number]|null{const m=q.trim().match(/^(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)$/);if(!m)return null;
 const a=+m[1],b=+m[2];return Math.abs(a)<=90&&Math.abs(b)<=180?[a,b]:null}

/** Click (or drag the pin) to set a tower's coordinates. Other towers are shown faintly for orientation. */
export function LocationPicker({lat,lon,onPick,others=[]}:{lat:number|null;lon:number|null;onPick:(lat:number,lon:number)=>void;others?:MapSite[]}){
 const {el,map,ready}=useMap(),pin=useRef<L.Marker|null>(null),pick=useRef(onPick),[locating,setLocating]=useState(''),centred=useRef(false);
 const [query,setQuery]=useState(''),[results,setResults]=useState<Place[]|null>(null),[searching,setSearching]=useState(false),[found,setFound]=useState(''),spot=useRef<L.CircleMarker|null>(null);
 pick.current=onPick;
 const pinTo=(p:L.LatLngExpression)=>{const m=map.current!;
  if(!pin.current){pin.current=L.marker(p,{draggable:true,autoPan:true,icon:L.divIcon({className:'st-pick',html:'<span></span>',iconSize:[26,32],iconAnchor:[13,30]})}).addTo(m);
   pin.current.on('dragend',()=>{const q=pin.current!.getLatLng();pick.current(round6(q.lat),round6(q.lng))})}
  else pin.current.setLatLng(p)};
 useEffect(()=>{const m=map.current;if(!m||!ready)return;
  const g=L.layerGroup(others.filter(located).map(s=>L.circleMarker([s.latitude!,s.longitude!],{radius:5,color:'#5f7287',weight:1.5,fillColor:'#cbd6e2',fillOpacity:.9,interactive:true})
   .bindTooltip(esc(s.code),{direction:'top'}))).addTo(m);
  const click=(e:L.LeafletMouseEvent)=>{const w=e.latlng.wrap();pinTo(w);pick.current(round6(w.lat),round6(w.lng))};
  m.on('click',click);
  if(!centred.current&&lat==null){const pts=others.filter(located).map(s=>[s.latitude!,s.longitude!] as L.LatLngTuple);if(pts.length)m.fitBounds(pts,{padding:[40,40],maxZoom:11})}
  return()=>{m.off('click',click);g.remove()}},[ready,others]);
 useEffect(()=>{const m=map.current;if(!m||!ready)return;  // typed coordinates move the pin too
  if(lat==null||lon==null||!isFinite(lat)||!isFinite(lon)||Math.abs(lat)>90||Math.abs(lon)>180){pin.current?.remove();pin.current=null;return}
  pinTo([lat,lon]);
  if(!centred.current){m.setView([lat,lon],15);centred.current=true}else if(!m.getBounds().contains([lat,lon]))m.panTo([lat,lon])},[lat,lon,ready]);
 function here(){if(!navigator.geolocation){setLocating('Location is not available in this browser');return}setLocating('Locating…');
  navigator.geolocation.getCurrentPosition(p=>{setLocating('');const a=round6(p.coords.latitude),b=round6(p.coords.longitude);centred.current=true;map.current?.setView([a,b],17);pick.current(a,b)},
   e=>setLocating(e.code===1?'Location permission denied':'Could not get your location'),{enableHighAccuracy:true,timeout:15000})}
 async function runSearch(){const q=query.trim();if(q.length<2||searching)return;setFound('');
  const c=parseCoords(q);if(c){setResults(null);centred.current=true;map.current?.setView(c,17);pinTo(c);pick.current(round6(c[0]),round6(c[1]));setFound('Placed at the pasted coordinates');return}
  setSearching(true);try{setResults(await searchPlaces(q,map.current?.getBounds()))}catch{setResults(null);setFound('Place search is unavailable right now. Pan and zoom the map instead.')}finally{setSearching(false)}}
 function goTo(p:Place){const m=map.current;if(!m)return;setResults(null);centred.current=true;
  const lat=+p.lat,lon=+p.lon,bb=p.boundingbox?.map(Number);
  if(bb&&bb.every(isFinite)&&(bb[1]-bb[0]>.0005||bb[3]-bb[2]>.0005))m.fitBounds([[bb[0],bb[2]],[bb[1],bb[3]]],{maxZoom:17});else m.setView([lat,lon],17);
  spot.current?.remove();spot.current=L.circleMarker([lat,lon],{radius:8,color:'#1769aa',weight:2,fillColor:'#1769aa',fillOpacity:.15,interactive:false}).addTo(m);  // where the place is, not yet the tower
  setFound(`Showing ${p.display_name.split(',').slice(0,3).join(',')}. Now click the tower's exact position.`)}
 return <div className="st-picker">
  <div className="st-place-search">
   <div className="search"><Search size={15}/><input value={query} onChange={e=>{setQuery(e.target.value);if(!e.target.value)setResults(null)}} placeholder="Search a place or paste coordinates (17.385, 78.4867)"
     onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();runSearch()}if(e.key==='Escape'&&results){e.preventDefault();e.stopPropagation();setResults(null)}}} aria-label="Search a place"/>
    {query&&<button type="button" className="st-icon-btn" onClick={()=>{setQuery('');setResults(null)}} aria-label="Clear"><X size={13}/></button>}</div>
   <button type="button" className="btn outline" onClick={runSearch} disabled={searching||query.trim().length<2}>{searching?<Loader2 size={14} className="st-spin"/>:<Search size={14}/>}Search</button>
   {results&&<div className="st-place-results" role="listbox">{results.length?results.map((p,i)=><button type="button" key={i} role="option" onClick={()=>goTo(p)}><MapPin size={13}/>
     <span><b>{p.display_name.split(',')[0]}</b><small>{p.display_name.split(',').slice(1).join(',').trim()}</small></span>{p.type&&<em>{p.type.replaceAll('_',' ')}</em>}</button>)
     :<div className="st-place-empty">No places found. Try a town, area or landmark name.</div>}</div>}
  </div>
  <div className="st-map-wrap"><div ref={el} className="st-map st-map-small"/>
   <button type="button" className="st-map-btn" title="Use my current location" onClick={here}><Crosshair size={15}/></button></div>
  <small className="muted">{locating||found||'Click the map to place the tower (it zooms in on that spot), then drag the pin to the exact position.'}</small>
  <small className="st-osm-note">Search by OpenStreetMap Nominatim</small>
 </div>;
}
