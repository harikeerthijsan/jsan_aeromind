import {useEffect,useRef,useState} from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {Crosshair,Maximize2} from 'lucide-react';

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

const pinIcon=(s:MapSite,on:boolean)=>L.divIcon({className:'st-pin'+(on?' on':''),iconSize:[18,18],iconAnchor:[9,9],
 html:`<i class="${esc(s.status)}"></i><b>${esc(s.code)}</b>`});

/** Every tower with coordinates; clicking one calls onSelect (the page opens its detail panel). */
export function SitesMap({sites,selectedId,onSelect}:{sites:MapSite[];selectedId:number|null;onSelect:(id:number)=>void}){
 const {el,map,ready}=useMap(),layer=useRef<L.LayerGroup|null>(null),fitted=useRef('');
 const fit=()=>{const m=map.current,pts=sites.filter(located).map(s=>[s.latitude!,s.longitude!] as L.LatLngTuple);if(!m)return;
  if(pts.length>1)m.fitBounds(pts,{padding:[50,50],maxZoom:14});else if(pts.length===1)m.setView(pts[0],13);else m.setView(HOME,HOME_ZOOM)};
 useEffect(()=>{const m=map.current;if(!m||!ready)return;
  layer.current?.remove();const g=L.layerGroup().addTo(m);layer.current=g;
  for(const s of sites.filter(located))
   L.marker([s.latitude!,s.longitude!],{icon:pinIcon(s,s.id===selectedId),riseOnHover:true,zIndexOffset:s.id===selectedId?1000:0,keyboard:true,title:`${s.code} · ${s.name}`})
    .bindTooltip(`<b>${esc(s.code)}</b> · ${esc(s.name)}`,{direction:'top',offset:[0,-10]}).on('click',()=>onSelect(s.id)).addTo(g);
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

/** Click (or drag the pin) to set a tower's coordinates. Other towers are shown faintly for orientation. */
export function LocationPicker({lat,lon,onPick,others=[]}:{lat:number|null;lon:number|null;onPick:(lat:number,lon:number)=>void;others?:MapSite[]}){
 const {el,map,ready}=useMap(),pin=useRef<L.Marker|null>(null),pick=useRef(onPick),[locating,setLocating]=useState(''),centred=useRef(false);
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
 return <div className="st-picker">
  <div className="st-map-wrap"><div ref={el} className="st-map st-map-small"/>
   <button type="button" className="st-map-btn" title="Use my current location" onClick={here}><Crosshair size={15}/></button></div>
  <small className="muted">{locating||'Click the map to place the tower (it zooms in on that spot), then drag the pin to the exact position.'}</small>
 </div>;
}
