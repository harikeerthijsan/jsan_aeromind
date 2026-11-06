import type {CSSProperties} from 'react';

/** The 3D navigation bar, the same in every 3D view (full LiDAR and the regular model view), as in the PLA QC app:
 * view presets, orbit / pan, fit, focus and close-up on the selected box, point size, and a hint for the mouse. */
export type ViewMode='3d'|'top'|'front'|'side';
const btn=(on:boolean,off=false):CSSProperties=>({padding:'3px 8px',borderRadius:6,border:'1px solid '+(on?'#5fa8e6':'transparent'),background:on?'#1d4a73':'transparent',
 color:'inherit',fontSize:11,cursor:off?'default':'pointer',opacity:off?.45:1});
const bar:CSSProperties={display:'flex',flexWrap:'wrap',justifyContent:'flex-end',alignItems:'center',gap:4,padding:'4px 6px',borderRadius:8,background:'rgba(15,31,49,.88)',
 border:'1px solid #2d5878',color:'#cfe0ee',fontSize:11};
const sep=<span style={{width:1,height:16,background:'#2d5878',margin:'0 2px'}}/>;
export const navHint=(pan:boolean)=>(pan?'Left-drag: pan · Right-drag: rotate · Wheel: zoom to cursor':'Left-drag: rotate · Right/middle-drag: pan · Wheel: zoom')
 +' · Double-click: zoom in on a spot (Shift: out) · Double-click a box: focus it';

export default function ViewToolbar({mode,onView,pan,onPan,onFit,canFocus,onFocus,isolated,onIsolate,size,onSize,orthographic=false,hint=true}:{mode:ViewMode;onView:(m:ViewMode)=>void;
 pan:boolean;onPan:()=>void;onFit:()=>void;canFocus:boolean;onFocus:()=>void;isolated:boolean;onIsolate:()=>void;size?:number;onSize?:(v:number)=>void;orthographic?:boolean;hint?:boolean}){
 const stop={onPointerDown:(e:React.PointerEvent)=>e.stopPropagation(),onPointerUp:(e:React.PointerEvent)=>e.stopPropagation(),onDoubleClick:(e:React.MouseEvent)=>e.stopPropagation()};
 return <div {...stop} style={{position:'absolute',right:12,top:12,zIndex:6,maxWidth:'calc(100% - 200px)',display:'flex',flexDirection:'column',alignItems:'flex-end',gap:4}}>
  <div style={bar}>
   {(['3d','top','front','side'] as ViewMode[]).map(v=><button key={v} type="button" onClick={()=>onView(v)} style={btn(mode===v)}
    title={v==='3d'?'Perspective 3D view':`${v[0].toUpperCase()+v.slice(1)} view${orthographic?' (orthographic)':''}`}>{v==='3d'?'3D':v[0].toUpperCase()+v.slice(1)}</button>)}
   {sep}
   <button type="button" title="Orbit: left-drag rotates around the focus. Pan: left-drag moves, right-drag rotates" onClick={onPan} style={btn(false)}>{pan?'Pan':'Orbit'}</button>
   <button type="button" title="Show the whole model" onClick={onFit} style={btn(false)}>Fit</button>
   <button type="button" disabled={!canFocus} title="Zoom onto the selected box" onClick={onFocus} style={btn(false,!canFocus)}>Focus</button>
   <button type="button" disabled={!canFocus&&!isolated} title="Close-up: show only the selected box and 1.5 m around it" onClick={onIsolate} style={btn(isolated,!canFocus&&!isolated)}>{isolated?'Show all':'Isolate'}</button>
   {onSize&&size!=null&&<>{sep}<label title="Point size" style={{display:'flex',alignItems:'center',gap:4}}>Size<input type="range" min={.3} max={3} step={.1} value={size} onChange={e=>onSize(+e.target.value)} style={{width:70}}/></label></>}
  </div>
  {hint&&<div style={{padding:'2px 8px',borderRadius:6,background:'rgba(15,31,49,.72)',color:'#9fb6c8',fontSize:10.5,textAlign:'right',pointerEvents:'none'}}>{navHint(pan)}</div>}
 </div>;
}
