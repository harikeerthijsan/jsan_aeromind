import {useEffect,useRef} from 'react';
import {useFrame,useThree} from '@react-three/fiber';
import * as THREE from 'three';

/** Keeps an ordinary page element (rendered next to the <Canvas>, not inside it) glued to a 3D point.
 * Replaces drei <Html>, which mounts every label in its own React root: when the whole canvas was swapped out
 * (Digital Twin: demo tower -> reconstruction) its cleanup ran mid-render and failed with "removeChild ... not a child".
 * The element must sit in the same positioned container as the canvas, at top:0/left:0 (class "floating-label"). */
export function LabelAnchor({target,position}:{target:React.RefObject<HTMLElement|null>;position:[number,number,number]|null}){
 const {camera,size}=useThree();const v=useRef(new THREE.Vector3());
 useEffect(()=>()=>{if(target.current)target.current.style.visibility='hidden'},[target]);
 useFrame(()=>{const el=target.current;if(!el)return;
  if(!position){el.style.visibility='hidden';return}
  v.current.set(...position).project(camera);const inView=v.current.z>-1&&v.current.z<1;
  el.style.visibility=inView?'visible':'hidden';
  el.style.transform=`translate(${(v.current.x+1)/2*size.width}px,${(1-v.current.y)/2*size.height}px) translate(-50%,calc(-100% - 8px))`});
 return null;
}
