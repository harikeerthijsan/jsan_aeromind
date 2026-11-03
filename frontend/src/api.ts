export const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '');  // production build: same origin as the page
export type Site={id:number;code:string;name:string;region:string;tower_type:string;height_m:number|null;status:string};
export type Inspection={id:number;site_id:number;label:string;status:string;created_at:string};
export type Asset={id:number;site_id:number;code:string;category:string;condition:string;identity_status:string;localization_status:string;x:number|null;y:number|null;z:number|null;description:string};
export type Defect={id:number;asset_id:number;defect_type:string;proposed_grade:string;review_status:string;notes:string};
export type Job={id:number;inspection_id:number;status:string;stage:string;progress:number;detail:string;kind?:string;media_ids?:number[]|null};
export type Media={id:number;inspection_id:number;filename:string;content_type:string;size_bytes:number;url:string;parent_media_id:number|null;frame_index:number|null;frame_time:number|null};
export type Detection={id:number;job_id:number;media_id:number;asset_id:number|null;frame_index:number;class_id:number;class_name:string;confidence:number;x1:number;y1:number;x2:number;y2:number;image_width:number;image_height:number;model_name:string;review_status:string};
export type Review={id:number;inspection_id:number;media_id:number;frame_index:number;status:string;workflow:string;job_id:number|null;box_count:number;notes:string;reviewed_at:string};
export type Annotation={id:number;review_id:number;media_id:number;detection_id:number|null;source:string;class_id:number;class_name:string;confidence:number|null;x1:number;y1:number;x2:number;y2:number;image_width:number;image_height:number;manufacturer:string|null;product_model:string|null;vendor_source:string|null;vendor_confidence:number|null;vendor_evidence:string|null;vendor_basis:string|null;condition_grade:string|null;condition_score:number|null;condition_defects:ConditionDefect[]|null;condition_source:string|null;condition_confidence:number|null;condition_verified:string|null;condition_summary:string|null};
export type MediaAnnotations={review:Review|null;annotations:Annotation[]};
export type ClassInfo={id:number;name:string};
export type VendorInfo={classes:string[];manufacturers:Record<string,string[]>;ai:{status:string;model:string|null;url:string};reference_dir:string;folder_references:Record<string,number>;reviewer_references:Record<string,number>;library_manufacturers:Record<string,string[]>};
export type VendorRef={manufacturer:string;model:string;view:string;status:string;page:string;notes:string;match:boolean;url:string};
export type ConditionDefect={type:string;severity:string;extent_percent:number;location:string;evidence:string;source?:string;confirmed?:boolean};
export type ConditionResult={grade:string;health_score:number|null;defects:ConditionDefect[];rust_percent:number;image_quality:string;confidence:number;summary:string;recommended_action:string;verified?:string;skipped?:boolean;model:string|null;seconds:number};
export type VendorCandidate={manufacturer:string;similarity:number;brands:string[];matching_features:string[]};
export type VendorSuggestion={candidates:VendorCandidate[];ambiguous:boolean;manufacturer:string;product_model:string|null;confidence:number;basis:string;evidence:string;features:Record<string,string>;reference_vendors:string[];visible_text:string;model:string;seconds:number};
export type DetectorInfo={status:string;model:string|null;imgsz:number;conf:number;device:string;vid_stride:number;loaded:boolean;classes?:Record<string,string>};
export type RegistrationCandidate={id:number;model_id:number;inspection_id:number;site_id:number;class_name:string;x:number;y:number;z:number;confidence:number;status:string;evidence_count:number;media_count:number;point_count:number;spread:number;detection_ids:number[];created_at:string;size_w?:number|null;size_h?:number|null;size_d?:number|null;yaw?:number|null;box_method?:string|null;box_points?:number|null;fit_iou?:number|null};
export async function request<T>(path:string, options:RequestInit={}):Promise<T>{const r=await fetch(API+path,options); if(!r.ok)throw Error((await r.text()).slice(0,250));return r.json();}
export const json=(method:string,body:unknown):RequestInit=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
