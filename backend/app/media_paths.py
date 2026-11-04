"""Stable, human-readable storage keys for inspection media.

Local files live below UPLOAD_DIR. The storage adapter prepends ``uploads/`` to the same relative key in S3, so local
and production bucket layouts always match.
"""
from __future__ import annotations

import re, uuid
from pathlib import Path, PurePosixPath

_UNSAFE = re.compile(r'[^A-Za-z0-9._-]+')

def _segment(value:str,fallback:str,limit:int=80)->str:
    value=_UNSAFE.sub('-',value.strip()).strip('._-')
    return (value or fallback)[:limit]

def inspection_root(site_code:str,inspection_id:int)->PurePosixPath:
    site=_segment(site_code.upper(),f'SITE-{inspection_id}')
    return PurePosixPath('sites',site,'inspections',f'INS-{inspection_id:06d}')

def upload_key(site_code:str,inspection_id:int,filename:str,content_type:str,suffix:str,token:str|None=None)->str:
    """Key for a directly uploaded image/video. The suffix comes from the validated MIME type, never the client."""
    kind='images' if content_type.startswith('image/') else 'videos'
    stem=_segment(Path(filename or 'upload').stem,'upload',70)
    unique=_segment(token or uuid.uuid4().hex[:12],'media',32)
    return (inspection_root(site_code,inspection_id)/kind/f'{unique}_{stem}{suffix.lower()}').as_posix()

def frame_dir(parent_storage_key:str,parent_media_id:int)->PurePosixPath:
    """Put extracted frames beside their source inspection; retain the legacy fallback for old flat video keys."""
    parent=PurePosixPath(parent_storage_key)
    parts=parent.parts
    if len(parts)>=6 and parts[0]=='sites' and parts[2]=='inspections' and parts[4] in ('images','videos'):
        return parent.parent.parent/'frames'/f'MEDIA-{parent_media_id:06d}'
    return PurePosixPath('frames',str(parent_media_id))

def migrated_key(site_code:str,inspection_id:int,media_id:int,filename:str,content_type:str,current_key:str,
                 parent_key:str|None=None,parent_media_id:int|None=None,frame_index:int|None=None)->str:
    """Deterministic destination used when reorganising existing media."""
    suffix=Path(current_key).suffix.lower() or ('.jpg' if content_type.startswith('image/') else '.mp4')
    if parent_key is not None and parent_media_id is not None:
        index=frame_index if frame_index is not None else media_id
        return (frame_dir(parent_key,parent_media_id)/f'{index:06d}{suffix}').as_posix()
    return upload_key(site_code,inspection_id,filename,content_type,suffix,f'MEDIA-{media_id:06d}')

def is_structured(key:str)->bool:
    p=PurePosixPath(key).parts
    return len(p)>=6 and p[0]=='sites' and p[2]=='inspections' and p[4] in ('images','videos','frames')
