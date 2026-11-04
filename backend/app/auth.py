"""Database-backed authentication and role-based access control for AeroMind."""
from __future__ import annotations

import base64, hashlib, hmac, os, secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import AuthSession, User, now

ROLES=('viewer','inspector','manager','admin')
ROLE_LEVEL={role:index for index,role in enumerate(ROLES)}
ITERATIONS=310_000
DEMO_PASSWORD=os.getenv('AUTH_DEMO_PASSWORD','AeroMind@123')
DEMO_USERS=(
    ('admin@aeromind.local','AeroMind Admin','admin'),
    ('manager@aeromind.local','Operations Manager','manager'),
    ('inspector@aeromind.local','Tower Inspector','inspector'),
    ('viewer@aeromind.local','Client Viewer','viewer'),
)

def _b64(data:bytes)->str:return base64.urlsafe_b64encode(data).decode().rstrip('=')
def _unb64(value:str)->bytes:return base64.urlsafe_b64decode(value+'='*(-len(value)%4))
def hash_password(password:str)->str:
    salt=secrets.token_bytes(18);digest=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,ITERATIONS)
    return f'pbkdf2_sha256${ITERATIONS}${_b64(salt)}${_b64(digest)}'
def verify_password(password:str,encoded:str)->bool:
    try:
        algorithm,rounds,salt,digest=encoded.split('$',3)
        if algorithm!='pbkdf2_sha256':return False
        actual=hashlib.pbkdf2_hmac('sha256',password.encode(),_unb64(salt),int(rounds))
        return hmac.compare_digest(actual,_unb64(digest))
    except (ValueError,TypeError):return False
def public_user(user:User)->dict:
    return {'id':user.id,'email':user.email,'name':user.name,'role':user.role,'active':user.active,'created_at':user.created_at,'last_login':user.last_login}
def seed_users(db:Session):
    if db.scalar(select(func.count()).select_from(User)):return
    if os.getenv('ENABLE_DEMO_SEED','true').lower()=='true':
        for email,name,role in DEMO_USERS:db.add(User(email=email,name=name,role=role,password_hash=hash_password(DEMO_PASSWORD),active=True))
    else:
        email=os.getenv('AUTH_ADMIN_EMAIL','admin@aeromind.local').strip().lower();password=os.getenv('AUTH_ADMIN_PASSWORD','')
        if not password:raise RuntimeError('AUTH_ADMIN_PASSWORD is required when demo seed is disabled')
        db.add(User(email=email,name=os.getenv('AUTH_ADMIN_NAME','AeroMind Admin'),role='admin',password_hash=hash_password(password),active=True))
    db.commit()
def _aware(value:datetime)->datetime:return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
COOKIE='aeromind_session'
def token_from(request:Request)->str:
    """The session token: the Authorization header, or for reads (GET/HEAD) the session cookie. Images, 3D model files
    and download links are fetched by the browser itself and cannot carry the header; changes still need the header,
    so a cross-site page cannot make a logged-in browser change anything with its cookie (CSRF)."""
    scheme,_,token=request.headers.get('authorization','').partition(' ')
    if scheme.lower()=='bearer' and token.strip():return token.strip()
    return request.cookies.get(COOKIE,'') if request.method in ('GET','HEAD') else ''
def set_cookie(response:Response,request:Request,token:str,expires:datetime):
    https=request.url.scheme=='https' or request.headers.get('x-forwarded-proto','')=='https'
    response.set_cookie(COOKIE,token,max_age=max(60,int((_aware(expires)-now()).total_seconds())),httponly=True,secure=https,samesite='lax',path='/')
def authenticate(db:Session,token:str)->User|None:
    if not token:return None
    row=db.scalar(select(AuthSession).where(AuthSession.token_hash==hashlib.sha256(token.encode()).hexdigest(),AuthSession.revoked_at.is_(None)))
    if not row or _aware(row.expires_at)<=now():return None
    user=db.get(User,row.user_id)
    return user if user and user.active else None
def current_user(request:Request)->User:
    user=getattr(request.state,'auth_user',None)
    if not user:raise HTTPException(401,'Authentication required',headers={'WWW-Authenticate':'Bearer'})
    return user
def require_admin(request:Request)->User:
    user=current_user(request)
    if user.role!='admin':raise HTTPException(403,'Administrator role required')
    return user
def minimum_role(path:str,method:str)->str:
    if path.startswith('/api/v1/auth/'):return 'viewer'
    if method in ('GET','HEAD'):return 'viewer'
    if path.startswith('/api/v1/users'):return 'admin'
    if path=='/api/v1/inspections' and method=='POST':return 'manager'
    if path.startswith('/api/v1/sites'):return 'manager'
    if method=='DELETE':return 'manager'
    return 'inspector'

class LoginIn(BaseModel):
    email:str=Field(min_length=3,max_length=255)
    password:str=Field(min_length=1,max_length=256)
    remember:bool=False
class UserCreateIn(BaseModel):
    email:str=Field(min_length=3,max_length=255)
    name:str=Field(min_length=2,max_length=160)
    role:str='viewer'
    password:str=Field(min_length=10,max_length=256)
class UserUpdateIn(BaseModel):
    name:str|None=Field(None,min_length=2,max_length=160)
    role:str|None=None
    active:bool|None=None
    password:str|None=Field(None,min_length=10,max_length=256)

router=APIRouter(prefix='/api/v1',tags=['authentication'])

@router.get('/auth/config')
def auth_config():
    demo=os.getenv('ENABLE_DEMO_SEED','true').lower()=='true' and os.getenv('AUTH_SHOW_DEMO_CREDENTIALS','true').lower()=='true'
    return {'demo':demo,'accounts':[{'email':e,'name':n,'role':r} for e,n,r in DEMO_USERS] if demo else [],'demo_password':DEMO_PASSWORD if demo else None}

@router.post('/auth/login')
def login(data:LoginIn,request:Request,response:Response):
    with SessionLocal() as db:
        email=data.email.strip().lower();user=db.scalar(select(User).where(func.lower(User.email)==email))
        if not user or not user.active or not verify_password(data.password,user.password_hash):raise HTTPException(401,'Invalid email or password')
        raw=secrets.token_urlsafe(40);expires=now()+timedelta(days=7 if data.remember else 1)
        db.execute(delete(AuthSession).where(AuthSession.expires_at<now()))
        db.add(AuthSession(user_id=user.id,token_hash=hashlib.sha256(raw.encode()).hexdigest(),expires_at=expires));user.last_login=now();db.commit();db.refresh(user)
        set_cookie(response,request,raw,expires)
        return {'token':raw,'expires_at':expires,'user':public_user(user)}

@router.get('/auth/me')
def me(request:Request,response:Response):
    """The signed-in user. Also (re)sets the session cookie for sessions that began before it existed."""
    user=current_user(request);token=token_from(request)
    with SessionLocal() as db:
        row=db.scalar(select(AuthSession).where(AuthSession.token_hash==hashlib.sha256(token.encode()).hexdigest())) if token else None
    if row:set_cookie(response,request,token,row.expires_at)
    return public_user(user)

@router.post('/auth/logout',status_code=204)
def logout(request:Request,response:Response):
    token=token_from(request);response.delete_cookie(COOKIE,path='/')
    with SessionLocal() as db:
        row=db.scalar(select(AuthSession).where(AuthSession.token_hash==hashlib.sha256(token.encode()).hexdigest())) if token else None
        if row:row.revoked_at=now();db.commit()

@router.get('/users')
def users(request:Request):
    require_admin(request)
    with SessionLocal() as db:return [public_user(u) for u in db.scalars(select(User).order_by(User.name)).all()]

@router.post('/users',status_code=201)
def create_user(data:UserCreateIn,request:Request):
    require_admin(request);email=data.email.strip().lower();name=data.name.strip()
    if '@' not in email:raise HTTPException(422,'Enter a valid email address')
    if data.role not in ROLES:raise HTTPException(422,'Invalid role')
    with SessionLocal() as db:
        if db.scalar(select(User).where(func.lower(User.email)==email)):raise HTTPException(409,'A user with this email already exists')
        user=User(email=email,name=name,role=data.role,password_hash=hash_password(data.password),active=True);db.add(user);db.commit();db.refresh(user);return public_user(user)

@router.patch('/users/{user_id}')
def update_user(user_id:int,data:UserUpdateIn,request:Request):
    actor=require_admin(request)
    with SessionLocal() as db:
        user=db.get(User,user_id)
        if not user:raise HTTPException(404,'User not found')
        if data.role is not None and data.role not in ROLES:raise HTTPException(422,'Invalid role')
        removing_admin=user.role=='admin' and ((data.role is not None and data.role!='admin') or data.active is False)
        if removing_admin and db.scalar(select(func.count()).select_from(User).where(User.role=='admin',User.active.is_(True)))<=1:raise HTTPException(409,'At least one active administrator is required')
        if user.id==actor.id and data.active is False:raise HTTPException(409,'You cannot deactivate your own account')
        if data.name is not None:user.name=data.name.strip()
        if data.role is not None:user.role=data.role
        if data.active is not None:user.active=data.active
        if data.password is not None:
            user.password_hash=hash_password(data.password);db.execute(delete(AuthSession).where(AuthSession.user_id==user.id))
        db.commit();db.refresh(user);return public_user(user)
