import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

def normalise_url(url: str) -> str:
    """Railway/Heroku style postgres:// URLs -> SQLAlchemy with the psycopg (v3) driver."""
    for old in ('postgres://', 'postgresql://'):
        if url.startswith(old): return 'postgresql+psycopg://' + url[len(old):]
    return url

DB_URL = normalise_url(os.getenv('DATABASE_URL', 'sqlite:///./tower_vision.db'))
engine = create_engine(DB_URL, connect_args={'check_same_thread': False} if DB_URL.startswith('sqlite') else {}, pool_pre_ping=True,
                       **({} if DB_URL.startswith('sqlite') else {'pool_size': 10, 'max_overflow': 10, 'pool_recycle': 1800}))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
class Base(DeclarativeBase): pass
