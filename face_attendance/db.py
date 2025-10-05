# db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Time, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool
from datetime import date
from config import DATABASE_URL

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    # store FaceNet/Facenet512 embedding JSON string
    face_embedding = Column(Text, nullable=False)

    attendance = relationship("Attendance", back_populates="user", cascade="all, delete-orphan")

class Attendance(Base):
    __tablename__ = "attendance"
    record_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)  # ISO date string "YYYY-MM-DD"
    quarter_id = Column(Integer, nullable=False, index=True)

    checkin1_time = Column(String(8), nullable=True)
    checkin2_time = Column(String(8), nullable=True)
    checkin3_time = Column(String(8), nullable=True)
    checkin4_time = Column(String(8), nullable=True)

    user = relationship("User", back_populates="attendance")

def _engine_kwargs(url: str):
    if url.startswith("sqlite"):
        return dict(connect_args={"check_same_thread": False}, future=True)
    # MySQL sensible defaults
    return dict(
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )

engine = create_engine(DATABASE_URL, **_engine_kwargs(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def current_quarter_id(d: date) -> int:
    q = (d.month - 1) // 3 + 1
    return d.year * 10 + q  # e.g., 20254 for Q4 2025
