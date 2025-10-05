# db_extras.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from db import Base, engine  # uses your existing Base and engine

class AttendanceNote(Base):
    __tablename__ = "attendance_notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, ForeignKey("attendance.record_id"), nullable=False)
    gate = Column(String(20), nullable=False)  # "1","2","3","4"
    note = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# one-time create (idempotent)
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✓ attendance_notes table ready")
