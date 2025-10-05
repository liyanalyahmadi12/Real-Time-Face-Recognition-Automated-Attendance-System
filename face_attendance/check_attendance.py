from db import SessionLocal, Attendance, User
s = SessionLocal()
u = s.query(User).first()
rows = s.query(Attendance).filter_by(user_id=u.user_id).all() if u else []
print("User:", (u.name if u else None), "| attendance rows:", len(rows))
for r in rows[-3:]:
    print(r.date, r.checkin1_time, r.checkin2_time, r.checkin3_time, r.checkin4_time)
s.close()
