# attendance_logic.py
import datetime as dt
from typing import Optional
from db import Attendance

# Gate windows (24h)
GATES = {
    1: {"label": "Morning Start", "window": ("07:00", "09:00"), "expected": "08:00"},
    2: {"label": "Lunch Break Out", "window": ("12:00", "13:30"), "expected": "12:30"},
    3: {"label": "Lunch Break In",  "window": ("13:00", "14:30"), "expected": "14:00"},
    4: {"label": "Evening End",     "window": ("17:00", "19:30"), "expected": "17:00"},
}

def _time_in_window(now: dt.time, start: str, end: str) -> bool:
    s = dt.datetime.strptime(start, "%H:%M").time()
    e = dt.datetime.strptime(end, "%H:%M").time()
    return s <= now <= e if s <= e else (now >= s or now <= e)

def gate_for_now(now: Optional[dt.datetime] = None) -> Optional[int]:
    if now is None:
        now = dt.datetime.now()
    t = now.time()
    # order: 1 → 2 → 3 → 4
    for gid in (1, 2, 3, 4):
        st, en = GATES[gid]["window"]
        if _time_in_window(t, st, en):
            return gid
    return None

def write_gate_timestamp(row: Attendance, gate_id: int, hhmmss: str, overwrite: bool = True) -> bool:
    field = {1: "checkin1_time", 2: "checkin2_time", 3: "checkin3_time", 4: "checkin4_time"}[gate_id]
    cur = getattr(row, field)
    if cur and not overwrite:
        return False
    setattr(row, field, hhmmss)
    return True

def compute_lateness_minutes(gate_id: int, when: dt.datetime) -> int:
    """Positive minutes if late vs expected time; else 0."""
    exp = dt.datetime.strptime(GATES[gate_id]["expected"], "%H:%M").time()
    expected_dt = when.replace(hour=exp.hour, minute=exp.minute, second=0, microsecond=0)
    delta = (when - expected_dt).total_seconds() / 60.0
    return max(0, int(round(delta)))
