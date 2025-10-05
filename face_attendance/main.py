# main.py
import os
import sys
import argparse
import datetime as dt
from typing import Optional

from db import SessionLocal, Attendance, User

# -----------------------------
# Helpers
# -----------------------------
def _parse_date(s: Optional[str]) -> Optional[str]:
    """Accepts 'today', 'yesterday', or YYYY-MM-DD; returns ISO date string or None."""
    if not s:
        return None
    s = s.strip().lower()
    if s in ("today", "tod"):
        d = dt.date.today()
        return d.isoformat()
    if s in ("yesterday", "yday", "yd"):
        d = dt.date.today() - dt.timedelta(days=1)
        return d.isoformat()
    # try parse YYYY-MM-DD
    try:
        d = dt.date.fromisoformat(s)
        return d.isoformat()
    except Exception:
        raise SystemExit(f"Invalid date: {s}. Use 'today', 'yesterday', or YYYY-MM-DD.")

def _print_rows(rows):
    if not rows:
        print("No records.")
        return
    # pretty print
    print("-" * 86)
    print(f"{'Date':<12} {'User':<18} {'G1':<8} {'G2':<8} {'G3':<8} {'G4':<8} {'Quarter':<8}")
    print("-" * 86)
    for a, u in rows:
        g1 = a.checkin1_time or "-"
        g2 = a.checkin2_time or "-"
        g3 = a.checkin3_time or "-"
        g4 = a.checkin4_time or "-"
        q  = a.quarter_id or "-"
        print(f"{a.date:<12} {u.name:<18} {g1:<8} {g2:<8} {g3:<8} {g4:<8} {str(q):<8}")
    print("-" * 86)

# -----------------------------
# Commands
# -----------------------------
def cmd_engine(args):
    """
    Launch the live recognition engine window.
    You can pick your camera with --camera (e.g., --camera 1 for your 2nd camera),
    and optionally override the match threshold with --dist (e.g., --dist 0.62).
    """
    if args.camera is not None:
        os.environ["CAM_INDEX"] = str(args.camera)
    if args.dist is not None:
        os.environ["DIST_THRESHOLD"] = str(args.dist)

    # Optional: toggle YOLO by env var if you added it in live_engine.py
    if args.use_yolo:
        os.environ["USE_YOLO"] = "1"

    # Import here so any env vars above are in effect for the engine
    from live_engine import main as engine_main
    engine_main()

def cmd_check(args):
    """
    Show attendance rows for a date (default: today) or a range (--days N).
    You can filter by user name with --user.
    """
    sess = SessionLocal()
    try:
        q = (
            sess.query(Attendance, User)
            .join(User, Attendance.user_id == User.user_id)
        )

        # date filtering
        if args.days is not None:
            # last N days inclusive of today
            today = dt.date.today()
            start = today - dt.timedelta(days=int(args.days) - 1)
            # since dates are stored as ISO strings, string comparison works
            q = q.filter(Attendance.date >= start.isoformat(), Attendance.date <= today.isoformat())
        else:
            date_str = _parse_date(args.date) or dt.date.today().isoformat()
            q = q.filter(Attendance.date == date_str)

        # user filter (case-insensitive contains)
        if args.user:
            name_like = f"%{args.user}%"
            q = q.filter(User.name.ilike(name_like))

        rows = (
            q.order_by(Attendance.date.desc(), User.name.asc())
             .all()
        )
        _print_rows(rows)
    finally:
        sess.close()

def cmd_users(_args):
    """List all enrolled users."""
    sess = SessionLocal()
    try:
        users = [u.name for u in sess.query(User).order_by(User.name.asc()).all()]
        if not users:
            print("No users enrolled yet.")
        else:
            print("Users ->", users)
    finally:
        sess.close()

def cmd_dashboard(_args):
    """
    Launch the Streamlit dashboard if you have streamlit_dashboard.py in this folder.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "streamlit_dashboard.py")
    if not os.path.exists(script):
        print("streamlit_dashboard.py not found in this folder.")
        sys.exit(1)
    # Forward the current environment; open Streamlit in the browser
    try:
        subprocess.run(["streamlit", "run", script], check=True)
    except FileNotFoundError:
        print("Streamlit not found. Install it with: pip install streamlit")
        sys.exit(1)

# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Real-Time Face Recognition Attendance")
    sub = p.add_subparsers(dest="cmd", required=True)

    # engine
    p_eng = sub.add_parser("engine", help="Run the live recognition engine")
    p_eng.add_argument("--camera", type=int, default=None, help="Camera index (e.g., 1 for your 2nd camera)")
    p_eng.add_argument("--dist", type=float, default=None, help="Override distance threshold (e.g., 0.62)")
    p_eng.add_argument("--use-yolo", action="store_true", help="Use YOLO detector if enabled in live_engine.py")
    p_eng.set_defaults(func=cmd_engine)

    # check
    p_chk = sub.add_parser("check", help="Show attendance rows")
    p_chk.add_argument("--date", type=str, default="today", help="today|yesterday|YYYY-MM-DD (default: today)")
    p_chk.add_argument("--days", type=int, default=None, help="Show the last N days (overrides --date)")
    p_chk.add_argument("--user", type=str, default=None, help="Filter by user name (case-insensitive)")
    p_chk.set_defaults(func=cmd_check)

    # users
    p_usr = sub.add_parser("users", help="List enrolled users")
    p_usr.set_defaults(func=cmd_users)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Run the Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
