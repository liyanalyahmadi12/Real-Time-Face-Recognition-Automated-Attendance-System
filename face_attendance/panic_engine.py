# panic_engine.py
import os, cv2, time, json, argparse, datetime as dt
import numpy as np
from deepface import DeepFace
from sqlalchemy.exc import IntegrityError

from db import SessionLocal, User, Attendance, init_db, current_quarter_id

def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def cosine_distance(a, b):
    return float(1.0 - float(np.dot(a, b)))

def open_cam(index: int):
    backends = []
    if hasattr(cv2, "CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):  backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            print(f"[camera] using index {index} backend {be}")
            return cap
        cap.release()
    raise RuntimeError("Cannot open camera")

def main():
    ap = argparse.ArgumentParser("panic_engine")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dist", type=float, default=float(os.getenv("DIST_THRESHOLD", "0.75")))
    ap.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "Facenet"))  # faster than Facenet512
    args = ap.parse_args()

    init_db()
    sess = SessionLocal()
    users = sess.query(User).all()
    if not users:
        print("No users enrolled. Run enroll_webcam.py first.")
        return

    # Load embeddings matrix
    names, uids, embs = [], [], []
    for u in users:
        try:
            e = np.asarray(json.loads(u.face_embedding), dtype=np.float32)
            embs.append(l2norm(e)); names.append(u.name); uids.append(u.user_id)
        except Exception:
            pass
    if not embs:
        print("No valid embeddings in DB."); return
    EMB = np.stack(embs, axis=0).astype(np.float32)

    print(f"Loaded users: {names}")
    print(f"Model: {args.model}, threshold: {args.dist}")
    # Warm up the model so DeepFace caches it (optional)
    _ = DeepFace.build_model(args.model)

    # Haar detector (fast & simple)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = open_cam(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_log = {}   # (uid)->last_time
    cooldown = 8    # seconds
    fps_t0, fps_n = time.time(), 0
    print("Press 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0
            else:
                fps = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60)
            )

            msg = "No face"
            color = (0,255,255)
            best_txt = ""

            if len(faces):
                # take largest
                x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
                pad = int(max(w,h)*0.20)
                x1,y1 = max(0,x-pad), max(0,y-pad)
                x2,y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
                face = frame[y1:y2, x1:x2]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

                # embed with skip (no second detection)
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                try:
                    rep = DeepFace.represent(
                        img_path=rgb,
                        model_name=args.model,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=False,
                    )
                except TypeError as e:
                    # very old DeepFace may not support align=; try without it
                    rep = DeepFace.represent(
                        img_path=rgb,
                        model_name=args.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )

                if rep:
                    q = l2norm(np.asarray(rep[0]["embedding"], dtype=np.float32))
                    # vectorized distance to all
                    dists = 1.0 - np.dot(EMB, q)
                    i = int(np.argmin(dists)); d = float(dists[i])
                    best_name, best_uid = names[i], uids[i]
                    best_txt = f"{best_name} d={d:.3f}"
                    if d <= args.dist:
                        # cooldown
                        now = time.time()
                        if now - last_log.get(best_uid, 0) >= cooldown:
                            # log gate 1 now
                            nowdt = dt.datetime.now()
                            date_iso = nowdt.date().isoformat()
                            rec = (sess.query(Attendance)
                                   .filter_by(user_id=best_uid, date=date_iso)
                                   .one_or_none())
                            if not rec:
                                rec = Attendance(
                                    user_id=best_uid, date=date_iso,
                                    quarter_id=current_quarter_id(nowdt.date())
                                )
                                sess.add(rec); sess.flush()
                            rec.checkin1_time = nowdt.strftime("%H:%M:%S")
                            try:
                                sess.commit()
                                last_log[best_uid] = now
                                msg = f"✓ {best_name} logged Gate 1 ({rec.checkin1_time})"
                                color = (0,255,0)
                            except IntegrityError:
                                sess.rollback()
                                msg = f"{best_name} already logged today"
                                color = (255,255,0)
                        else:
                            msg = f"{best_name} cooldown {(cooldown - int(now - last_log[best_uid]))}s"
                            color = (255,165,0)
                    else:
                        msg = f"Analyzing… {best_txt}"
                        color = (0,255,255)
                else:
                    msg = "Face crop but embed failed"
                    color = (0,0,255)

            # overlay
            cv2.rectangle(frame,(0,0),(frame.shape[1],50),(0,0,0),-1)
            cv2.putText(frame, msg, (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if fps is not None:
                cv2.putText(frame, f"FPS {fps:.1f}", (10, frame.shape[0]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if best_txt:
                cv2.putText(frame, best_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Panic Engine (q=quit)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sess.close()

if __name__ == "__main__":
    main()
