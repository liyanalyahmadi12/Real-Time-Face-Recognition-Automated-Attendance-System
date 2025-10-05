# live_engine.py — combined detector + DeepFace embed (skip re-detection)
import os
import cv2
import json
import time
import datetime as dt
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from db import init_db, SessionLocal, User, Attendance, current_quarter_id
from attendance_logic import gate_for_now, write_gate_timestamp
from recognition import FaceRecognitionEngine

# ------------------- Config (env / sane defaults) -------------------
MODEL_NAME     = os.getenv("MODEL_NAME", "Facenet")
DIST_THRESHOLD = float(os.getenv("DIST_THRESHOLD", "0.75"))
CAMERA_INDEX   = int(os.getenv("CAMERA_INDEX", "-1"))  # -1 = auto
DETECTION_SCALE = float(os.getenv("DETECTION_SCALE", "0.75"))  # 0.5..0.8 fast & solid
MIN_FACE_SIZE   = int(os.getenv("MIN_FACE_SIZE", "48"))        # in ORIGINAL frame pixels

VOTE_WINDOW   = int(os.getenv("VOTE_WINDOW", "3"))
VOTE_MIN_SAME = int(os.getenv("VOTE_MIN_SAME", "2"))
COOLDOWN_SEC  = int(os.getenv("COOLDOWN_SEC", "10"))

SHOW_DISTANCE = True
SHOW_BOXES    = True


#---------------------------
#-------------------------
# ------------------- Optional detectors (lazy import) -------------------
def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception:
        return None

def _try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except Exception:
        return None

def _try_import_retinaface():
    try:
        from retinaface import RetinaFace  # type: ignore
        return RetinaFace
    except Exception:
        return None

def _try_import_mtcnn():
    try:
        from mtcnn import MTCNN  # type: ignore
        return MTCNN
    except Exception:
        return None

# ------------------- Camera -------------------
def open_capture(preferred: Optional[int] = None) -> cv2.VideoCapture:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"): backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):  backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    indices = [preferred] if (preferred is not None and preferred >= 0) else [0, 1, 2, 3]
    for idx in indices:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[camera] using index {idx} backend {be}")
                    return cap
            cap.release()
    raise RuntimeError("Cannot open camera")

# ------------------- Combined detector -------------------
class CombinedDetector:
    """
    Tries multiple detectors in a fast-to-slower order:
      1) YOLO (if 'yolov8n-face.pt' exists and ultralytics installed)
      2) MediaPipe
      3) Haar cascades
      4) OpenCV DNN SSD (if prototxt/caffemodel present)
    Returns boxes in ORIGINAL image coordinates [(x,y,w,h), ...]
    """

    def __init__(self):
        self._init_yolo()
        self._init_mediapipe()
        self._init_haar()
        self._init_dnn()

    def _init_yolo(self):
        self.yolo = None
        YOLO = _try_import_ultralytics()
        if YOLO is not None:
            weights = os.getenv("YOLO_FACE_WEIGHTS", "yolov8n-face.pt")
            if os.path.exists(weights):
                try:
                    self.yolo = YOLO(weights)
                    print("[det] YOLO face model loaded")
                except Exception:
                    self.yolo = None

    def _init_mediapipe(self):
        self.mp = _try_import_mediapipe()
        self.mp_detector = None
        if self.mp:
            try:
                self.mp_detector = self.mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
                print("[det] MediaPipe loaded")
            except Exception:
                self.mp = None
                self.mp_detector = None

    def _init_haar(self):
        self.haar = []
        for name in ("haarcascade_frontalface_default.xml",
                     "haarcascade_frontalface_alt2.xml"):
            c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if not c.empty():
                self.haar.append(c)
        if self.haar:
            print("[det] Haar loaded")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _init_dnn(self):
        self.dnn = None
        proto = os.getenv("DNN_PROTO", "deploy.prototxt")
        caff = os.getenv("DNN_WEIGHTS", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(proto) and os.path.exists(caff):
            try:
                self.dnn = cv2.dnn.readNetFromCaffe(proto, caff)
                self.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[det] OpenCV DNN loaded")
            except Exception:
                self.dnn = None

    def detect(self, frame_bgr: np.ndarray, scale: float, min_size: int) -> List[Tuple[int,int,int,int]]:
        H, W = frame_bgr.shape[:2]
        boxes: List[Tuple[int,int,int,int]] = []

        # 1) YOLO
        if self.yolo is not None:
            try:
                r = self.yolo.predict(source=frame_bgr, verbose=False, imgsz=640)
                for det in r:
                    for b in det.boxes.xyxy.cpu().numpy().astype(int):
                        x1, y1, x2, y2 = b[:4]
                        w, h = x2 - x1, y2 - y1
                        if w >= min_size and h >= min_size:
                            boxes.append((x1, y1, w, h))
                if boxes:
                    return boxes
            except Exception:
                pass

        # 2) MediaPipe
        if self.mp_detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                res = self.mp_detector.process(rgb)
                if res and res.detections:
                    for d in res.detections:
                        b = d.location_data.relative_bounding_box
                        x1 = int(b.xmin * W); y1 = int(b.ymin * H)
                        w  = int(b.width * W); h = int(b.height * H)
                        if w >= min_size and h >= min_size:
                            boxes.append((max(0,x1), max(0,y1), min(W-x1,w), min(H-y1,h)))
                if boxes:
                    return boxes
            except Exception:
                pass

        # 3) Haar (downscale for speed)
        if self.haar:
            if scale != 1.0:
                small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small = frame_bgr
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            for det in self.haar:
                faces = det.detectMultiScale(
                    gray, scaleFactor=1.07, minNeighbors=3,
                    minSize=(int(min_size*scale), int(min_size*scale)),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in faces:
                    boxes.append((int(x/scale), int(y/scale), int(w/scale), int(h/scale)))
            if boxes:
                return _nms(boxes, 0.35)

        # 4) OpenCV DNN SSD
        if self.dnn is not None:
            blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.dnn.setInput(blob)
            detections = self.dnn.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.5:
                    x1 = int(detections[0, 0, i, 3] * W)
                    y1 = int(detections[0, 0, i, 4] * H)
                    x2 = int(detections[0, 0, i, 5] * W)
                    y2 = int(detections[0, 0, i, 6] * H)
                    w, h = x2 - x1, y2 - y1
                    if w >= min_size and h >= min_size:
                        boxes.append((x1, y1, w, h))
            if boxes:
                return boxes

        return boxes

def _nms(boxes: List[Tuple[int,int,int,int]], thr=0.35) -> List[Tuple[int,int,int,int]]:
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,0]+b[:,2], b[:,1]+b[:,3]
    area = b[:,2]*b[:,3]
    idxs = np.argsort(area)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        overlap = (w*h) / (area[idxs[1:]] + 1e-8)
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > thr)[0] + 1)))
    return [tuple(map(int, boxes[i])) for i in keep]

# ------------------- Helpers -------------------
def load_known_users(sess) -> List[Dict[str, Any]]:
    users = []
    for u in sess.query(User).all():
        vec = np.asarray(json.loads(u.face_embedding), dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        users.append({"user_id": u.user_id, "name": u.name, "embedding": vec})
    return users

# ------------------- Main loop -------------------
def main():
    init_db()

    # camera
    cam_idx = CAMERA_INDEX if CAMERA_INDEX >= 0 else None
    cap = open_capture(preferred=cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # db + known users
    sess = SessionLocal()
    known_users = load_known_users(sess)
    print(f"[db] users: {len(known_users)}")

    # engines
    detector = CombinedDetector()
    recog    = FaceRecognitionEngine(model_name=MODEL_NAME, dist_threshold=DIST_THRESHOLD)

    votes: deque = deque(maxlen=VOTE_WINDOW)
    last_logged: Dict[Tuple[int,int], float] = defaultdict(float)
    last_face_box: Optional[Tuple[int,int,int,int]] = None
    last_dist: Optional[float] = None

    # fps
    t0 = time.time(); n = 0; fps = 0.0

    print(f"[engine] model={MODEL_NAME} thr={DIST_THRESHOLD:.3f} scale={DETECTION_SCALE}")
    print("q=quit, r=reload users")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02); continue
            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                fps = n / (now - t0); t0 = now; n = 0

            gate_id = gate_for_now()

            # detect
            boxes = detector.detect(frame, scale=DETECTION_SCALE, min_size=MIN_FACE_SIZE)
            if SHOW_BOXES:
                for (x,y,w,h) in boxes:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)

            # choose largest + identify with detector='skip'
            if boxes:
                x,y,w,h = max(boxes, key=lambda b: b[2]*b[3])
                last_face_box = (x,y,w,h)
                pad = int(max(w,h) * 0.30)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face = frame[y1:y2, x1:x2]

                matches = recog.identify(face, known_users, top_k=1, detector_backend_override="skip")
                if matches:
                    m = matches[0]
                    last_dist = m.distance
                    if m.matched:
                        votes.append((m.user_id, m.name, m.distance))
                    else:
                        votes.append((None, None, m.distance))
            else:
                votes.append((None, None, None))

            # voting
            voted_uid = None; voted_name = None; voted_dist = None
            if len(votes) >= VOTE_MIN_SAME:
                counts: Dict[int,int] = {}
                dmap: Dict[int,List[float]] = {}
                for uid, name, dist in votes:
                    if uid is not None:
                        counts[uid] = counts.get(uid, 0) + 1
                        if dist is not None:
                            dmap.setdefault(uid, []).append(dist)
                if counts:
                    best = max(counts, key=lambda k: counts[k])
                    if counts[best] >= VOTE_MIN_SAME:
                        voted_uid = best
                        # latest name
                        for uid, name, dist in reversed(votes):
                            if uid == best:
                                voted_name = name; break
                        if best in dmap and dmap[best]:
                            voted_dist = float(sum(dmap[best]) / len(dmap[best]))

            # message
            if not gate_id:
                msg = "NO GATE OPEN"; color = (0,0,255)
            else:
                if voted_uid is not None:
                    key = (voted_uid, gate_id)
                    if time.time() - last_logged[key] >= COOLDOWN_SEC:
                        # write attendance
                        ts = dt.datetime.now()
                        date_iso = ts.date().isoformat()
                        rec = (sess.query(Attendance)
                               .filter_by(user_id=voted_uid, date=date_iso)
                               .one_or_none())
                        if not rec:
                            rec = Attendance(user_id=voted_uid, date=date_iso, quarter_id=current_quarter_id(ts.date()))
                            sess.add(rec); sess.flush()
                        if write_gate_timestamp(rec, gate_id, ts.strftime("%H:%M:%S"), overwrite=True):
                            sess.commit(); last_logged[key] = time.time()
                            msg = f"✓ {voted_name} → Gate {gate_id} (d={voted_dist:.3f} ≤ {DIST_THRESHOLD:.2f})"
                            color = (0,255,0)
                        else:
                            msg = f"{voted_name} → Gate {gate_id} (already set)"
                            color = (255,255,0)
                    else:
                        remain = COOLDOWN_SEC - int(time.time() - last_logged[key])
                        msg = f"{voted_name} cooldown {remain}s (d={voted_dist:.3f})"
                        color = (255,165,0)
                else:
                    if SHOW_DISTANCE and last_dist is not None:
                        msg = f"Gate {gate_id} • Analyzing (d={last_dist:.3f} / thr={DIST_THRESHOLD:.2f})"
                    else:
                        msg = f"Gate {gate_id} • Scanning…"
                    color = (0,255,255)

            # draw focus box
            if last_face_box:
                x,y,w,h = last_face_box
                box_color = (0,255,0) if voted_uid else (0,255,255)
                cv2.rectangle(frame, (x,y), (x+w,y+h), box_color, 2)
                label = voted_name if voted_name else "Scanning…"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-th-8), (x+tw+10, y), box_color, -1)
                cv2.putText(frame, label, (x+5, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # HUD
            cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0,0,0), -1)
            cv2.putText(frame, msg, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"FPS {fps:.1f}  Users {len(known_users)}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Real-Time Attendance (q=quit, r=reload)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                known_users = load_known_users(sess)
                print(f"[db] reloaded users: {len(known_users)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sess.close()

# export for main.py compatibility
engine_main = main

if __name__ == "__main__":
    main()
