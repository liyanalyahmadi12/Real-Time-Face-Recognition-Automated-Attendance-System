# enroll_webcam.py
import argparse, time, json
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from db import SessionLocal, init_db, User
from recognition import MODEL_NAME  # keep consistent with runtime

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MARGIN = 0.30  # expand face box
TARGET = (224, 224)  # embed size target

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def biggest_face(gray):
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])  # (x,y,w,h)

def crop_face(bgr, box, margin=MARGIN):
    x, y, w, h = box
    H, W = bgr.shape[:2]
    mx, my = int(margin * w), int(margin * h)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    face = bgr[y1:y2, x1:x2].copy()
    if face.size == 0: 
        return None
    return cv2.resize(face, TARGET, interpolation=cv2.INTER_LINEAR)

def embed_face_model(model, face_bgr):
    # slight contrast normalization
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab2 = cv2.merge((clahe.apply(l),a,b))
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    reps = DeepFace.represent(
        img_path=rgb,
        model_name=MODEL_NAME,
        model=model,
        detector_backend="skip",      # we already cropped
        enforce_detection=False,
        align=False,
    )
    if not reps:
        return None
    emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
    return l2norm(emb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person name to enroll (e.g., Liyan)")
    ap.add_argument("--samples", type=int, default=5, help="Number of captures to average (3–7 is good)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = ap.parse_args()

    init_db()
    print(f"Loading model {MODEL_NAME}...")
    model = DeepFace.build_model(MODEL_NAME)
    print("Model ready. Opening camera...")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    samples = []
    try:
        last_info = ""
        while True:
            ok, frame = cap.read()
            if not ok: 
                break

            disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = biggest_face(gray)

            if box is not None:
                x,y,w,h = box
                cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
                # blur metric
                face_roi = gray[max(0,y):y+h, max(0,x):x+w]
                blur = cv2.Laplacian(face_roi, cv2.CV_64F).var() if face_roi.size else 0.0
                last_info = f"face ok | blur={blur:.0f}"
            else:
                last_info = "no face"

            cv2.putText(disp, f"Enroll: {args.name}", (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(disp, f"samples {len(samples)}/{args.samples}  ({last_info})", (12,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
            cv2.putText(disp, "SPACE=capture  R=reset  Q=quit", (12,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
            cv2.imshow("Enroll", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                samples.clear()
                print("samples cleared")
            if key == 32:  # SPACE
                if box is None:
                    print("no face detected; hold steady facing camera")
                    continue
                face = crop_face(frame, box)
                if face is None:
                    print("bad crop; try again"); 
                    continue
                # basic blur check
                bvar = cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if bvar < 80:
                    print(f"too blurry ({bvar:.0f}); try again")
                    continue

                emb = embed_face_model(model, face)
                if emb is None:
                    print("embedding failed; try again")
                    continue
                samples.append(emb)
                print(f"captured {len(samples)}/{args.samples}")

                if len(samples) >= args.samples:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not samples:
        print("No samples captured; nothing saved.")
        return

    avg = l2norm(np.mean(np.vstack(samples), axis=0))
    sess = SessionLocal()
    u = sess.query(User).filter_by(name=args.name).one_or_none()
    if u:
        u.face_embedding = json.dumps(avg.tolist())
        print(f"Updated user: {args.name}")
    else:
        sess.add(User(name=args.name, face_embedding=json.dumps(avg.tolist())))
        print(f"Enrolled user: {args.name}")
    sess.commit(); sess.close()
    print("Done.")
    
if __name__ == "__main__":
    main()
