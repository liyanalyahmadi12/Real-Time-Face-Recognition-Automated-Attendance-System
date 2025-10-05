# seed_users.py
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from deepface import DeepFace

from db import init_db, SessionLocal, User
from recognition import MODEL_NAME  # keep model consistent with runtime

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_MARGIN = 0.30  # enlarge box by 30% on each side

def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def read_image_bgr(path: Path) -> np.ndarray:
    im = Image.open(str(path))
    im = ImageOps.exif_transpose(im)      # fix EXIF rotation
    rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def crop_with_haar(img_bgr: np.ndarray, margin: float = FACE_MARGIN) -> np.ndarray | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    # more tolerant settings for static photos
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # take the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    H, W = img_bgr.shape[:2]
    # expand with margin
    mx = int(margin * w); my = int(margin * h)
    x1 = max(0, x - mx); y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    face = img_bgr[y1:y2, x1:x2].copy()
    # standardize size a bit (DeepFace will handle alignment off)
    face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
    return face

def try_embed(model, img_bgr: np.ndarray) -> np.ndarray | None:
    # 1) Try DeepFace’s own detectors at a couple of scales
    for det in ("retinaface", "mtcnn", "opencv"):
        for scale in (1.0, 0.85, 0.70):
            try:
                src = img_bgr if scale == 1.0 else cv2.resize(img_bgr, None, fx=scale, fy=scale)
                # mild contrast normalization
                lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab2 = cv2.merge((clahe.apply(l), a, b))
                rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

                reps = DeepFace.represent(
                    img_path=rgb,
                    model_name=MODEL_NAME,
                    model=model,
                    detector_backend=det,
                    enforce_detection=True,
                    align=True,
                )
                if reps:
                    emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
                    return l2norm(emb)
            except Exception:
                pass

    # 2) Fallback: Haar crop + skip detector
    face = crop_with_haar(img_bgr)
    if face is not None:
        try:
            reps = DeepFace.represent(
                img_path=cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
                model_name=MODEL_NAME,
                model=model,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
            )
            if reps:
                emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
                return l2norm(emb)
        except Exception:
            pass
    return None

def main():
    init_db()
    sess = SessionLocal()

    print(f"Loading model {MODEL_NAME} for enrollment...")
    model = DeepFace.build_model(MODEL_NAME)
    print(f"Model {MODEL_NAME} ready.")

    # ✅ resolve seed_images relative to this file (or allow override via SEED_DIR)
    SCRIPT_DIR = Path(__file__).resolve().parent
    p = Path(os.getenv("SEED_DIR", SCRIPT_DIR / "seed_images"))
    p.mkdir(exist_ok=True)
    print("Looking for images in:", p)  # helpful debug

    files = {
        f.resolve()
        for pat in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
        for f in p.glob(pat)
    }
    if not files:
        print("No images found in:", p)
        return

    total = 0
    for path in sorted(files):
        name = path.stem
        print(f"Processing: {path} (name='{name}')")
        img = read_image_bgr(path)
        if img is None:
            print("  Skip: cannot read file"); continue

        emb = try_embed(model, img)
        if emb is None:
            print("  Skip: face not detected or embedding failed after fallbacks.")
            continue

        u = sess.query(User).filter_by(name=name).one_or_none()
        if u:
            u.face_embedding = json.dumps(emb.tolist())
            print(f"  Updated: {name}")
        else:
            sess.add(User(name=name, face_embedding=json.dumps(emb.tolist())))
            print(f"  Enrolled: {name}")
        total += 1

    sess.commit(); sess.close()
    print(f"Done. Total successfully enrolled/updated: {total}")

if __name__ == "__main__":
    main()

