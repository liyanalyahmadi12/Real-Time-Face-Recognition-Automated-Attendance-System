# recognition.py
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

# -------------------------------------------------------
# Configuration (env overrides)
# -------------------------------------------------------
MODEL_NAME     = os.getenv("MODEL_NAME", "Facenet")  # "Facenet" is fast; "Facenet512" is more precise but slower
DEFAULT_DET    = os.getenv("DETECTOR", "opencv")     # used when we DON'T pass a face crop
DIST_THRESHOLD = float(os.getenv("DIST_THRESHOLD", "0.75"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# -------------------------------------------------------
# Result object
# -------------------------------------------------------
@dataclass
class FaceMatch:
    user_id: Optional[int] = None
    name: Optional[str] = None
    distance: float = float("inf")
    confidence: float = 0.0
    matched: bool = False
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "distance": round(self.distance, 4),
            "confidence": round(self.confidence, 4),
            "matched": self.matched,
            "quality_score": round(self.quality_score, 2),
        }

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

def _simple_quality(face_bgr: np.ndarray) -> float:
    """0..100 score from blur + brightness (no detector fields required)."""
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 800.0, 1.0) * 100.0
    mean = float(np.mean(gray))
    bright_score = 100.0 - min(abs(mean - 128.0) / 128.0, 1.0) * 100.0
    return max(0.0, min(100.0, 0.6 * blur_score + 0.4 * bright_score))

def _preprocess_bgr_for_embed(bgr: np.ndarray) -> np.ndarray:
    """Light CLAHE for robustness; returns RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab2 = cv2.merge((clahe.apply(l), a, b))
    rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb2

# -------------------------------------------------------
# Engine
# -------------------------------------------------------
class FaceRecognitionEngine:
    """
    DeepFace-based face recognition with:
      - warmup & caching
      - robust embedding extraction (works with or without prior detection)
      - vectorized cosine distance
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        detector: str   = DEFAULT_DET,
        dist_threshold: float = DIST_THRESHOLD,
    ):
        self.model_name     = model_name
        self.detector       = detector
        self.dist_threshold = dist_threshold
        self._warmup_model()

    # --------------- model warmup ---------------
    def _warmup_model(self) -> None:
        try:
            # ensure model weights are downloaded/cached
            DeepFace.build_model(self.model_name)
            # compile a dummy inference so first real call is faster
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                # some DeepFace versions accept align=
                DeepFace.represent(img_path=dummy, model_name=self.model_name,
                                   detector_backend="skip", enforce_detection=False, align=False)
            except TypeError:
                DeepFace.represent(img_path=dummy, model_name=self.model_name,
                                   detector_backend="skip", enforce_detection=False)
            log.info("Loaded model %s", self.model_name)
        except Exception as e:
            log.warning("Model warmup failed: %s", e)

    # --------------- embedding ---------------
    def embed_face(
        self,
        img_bgr: np.ndarray,
        detector_backend: Optional[str] = None,  # override (e.g., "skip" when we already have a crop)
        return_quality: bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (embedding, quality).
        If no face found, returns (None, 0.0).
        """
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0

        rgb = _preprocess_bgr_for_embed(img_bgr)
        det = detector_backend or self.detector

        try:
            try:
                reps = DeepFace.represent(
                    img_path=rgb,
                    model_name=self.model_name,
                    detector_backend=det,
                    enforce_detection=False if det == "skip" else True,
                    align=False if det == "skip" else True,
                )
            except TypeError:
                # older DeepFace may not support align=
                reps = DeepFace.represent(
                    img_path=rgb,
                    model_name=self.model_name,
                    detector_backend=det,
                    enforce_detection=False if det == "skip" else True,
                )
        except Exception as e:
            log.debug("represent() failed (%s)", e)
            reps = None

        if not reps:
            return None, 0.0

        emb = _l2norm(np.asarray(reps[0]["embedding"], dtype=np.float32))
        q   = _simple_quality(img_bgr) if return_quality else 0.0
        return emb, q

    # --------------- distances ---------------
    @staticmethod
    def cosine_distances(query_emb: np.ndarray, known_embs: np.ndarray) -> np.ndarray:
        """1 - cosine similarity for L2-normalized inputs."""
        return 1.0 - np.dot(known_embs, query_emb)

    # --------------- identify ---------------
    def identify(
        self,
        img_bgr: np.ndarray,
        known_users: List[Dict[str, Any]],
        top_k: int = 1,
        detector_backend_override: Optional[str] = "skip",  # we pass cropped faces → skip re-detection
    ) -> List[FaceMatch]:
        """
        Compares a face crop against known users.
        known_users: list of dicts {user_id, name, embedding(np.ndarray or list)}
        Returns sorted FaceMatch list (top_k).
        """
        if not known_users:
            return [FaceMatch(quality_score=0.0)]

        # Ensure matrix of normalized embeddings
        kb = []
        meta = []
        for u in known_users:
            e = np.asarray(u["embedding"], dtype=np.float32)
            kb.append(_l2norm(e))
            meta.append((u.get("user_id"), u.get("name")))
        known_matrix = np.stack(kb, axis=0)

        emb, quality = self.embed_face(img_bgr, detector_backend=detector_backend_override, return_quality=True)
        if emb is None:
            return [FaceMatch(quality_score=quality)]

        dists = self.cosine_distances(emb, known_matrix)
        order = np.argsort(dists)[: max(1, top_k)]

        out: List[FaceMatch] = []
        for idx in order:
            idx = int(idx)
            d = float(dists[idx])
            uid, name = meta[idx]
            # cosine distance [0..2], for L2-normalized tends to [0..2], closer=better
            # Convert to a simple confidence proxy:
            conf = max(0.0, 1.0 - d / 2.0)
            matched = (d <= self.dist_threshold) and (conf >= MIN_CONFIDENCE)
            out.append(FaceMatch(user_id=uid, name=name, distance=d, confidence=conf, matched=matched, quality_score=quality))
        return out

# -------------------------------------------------------
# Legacy helpers for backward compatibility
# -------------------------------------------------------
_default_engine: Optional[FaceRecognitionEngine] = None

def get_engine() -> FaceRecognitionEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = FaceRecognitionEngine()
    return _default_engine

def embed_face(img_bgr: np.ndarray) -> np.ndarray:
    eng = get_engine()
    emb, _ = eng.embed_face(img_bgr, detector_backend="skip")
    if emb is None:
        raise ValueError("No face embedding returned.")
    return emb

def identify(img_bgr: np.ndarray, known_users: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    eng = get_engine()
    matches = eng.identify(img_bgr, known_users, top_k=1, detector_backend_override="skip")
    if not matches or not matches[0].matched:
        return None, matches[0].distance if matches else None
    m = matches[0]
    return {"user_id": m.user_id, "name": m.name}, m.distance

if __name__ == "__main__":
    print("recognition.py is a library. Run live_engine.py to test camera.")
