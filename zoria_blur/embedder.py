"""ArcFace face embedder (InsightFace w600k_mbf MobileFaceNet, 512-d, L2-normalised)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .backend import make_session

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_face(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(kps[:3].astype(np.float32), dst[:3])
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def face_quality(box: np.ndarray, kps: np.ndarray) -> float:
    """Cheap frontalness*size score in [0,1+]: used to pick thumbnails / weight embeddings."""
    w = box[2] - box[0]
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2
    eye_dist = np.linalg.norm(re - le) + 1e-6
    yaw = abs(nose[0] - eye_mid[0]) / eye_dist          # 0 = frontal
    frontal = max(0.0, 1.0 - yaw * 1.5)
    size = min(1.0, w / 112.0)
    return float(frontal * size)


# GPU batch sizes (CUDA / DirectML: one session per input shape, backend.SafeSession). DirectML has no batch 1: its
# first pinned batch-1 w600k_mbf session takes 2.1-2.9 s to create on an AMD test GPU (batch 2/4/8: 0.3-0.5 s).
EMB_BATCHES = {"cuda": (1, 2, 4, 8), "dml": (2, 4, 8)}


class ArcFace:
    def __init__(self, model_path: Path | str, backend: str = "cpu", threads: int = 0):
        self.session = make_session(model_path, backend, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 112
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, img: np.ndarray, kps_list: list[np.ndarray]) -> np.ndarray:
        if len(kps_list) == 0:
            return np.zeros((0, 512), np.float32)
        crops = [align_face(img, k, self.size) for k in kps_list]
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (self.size, self.size),
                                      (127.5, 127.5, 127.5), swapRB=True)
        feats = self._run(blob)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        return feats.astype(np.float32)

    def _run(self, blob: np.ndarray) -> np.ndarray:
        """CPU runs the batch as it is. CUDA / DirectML keep one session per input shape, and every new batch size
        would cost a session (~0.25 s on DirectML): batches are cut to <= 8 and zero-padded to EMB_BATCHES (each face
        is embedded on its own inside the batch, so padding changes no result)."""
        sizes = EMB_BATCHES.get(getattr(self.session, "backend", "cpu"))
        if not sizes:
            return self.session.run([self.output_name], {self.input_name: blob})[0]
        out = []
        for i in range(0, len(blob), sizes[-1]):
            part = blob[i:i + sizes[-1]]
            n = len(part)
            b = next(x for x in sizes if x >= n)
            if b > n:
                part = np.concatenate([part, np.zeros((b - n,) + part.shape[1:], part.dtype)])
            out.append(self.session.run([self.output_name], {self.input_name: part})[0][:n])
        return np.concatenate(out)
