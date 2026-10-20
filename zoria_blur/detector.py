"""SCRFD face detector (InsightFace det_10g / det_2.5g / det_500m ONNX)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import os

from .backend import make_session

# Square input sizes the detector uses for crops on CUDA / DirectML (fixed set; bigger crops round up to a multiple of 320).
DET_BUCKETS = (160, 320, 480, 640)


def det_bucket(n: int) -> int:
    for b in DET_BUCKETS:
        if n <= b:
            return b
    return -(-n // 320) * 320


def _fixed_shapes(backend: str) -> bool:
    """CUDA and DirectML run each distinct input shape on its own session (backend.SafeSession), so the detector
    keeps its shapes to a small fixed set there. ZORIA_DET_FIXED_SHAPES=1/0 forces it on/off on any backend (tests)."""
    env = os.environ.get("ZORIA_DET_FIXED_SHAPES", "")
    return env not in ("", "0") if env else backend in ("cuda", "dml")


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    """Detects faces. Returns (boxes Nx5 [x1,y1,x2,y2,score], kps Nx5x2) in input-image pixels.

    The frame is downscaled (aspect kept, letterboxed top-left) to `det_size`
    before inference, so cost is independent of video resolution.
    """

    def __init__(self, model_path: Path | str, backend: str = "cpu", det_size: int = 640,
                 threads: int = 0, nms_thresh: float = 0.4):
        self.session = make_session(model_path, backend, threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_size = int(det_size)
        self.nms_thresh = nms_thresh
        n_out = len(self.output_names)
        if n_out in (6, 9):
            self.fmc, self.strides, self.num_anchors = 3, [8, 16, 32], 2
        elif n_out in (10, 15):
            self.fmc, self.strides, self.num_anchors = 5, [8, 16, 32, 64, 128], 1
        else:
            raise RuntimeError(f"unexpected SCRFD output count {n_out}")
        self.use_kps = n_out in (9, 15)
        if not self.use_kps:
            raise RuntimeError("this SCRFD model has no landmarks; use a *_kps / det_* model")
        self._center_cache: dict = {}

    def detect(self, img: np.ndarray, thresh: float = 0.5, max_num: int = 0, det_size: int | None = None,
               nms_thresh: float | None = None):
        h, w = img.shape[:2]
        ds = int(det_size or self.det_size)
        scale = min(ds / w, ds / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        # the SCRFD exports take any H, W that are multiples of the largest stride: pad the resized
        # frame to the next multiple instead of a ds x ds square (a 16:9 frame wasted ~44 % of the
        # square input on padding; same scale, same detections, ~40 % less compute).
        m = max(self.strides)
        ph, pw = -(-nh // m) * m, -(-nw // m) * m
        backend = getattr(self.session, "backend", "cpu")
        # (DirectML used to need 640x640 here: the exports declare 640x640 output shapes, freed in backend.py)
        if ds != self.det_size and _fixed_shapes(backend):
            # small-face ROIs / face-split crops come at any size: pad to a square bucket (zero padding
            # bottom/right, same scale, so the same faces at the same pixels). The full frame and the 640 px
            # tiles keep their exact padded shape: one or two shapes per video.
            ph = pw = det_bucket(max(ph, pw))
        det_img = np.zeros((ph, pw, 3), dtype=np.uint8)
        det_img[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, (pw, ph), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + self.fmc].reshape(-1, 4) * stride
            kps = outs[idx + self.fmc * 2].reshape(-1, 10) * stride
            fh, fw = ph // stride, pw // stride
            key = (fh, fw, stride)
            centers = self._center_cache.get(key)
            if centers is None:
                centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                if self.num_anchors > 1:
                    centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
                self._center_cache[key] = centers
            pos = np.where(scores >= thresh)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            b = bbox[pos]
            boxes = np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=-1)
            k = kps[pos].reshape(-1, 5, 2) + c[:, None, :]
            scores_l.append(scores[pos])
            boxes_l.append(boxes)
            kps_l.append(k)
        if not scores_l:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l) / scale
        kps = np.concatenate(kps_l) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        order = scores.argsort()[::-1]
        dets, kps = dets[order], kps[order]
        keep = _nms(dets, self.nms_thresh if nms_thresh is None else nms_thresh)
        dets, kps = dets[keep], kps[keep]
        # clip to image
        dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, w - 1)
        dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, h - 1)
        if max_num and dets.shape[0] > max_num:
            dets, kps = dets[:max_num], kps[:max_num]
        return dets, kps.astype(np.float32)

    def detect_region(self, img: np.ndarray, box, thresh: float = 0.3, expand: float = 1.0,
                      det_size: int = 320, nms_thresh: float = 0.6):
        """Re-detect inside an upscaled crop around `box` (x1,y1,x2,y2) with a looser NMS
        (higher IoU threshold keeps overlapping faces apart). Used to split two faces that the
        full-frame pass merged into one box. Returns (dets, kps) in full-image pixels."""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw, bh = x2 - x1, y2 - y1
        X1, Y1 = int(max(0, x1 - expand * bw)), int(max(0, y1 - expand * bh))
        X2, Y2 = int(min(W, x2 + expand * bw)), int(min(H, y2 + expand * bh))
        crop = img[Y1:Y2, X1:X2]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        # upscale small crops so faces are ~the size the detector likes
        sc = max(1.0, det_size / max(crop.shape[:2]))
        if sc > 1.0:
            crop = cv2.resize(crop, (int(round(crop.shape[1] * sc)), int(round(crop.shape[0] * sc))),
                              interpolation=cv2.INTER_CUBIC)
        d, k = self.detect(crop, thresh, det_size=det_size, nms_thresh=nms_thresh)
        if len(d):
            d = d.copy(); k = k.copy()
            d[:, :4] /= sc; k /= sc
            d[:, 0:4:2] += X1; d[:, 1:4:2] += Y1
            k[..., 0] += X1; k[..., 1] += Y1
        return d, k


def tile_grid(w: int, h: int, cols: int, rows: int, overlap: float = 0.2):
    """Overlapping tiles covering a w x h frame: list of (x1, y1, x2, y2)."""
    tw = int(np.ceil(w / (cols - (cols - 1) * overlap))) if cols > 1 else w
    th = int(np.ceil(h / (rows - (rows - 1) * overlap))) if rows > 1 else h
    xs = [0] if cols == 1 else [int(round(i * (w - tw) / (cols - 1))) for i in range(cols)]
    ys = [0] if rows == 1 else [int(round(j * (h - th) / (rows - 1))) for j in range(rows)]
    return [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]


def auto_grid(w: int, h: int, target: int = 640, upscale: float = 1.0):
    """Smallest cols x rows grid whose tiles, run at `target` px, see the frame at >= `upscale` x
    its native resolution (1.0 = every face at its real pixel size)."""
    best = None
    for cols in range(1, 6):
        for rows in range(1, 5):
            tw = w / (cols - (cols - 1) * 0.2) if cols > 1 else w
            th = h / (rows - (rows - 1) * 0.2) if rows > 1 else h
            if target / max(tw, th) >= upscale * 0.999:
                if best is None or cols * rows < best[0] * best[1]:
                    best = (cols, rows)
    return best or (5, 4)


def merge_dets(parts, nms_thresh: float = 0.4):
    """Merge (dets, kps) lists from several passes with NMS (highest score wins)."""
    parts = [(d, k) for d, k in parts if len(d)]
    if not parts:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    d = np.concatenate([p[0] for p in parts]); k = np.concatenate([p[1] for p in parts])
    o = np.argsort(-d[:, 4]); d, k = d[o], k[o]
    keep = _nms(d, nms_thresh)
    return d[keep], k[keep]


def detect_tiled(det: "SCRFD", img: np.ndarray, thresh: float, grid=(3, 2), tile_size: int = 640,
                 full_size: int | None = 640, overlap: float = 0.2, nms_thresh: float = 0.4):
    """Full-frame pass (large faces) + overlapping tiles each run at `tile_size` (small faces seen
    at a higher scale), merged with NMS. A tile detection touching an inner tile border is dropped
    (the neighbouring tile sees that face whole thanks to the overlap)."""
    H, W = img.shape[:2]
    parts = []
    if full_size:
        parts.append(det.detect(img, thresh, det_size=full_size))
    cols, rows = grid
    for (x1, y1, x2, y2) in tile_grid(W, H, cols, rows, overlap):
        d, k = det.detect(img[y1:y2, x1:x2], thresh, det_size=tile_size)
        if not len(d):
            continue
        d = d.copy(); k = k.copy()
        d[:, 0:4:2] += x1; d[:, 1:4:2] += y1; k[..., 0] += x1; k[..., 1] += y1
        m = np.ones(len(d), bool)
        if x1 > 0: m &= d[:, 0] > x1 + 2
        if y1 > 0: m &= d[:, 1] > y1 + 2
        if x2 < W: m &= d[:, 2] < x2 - 2
        if y2 < H: m &= d[:, 3] < y2 - 2
        # tiles exist for small faces; a face larger than ~half the tile is the full pass's job
        if full_size:
            m &= (d[:, 2] - d[:, 0]) < 0.5 * (x2 - x1)
        parts.append((d[m], k[m]))
    return merge_dets(parts, nms_thresh)
