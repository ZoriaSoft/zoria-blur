"""YOLOX person detector (Megvii YOLOX, Apache-2.0 code AND weights).

Uses the official ONNX exports from the YOLOX GitHub release 0.1.1rc0
(yolox_nano / yolox_tiny: 416x416 input, yolox_s: 640x640). Only COCO class 0 (person)
is kept. Output: Nx5 [x1, y1, x2, y2, score] in original-image pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .backend import make_session
from .detector import _nms

PERSON_MODELS = {"yolox_nano": "yolox_nano.onnx", "yolox_tiny": "yolox_tiny.onnx", "yolox_s": "yolox_s.onnx"}

# the 0.1.1rc0 ONNX exports take raw BGR 0..255 (verified on real frames); legacy=True = ImageNet norm
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class YOLOXPerson:
    def __init__(self, model_path: Path | str, backend: str = "cpu", threads: int = 0,
                 nms_thresh: float = 0.5, legacy: bool = False):
        self.session = make_session(model_path, backend, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = (int(inp.shape[2]), int(inp.shape[3]))
        self.nms_thresh = nms_thresh
        self.legacy = legacy
        self._grids = None

    def _decode(self, out):
        if self._grids is None:
            grids, strides = [], []
            H, W = self.size
            for s in (8, 16, 32):
                hs, ws = H // s, W // s
                xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
                grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
                strides.append(np.full((hs * ws, 1), s))
            self._grids = (np.concatenate(grids, 0).astype(np.float32),
                           np.concatenate(strides, 0).astype(np.float32))
        g, st = self._grids
        out = out.copy()
        out[:, :2] = (out[:, :2] + g) * st
        out[:, 2:4] = np.exp(out[:, 2:4]) * st
        return out

    def detect(self, img: np.ndarray, thresh: float = 0.3) -> np.ndarray:
        h, w = img.shape[:2]
        H, W = self.size
        r = min(H / h, W / w)
        nh, nw = int(h * r), int(w * r)
        pad = np.full((H, W, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if self.legacy:
            x = (pad[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD
        else:
            x = pad.astype(np.float32)
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = self.session.run(None, {self.input_name: blob})[0][0]
        out = self._decode(out)
        scores = out[:, 4] * out[:, 5]            # objectness * P(person)
        keep = scores >= thresh
        if not keep.any():
            return np.zeros((0, 5), np.float32)
        o, s = out[keep], scores[keep]
        boxes = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2,
                          o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2], 1) / r
        dets = np.hstack([boxes, s[:, None]]).astype(np.float32)
        dets = dets[np.argsort(-dets[:, 4])]
        dets = dets[_nms(dets, self.nms_thresh)]
        dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, w - 1)
        dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, h - 1)
        return dets
