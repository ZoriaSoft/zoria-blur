"""Small-face detection modes (`--det-mode fast|balanced|thorough`).

Measured on WIDER FACE val (tests/eval_wider.py, see README): a single full-frame SCRFD pass at 640
finds ~0-4 % of faces under 10 px and ~15-37 % of 10-16 px faces; seeing the frame at >= its native
resolution (tiles, or a bigger det-size) roughly triples that, at 4-8x the cost. A bigger full-frame
det-size also LOSES large faces (the model was trained on 640 crops), so the large-face pass stays at 640.

  fast      one full-frame pass at det-size (640): the v0.2 behaviour.
  balanced  (default) fast until the video shows small people, then "small mode" (sticky):
            * scout (every SCOUT_EVERY detection frames while not in small mode): body detector on
              2x2 tiles; small mode starts when >= 2 confident (>= 0.4) bodies whose head would be
              < SMALL_HEAD_PX at the full-frame detector scale are seen (head size: see head_size),
              or the full pass finds a small face scoring >= 0.6;
            * in small mode: bodies on tiles every BODY_TILE_EVERY-th detection frame, and on every
              detection frame the full-frame face pass + a
              head-ROI pass: the head zones of small bodies and the small faces of the previous
              detection frame are merged into a few rectangles and re-detected upscaled so the
              smallest expected head is ~ROI_TARGET_PX at the detector input;
            * every REFRESH_EVERY detection frames a full tile pass also runs (faces whose body is
              hidden in a crowd / sitting people the body detector misses).
  thorough  full-frame pass + overlapping tiles (frame seen at >= 1.3x native) on every detection
            frame, bodies on tiles. Highest recall, ~6-9x the fast face-detector cost.
"""
from __future__ import annotations

import numpy as np

from .detector import auto_grid, detect_tiled, merge_dets, tile_grid

DET_MODES = ("fast", "balanced", "thorough")
SMALL_HEAD_PX = 20      # a head under this many px at the full-pass detector input counts as small
ROI_TARGET_PX = 24      # ROI pass: upscale so the smallest expected head is ~this at the detector input
BODY_TILE_EVERY = 2     # small mode: bodies on tiles every Nth detection frame (full-frame + Kalman between)
ROI_MAX_UP = 2.0
ROI_MAX_PIXELS = 0.8e6  # cap on the summed ROI detector input per frame (keeps CPU cost bounded)
SCOUT_EVERY = 10
REFRESH_EVERY = 5
MAX_TILES = 12
BODY_BORDER_PX = 8


def face_tile_grid(w: int, h: int):
    cols, rows = auto_grid(w, h, 640, upscale=1.3)
    while cols * rows > MAX_TILES:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    return cols, rows


def body_tile_grid(w: int, h: int):
    return (2, 2) if max(w, h) <= 2000 else (3, 3)


def detect_persons_tiled(pdet, img, thresh, grid=(2, 2), overlap=0.2):
    """Full-frame + tiled person detection merged with NMS. The body detector runs at 416 px, so in
    a 720p street scene it misses most people under ~100 px tall; on 2x2 tiles it sees them at
    ~0.6x native (real clip: 15 -> 48 bodies)."""
    H, W = img.shape[:2]
    full = pdet.detect(img, thresh)
    parts = [(full, np.zeros((len(full), 5, 2), np.float32))]
    for (x1, y1, x2, y2) in tile_grid(W, H, grid[0], grid[1], overlap):
        d = pdet.detect(img[y1:y2, x1:x2], thresh)
        if not len(d):
            continue
        d = d.copy(); d[:, 0:4:2] += x1; d[:, 1:4:2] += y1
        m = np.ones(len(d), bool)
        # a body cut by an inner tile border is the neighbour's (overlap) or the full pass's job.
        # YOLOX boxes of a cut body often end a few px before the border, hence the margin.
        mg = BODY_BORDER_PX
        if x1 > 0: m &= d[:, 0] > x1 + mg
        if y1 > 0: m &= d[:, 1] > y1 + mg
        if x2 < W: m &= d[:, 2] < x2 - mg
        if y2 < H: m &= d[:, 3] < y2 - mg
        parts.append((d[m], np.zeros((int(m.sum()), 5, 2), np.float32)))
    d, _ = merge_dets(parts, 0.5)
    return drop_body_parts(d)


def drop_body_parts(d):
    """Drop a box that is the upper part of a bigger body box (same top, about the same width, shorter):
    a tile sees only the head-and-torso of a person the full pass sees whole. As a separate body with no
    face it became an 'unknown' person whose head blur covered the kept runner's face (real clip, 36 %
    of his frames). A child in front of an adult has a lower top and a narrower box, so it stays."""
    if len(d) < 2:
        return d
    keep = np.ones(len(d), bool)
    w = d[:, 2] - d[:, 0]; h = d[:, 3] - d[:, 1]
    for a in range(len(d)):
        for b in range(len(d)):
            if a == b or not keep[b] or h[a] >= 0.85 * h[b]:
                continue
            ox = min(d[a, 2], d[b, 2]) - max(d[a, 0], d[b, 0])
            if (abs(d[a, 1] - d[b, 1]) < 0.12 * h[b] and ox >= 0.7 * w[a] and w[a] >= 0.6 * w[b]
                    and w[a] <= 1.4 * w[b]):
                keep[a] = False
                break
    return d[keep]


def head_size(bodies):
    """Expected head width (px) of person boxes. A whole standing person (tall box) has a head of
    ~12 % of its height; a box that is not tall is a partial body (head and shoulders behind someone,
    cut by the frame) whose head is ~40 % of its width (real clip: a 156x235 head-and-shoulders box of
    a man in a group selfie counted as a 'small person' and switched small mode on for nothing)."""
    b = np.asarray(bodies, np.float64).reshape(-1, 5 if np.asarray(bodies).shape[-1] >= 5 else 4)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.where(bh >= 2.2 * bw, 0.12 * bh, 0.4 * bw)


def head_zone(body):
    """Generous region that holds the head of a person body box (top part, widened)."""
    x1, y1, x2, y2 = body[:4]
    bw, bh = x2 - x1, y2 - y1
    head = float(head_size([body[:4]])[0])
    hw = max(2.5 * head, bw * 0.8)
    hh = 2.5 * head
    cx = (x1 + x2) / 2
    return np.array([cx - hw / 2, y1 - 0.5 * head, cx + hw / 2, y1 + hh]), head


def merge_rois(rects, gap=24.0):
    """Greedy union of rectangles that overlap or are within `gap` px: list of [x1,y1,x2,y2,min_head]."""
    rs = [list(r) for r in rects]
    changed = True
    while changed and len(rs) > 1:
        changed = False
        out = []
        while rs:
            a = rs.pop()
            i = 0
            while i < len(rs):
                b = rs[i]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), min(a[4], b[4])]
                    rs.pop(i); changed = True
                else:
                    i += 1
            out.append(a)
        rs = out
    return rs


class SmallFaceDetector:
    """Wraps SCRFD (+ optional YOLOX) and decides per detection frame which passes to run."""

    def __init__(self, det, pdet, mode: str, width: int, height: int, *, full_size=640,
                 person_thresh=0.3):
        if mode not in DET_MODES:
            raise ValueError(f"det mode must be one of {DET_MODES}")
        self.det, self.pdet, self.mode = det, pdet, mode
        self.W, self.H = width, height
        self.full_size = full_size
        self.full_scale = min(full_size / width, full_size / height)
        self.person_thresh = person_thresh
        self.face_grid = face_tile_grid(width, height)
        self.body_grid = body_tile_grid(width, height)
        self.small_mode = mode == "thorough"
        self.small_since = 0 if self.small_mode else None
        self.prev_small: list = []          # small faces of the previous detection frame
        self.n = 0
        self.stats = {"tile_passes": 0, "roi_passes": 0, "roi_pixels": 0.0, "scouts": 0,
                      "tiled_body_passes": 0}

    # -- bodies --------------------------------------------------------------------------
    def bodies(self, frame, k):
        """Person boxes for detection frame k (None when no body model)."""
        if self.pdet is None:
            return None
        scout = (self.mode == "balanced" and not self.small_mode and k % SCOUT_EVERY == 0)
        tiled = self.small_mode and (self.mode == "thorough" or (k - self.small_since) % BODY_TILE_EVERY == 0)
        if tiled or scout:
            self.stats["tiled_body_passes"] += 1
            if scout:
                self.stats["scouts"] += 1
            pd = detect_persons_tiled(self.pdet, frame, self.person_thresh, self.body_grid)
            if scout and self._n_small_bodies(pd) >= 2:
                self._enter_small(k)
            return pd
        return self.pdet.detect(frame, self.person_thresh)

    def _n_small_bodies(self, pd):
        if not len(pd):
            return 0
        head = head_size(pd[:, :4])
        return int((((head * self.full_scale) < SMALL_HEAD_PX) & (pd[:, 4] >= 0.4)).sum())

    def _enter_small(self, k):
        if not self.small_mode:
            self.small_mode = True
            self.small_since = k

    # -- faces ---------------------------------------------------------------------------
    def faces(self, frame, thresh, k, bodies=None):
        self.n += 1
        if self.mode == "fast":
            return self.det.detect(frame, thresh, det_size=self.full_size)
        if self.mode == "thorough":
            self.stats["tile_passes"] += 1
            return detect_tiled(self.det, frame, thresh, self.face_grid, 640, self.full_size)
        # balanced
        full = self.det.detect(frame, thresh, det_size=self.full_size)
        if not self.small_mode and len(full):
            fw = (full[0][:, 2] - full[0][:, 0]) * self.full_scale
            if int(((fw < SMALL_HEAD_PX) & (full[0][:, 4] >= 0.6)).sum()) >= 1:
                self._enter_small(k)
        if not self.small_mode:
            return full
        parts = [full]
        if (k - self.small_since) % REFRESH_EVERY == 0:
            self.stats["tile_passes"] += 1
            parts.append(detect_tiled(self.det, frame, thresh, self.face_grid, 640, None))
        parts.append(self._roi_pass(frame, thresh, bodies, full))
        d, kk = merge_dets(parts, 0.4)
        self._remember_small(d)
        return d, kk

    def _remember_small(self, d):
        w = d[:, 2] - d[:, 0] if len(d) else np.zeros(0)
        self.prev_small = [d[i, :4].copy() for i in range(len(d)) if w[i] * self.full_scale < 1.5 * SMALL_HEAD_PX]

    def _roi_pass(self, frame, thresh, bodies, full):
        rects = []
        if bodies is not None and len(bodies):
            for b in bodies:
                z, head = head_zone(b)
                if head * self.full_scale < 1.5 * SMALL_HEAD_PX:
                    rects.append(list(z) + [max(head, 4.0)])
        for fb in self.prev_small:
            w, h = fb[2] - fb[0], fb[3] - fb[1]
            rects.append([fb[0] - 1.5 * w, fb[1] - 1.5 * h, fb[2] + 1.5 * w, fb[3] + 1.5 * h, max(w, 4.0)])
        if not rects:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        rois = merge_rois(rects)
        plans = []
        for r in rois:
            x1, y1 = max(0, int(r[0])), max(0, int(r[1]))
            x2, y2 = min(self.W, int(np.ceil(r[2]))), min(self.H, int(np.ceil(r[3])))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            up = float(np.clip(ROI_TARGET_PX / r[4], 1.0, ROI_MAX_UP))
            plans.append([x1, y1, x2, y2, up])
        total = sum((p[2] - p[0]) * (p[3] - p[1]) * p[4] ** 2 for p in plans)
        shrink = min(1.0, (ROI_MAX_PIXELS / total) ** 0.5) if total > 0 else 1.0
        parts = []
        for x1, y1, x2, y2, up in plans:
            up = max(1.0, up * shrink)
            crop = frame[y1:y2, x1:x2]
            ds = int(round(max(crop.shape[:2]) * up))
            d, kk = self.det.detect(crop, thresh, det_size=ds)
            self.stats["roi_passes"] += 1
            self.stats["roi_pixels"] += (x2 - x1) * (y2 - y1) * up * up
            if not len(d):
                continue
            d = d.copy(); kk = kk.copy()
            d[:, 0:4:2] += x1; d[:, 1:4:2] += y1; kk[..., 0] += x1; kk[..., 1] += y1
            m = np.ones(len(d), bool)
            if x1 > 0: m &= d[:, 0] > x1 + 1
            if y1 > 0: m &= d[:, 1] > y1 + 1
            if x2 < self.W: m &= d[:, 2] < x2 - 1
            if y2 < self.H: m &= d[:, 3] < y2 - 1
            parts.append((d[m], kk[m]))
        return merge_dets(parts, 0.4)

    def info(self):
        return {"det_mode": self.mode, "small_mode": bool(self.small_mode), "small_mode_from_det_frame": self.small_since,
                "face_tile_grid": list(self.face_grid), "body_tile_grid": list(self.body_grid),
                **{k: (round(v) if isinstance(v, float) else v) for k, v in self.stats.items()}}
