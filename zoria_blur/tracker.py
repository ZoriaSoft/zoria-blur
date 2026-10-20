"""BoT-SORT/ByteTrack-style face tracker: Kalman (constant velocity, xyah) + IoU,
two-stage high/low score association, and ArcFace re-ID for lost tracks.

The tracker is stepped only on detection frames; `gap` tells it how many video
frames passed since the previous step so the Kalman prediction covers them.
"""
from __future__ import annotations

import numpy as np
from .numeric import linear_sum_assignment


class KalmanXYAH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._wp = 1.0 / 20
        self._wv = 1.0 / 160

    def initiate(self, m):
        mean = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h,
               self._wv * h, self._wv * h, 1e-5, self._wv * h]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean, cov, m):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-1, self._wp * h]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ (m - self._H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m):
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


class Track:
    _kf = KalmanXYAH()

    def __init__(self, tid, box, score, emb, frame):
        self.id = tid
        self.mean, self.cov = self._kf.initiate(xyxy_to_xyah(box))
        self.emb = emb.copy() if emb is not None else None
        self.state = "tracked"   # tracked | lost | removed
        self.last_frame = frame
        self.start_frame = frame
        self.hits = 1
        # observations: list of (frame, box[4], score, det_index_in_frame)
        self.obs = []

    def predict(self, gap):
        m, c = self.mean, self.cov
        if self.state != "tracked":
            m = m.copy(); m[7] = 0  # freeze height velocity when lost
        for _ in range(max(1, gap)):
            m, c = self._kf.predict(m, c)
        self.mean, self.cov = m, c

    def box(self):
        return xyah_to_xyxy(self.mean)

    def update(self, box, score, emb, frame, alpha=0.9):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        if emb is not None:
            if self.emb is None:
                self.emb = emb.copy()
            else:
                e = alpha * self.emb + (1 - alpha) * emb
                self.emb = e / (np.linalg.norm(e) + 1e-9)
        self.state = "tracked"
        self.last_frame = frame
        self.hits += 1


class FaceTracker:
    def __init__(self, high_thresh=0.5, low_thresh=0.25, new_thresh=0.55,
                 match_iou=0.2, reid_sim=0.45, proximity_iou=0.1, buffer_frames=60):
        self.high, self.low, self.new = high_thresh, low_thresh, new_thresh
        self.match_dist = 1.0 - match_iou
        self.reid_sim = reid_sim          # cosine sim needed to re-attach a lost track by appearance
        self.prox_dist = 1.0 - proximity_iou
        self.buffer = buffer_frames
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1
        self._last_frame = None

    def _cost(self, tracks, boxes, embs, use_emb=True):
        tb = np.array([t.box() for t in tracks]).reshape(-1, 4)
        iou_d = 1.0 - iou_matrix(tb, boxes)
        if not use_emb or embs is None:
            return iou_d
        cost = iou_d.copy()
        for i, t in enumerate(tracks):
            if t.emb is None:
                continue
            for j in range(len(boxes)):
                if embs[j] is None:
                    continue
                emb_d = 1.0 - float(np.dot(t.emb, embs[j]))
                if t.state == "lost":
                    # appearance-only re-ID for lost tracks (person may have moved)
                    if 1.0 - emb_d >= self.reid_sim:
                        cost[i, j] = min(cost[i, j], emb_d * 0.5)
                else:
                    # BoT-SORT fusion: appearance only counts when boxes are near
                    if iou_d[i, j] < self.prox_dist and emb_d < 0.6:
                        cost[i, j] = min(cost[i, j], emb_d)
                    elif emb_d > 0.85 and iou_d[i, j] < self.match_dist:
                        cost[i, j] = (cost[i, j] + 1.0) / 2  # penalise strong appearance mismatch
        return cost

    @staticmethod
    def _assign(cost, max_cost):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        r, c = linear_sum_assignment(cost)
        matches, ut, ud = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
        for i, j in zip(r, c):
            if cost[i, j] <= max_cost:
                matches.append((i, j)); ut.discard(i); ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def step(self, frame_idx: int, dets: np.ndarray, embs: list):
        """dets: Nx5 boxes+score; embs: list of N embeddings (or None for faces too small)."""
        gap = 1 if self._last_frame is None else frame_idx - self._last_frame
        self._last_frame = frame_idx
        for t in self.tracks:
            t.predict(gap)
        scores = dets[:, 4] if len(dets) else np.zeros(0)
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]

        pool = [t for t in self.tracks if t.state in ("tracked", "lost")]
        # Stage 1: high-score dets vs all tracks (IoU + appearance)
        cost = self._cost(pool, dets[hi, :4], [embs[k] for k in hi])
        m1, ut, ud = self._assign(cost, self.match_dist)
        for i, j in m1:
            k = hi[j]
            pool[i].update(dets[k, :4], dets[k, 4], embs[k], frame_idx)
            pool[i].obs.append((frame_idx, dets[k, :4].tolist(), float(dets[k, 4]), int(k)))
        # Stage 2: low-score dets vs remaining *tracked* tracks, IoU only
        rem = [pool[i] for i in ut if pool[i].state == "tracked"]
        cost2 = self._cost(rem, dets[lo, :4], None, use_emb=False)
        m2, ut2, _ = self._assign(cost2, 0.5)
        for i, j in m2:
            k = lo[j]
            rem[i].update(dets[k, :4], dets[k, 4], embs[k], frame_idx)
            rem[i].obs.append((frame_idx, dets[k, :4].tolist(), float(dets[k, 4]), int(k)))
        matched_ids = {id(pool[i]) for i, _ in m1} | {id(rem[i]) for i, _ in m2}
        for t in pool:
            if id(t) not in matched_ids and t.state == "tracked":
                t.state = "lost"
        # New tracks
        for j in ud:
            k = hi[j]
            if dets[k, 4] >= self.new:
                t = Track(self._next, dets[k, :4], dets[k, 4], embs[k], frame_idx)
                t.obs.append((frame_idx, dets[k, :4].tolist(), float(dets[k, 4]), int(k)))
                self._next += 1
                self.tracks.append(t)
        # Retire old lost tracks
        alive = []
        for t in self.tracks:
            if t.state == "lost" and frame_idx - t.last_frame > self.buffer:
                t.state = "removed"
                self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive

    def finalize(self) -> list[Track]:
        out = self.finished + self.tracks
        self.finished, self.tracks = [], []
        return sorted(out, key=lambda t: t.id)
