"""Person (body) level tracking: the identity layer that survives hidden / turned / merged faces.

* BodyTracker  - ByteTrack-style Kalman (xyah) + IoU tracker over YOLOX person boxes, fused with a
                 torso HSV colour histogram so crossing people keep their ids.
* associate_faces - post-hoc: every face detection is matched (Hungarian) to the body whose upper part
                 contains it. A body then knows which face tracks (=> persons) it carried and when.
* link_bodies  - re-ID after occlusion: a body track that starts within a few seconds after another
                 ended is linked to it by torso appearance (never across tracks that coexist in time).
* head_box     - where the head of a body is in a frame where its face was not detected, learned from
                 the frames where both were seen (relative offset/size), with a generic prior otherwise.

Stored in project.json as data["bodies"] (schema in README).
"""
from __future__ import annotations

import cv2
import numpy as np
from .numeric import linear_sum_assignment

from .tracker import KalmanXYAH, iou_matrix, xyah_to_xyxy, xyxy_to_xyah


# ------------------------------------------------------------------ appearance
def torso_hist(img: np.ndarray, box) -> np.ndarray | None:
    """HSV histogram of the torso (centre 50 % width, 18-55 % height of the body box)."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    X1, X2 = int(max(0, x1 + 0.25 * w)), int(min(img.shape[1], x2 - 0.25 * w))
    Y1, Y2 = int(max(0, y1 + 0.18 * h)), int(min(img.shape[0], y1 + 0.55 * h))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return None
    hsv = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else None


def hist_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


# ------------------------------------------------------------------ tracker
class BodyTrack:
    _kf = KalmanXYAH()

    def __init__(self, tid, box, score, hist, frame):
        self.id = tid
        self.mean, self.cov = self._kf.initiate(xyxy_to_xyah(box))
        self.hist = hist.copy() if hist is not None else None
        self.state = "tracked"
        self.last_frame = frame
        self.obs: list[tuple[int, list, float]] = [(frame, [float(v) for v in box[:4]], float(score))]
        self.hists: list[np.ndarray] = [hist] if hist is not None else []
        self.coast: list[tuple[int, list]] = []   # predicted boxes after the last observation

    def predict(self, gap):
        m, c = self.mean, self.cov
        if self.state != "tracked":
            m = m.copy(); m[6] = 0; m[7] = 0
        for _ in range(max(1, gap)):
            m, c = self._kf.predict(m, c)
        self.mean, self.cov = m, c

    def box(self):
        return xyah_to_xyxy(self.mean)

    def update(self, box, score, hist, frame):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        if hist is not None:
            self.hist = hist.copy() if self.hist is None else (0.85 * self.hist + 0.15 * hist)
            self.hists.append(hist)
        self.state = "tracked"
        self.last_frame = frame
        self.coast = []
        self.obs.append((frame, [float(v) for v in box[:4]], float(score)))


class BodyTracker:
    def __init__(self, high=0.5, low=0.2, new=0.55, match_iou=0.25, buffer_frames=90, coast_frames=60):
        self.high, self.low, self.new = high, low, new
        self.max_cost = 1.0 - match_iou
        self.buffer = buffer_frames
        self.coast_frames = coast_frames
        self.tracks: list[BodyTrack] = []
        self.finished: list[BodyTrack] = []
        self._next = 1
        self._last = None

    def _cost(self, tracks, boxes, hists, use_app=True):
        tb = np.array([t.box() for t in tracks]).reshape(-1, 4)
        c = 1.0 - iou_matrix(tb, boxes)
        if use_app and hists is not None:
            for i, t in enumerate(tracks):
                for j in range(len(boxes)):
                    if c[i, j] >= 0.95 or t.hist is None or hists[j] is None:
                        continue
                    s = hist_sim(t.hist, hists[j])
                    # appearance refines IoU when boxes overlap (crossing people); a clear
                    # appearance mismatch makes the pair much less attractive
                    c[i, j] = 0.7 * c[i, j] + 0.3 * (1.0 - s)
                    if s < 0.35:
                        c[i, j] += 0.25
        return c

    @staticmethod
    def _assign(cost, max_cost):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        r, c = linear_sum_assignment(cost)
        m, ut, ud = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
        for i, j in zip(r, c):
            if cost[i, j] <= max_cost:
                m.append((i, j)); ut.discard(i); ud.discard(j)
        return m, sorted(ut), sorted(ud)

    def step(self, frame, dets: np.ndarray, hists: list):
        gap = 1 if self._last is None else frame - self._last
        self._last = frame
        for t in self.tracks:
            t.predict(gap)
        sc = dets[:, 4] if len(dets) else np.zeros(0)
        hi = np.where(sc >= self.high)[0]
        lo = np.where((sc >= self.low) & (sc < self.high))[0]
        pool = list(self.tracks)
        m1, ut, ud = self._assign(self._cost(pool, dets[hi, :4], [hists[k] for k in hi]), self.max_cost)
        for i, j in m1:
            k = hi[j]; pool[i].update(dets[k, :4], dets[k, 4], hists[k], frame)
        rem = [pool[i] for i in ut if pool[i].state == "tracked"]
        m2, _, _ = self._assign(self._cost(rem, dets[lo, :4], None, use_app=False), 0.5)
        for i, j in m2:
            k = lo[j]; rem[i].update(dets[k, :4], dets[k, 4], hists[k], frame)
        matched = {id(pool[i]) for i, _ in m1} | {id(rem[i]) for i, _ in m2}
        for t in pool:
            if id(t) not in matched:
                t.state = "lost"
                if frame - t.last_frame <= self.coast_frames:
                    t.coast.append((frame, [float(v) for v in t.box()]))
        for j in ud:
            k = hi[j]
            if dets[k, 4] >= self.new:
                self.tracks.append(BodyTrack(self._next, dets[k, :4], dets[k, 4], hists[k], frame))
                self._next += 1
        alive = []
        for t in self.tracks:
            if t.state == "lost" and frame - t.last_frame > self.buffer:
                self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive

    def finalize(self) -> list[BodyTrack]:
        out = self.finished + self.tracks
        self.finished, self.tracks = [], []
        return sorted(out, key=lambda t: t.id)


# ------------------------------------------------------------------ dense body boxes
def dense_body(body: dict, max_gap: int | None = None, with_coast=True) -> dict[int, tuple[np.ndarray, bool]]:
    """{frame: (box xyxy, is_coast)} for every frame the body is (believed) there."""
    fr = body["frames"]; bx = np.array(body["boxes"], np.float64)
    out = {}
    if not fr:
        return out
    full = np.arange(fr[0], fr[-1] + 1)
    dense = np.stack([np.interp(full, fr, bx[:, k]) for k in range(4)], 1)
    for f, b in zip(full, dense):
        out[int(f)] = (b, False)
    if with_coast:
        cf = body.get("coast_frames") or []
        cb = body.get("coast_boxes") or []
        if cf:
            pf = [fr[-1]] + list(cf); pb = np.array([bx[-1]] + list(cb), np.float64)
            fullc = np.arange(pf[0] + 1, pf[-1] + 1)
            for k, f in enumerate(fullc):
                out[int(f)] = (np.array([np.interp(f, pf, pb[:, j]) for j in range(4)]), True)
    return out


def body_box_at(body: dict, f: int):
    fr = body["frames"]
    if not fr or f < fr[0] or f > fr[-1]:
        return None
    bx = np.array(body["boxes"], np.float64)
    return np.array([np.interp(f, fr, bx[:, k]) for k in range(4)])


# ------------------------------------------------------------------ face <-> body association
def _assoc_cost(face, body):
    fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    fw = face[2] - face[0]
    x1, y1, x2, y2 = body[:4]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if fx < x1 - 0.15 * bw or fx > x2 + 0.15 * bw:
        return 9.0
    if fy < y1 - 1.0 * fw or fy > y1 + 0.45 * bh:
        return 9.0
    if fw > 0.9 * bw:          # face wider than the body: not this body's face
        return 9.0
    dx = abs(fx - (x1 + x2) / 2) / bw
    rel_y = (fy - y1) / bh
    dy = max(0.0, rel_y - 0.12) * 2.0 + max(0.0, -rel_y) * 1.0
    return dx + dy


def associate_faces(face_tracks: list[dict], bodies: list[dict], max_cost=0.75, ambig_margin=0.08):
    """Fills body["face_frames"], body["face_tracks"] and returns a list of ambiguous frames."""
    by_frame: dict[int, list] = {}
    for t in face_tracks:
        for f, b in zip(t["frames"], t["boxes"]):
            by_frame.setdefault(int(f), []).append((t["id"], b))
    for b in bodies:
        b["face_frames"], b["face_tracks"] = [], []
    bmap = {b["id"]: b for b in bodies}
    ambiguous = []
    for f in sorted(by_frame):
        cands = []
        for b in bodies:
            bb = body_box_at(b, f)
            if bb is not None:
                cands.append((b["id"], bb))
        if not cands:
            continue
        faces = by_frame[f]
        C = np.array([[_assoc_cost(fb, bb) for _, bb in cands] for _, fb in faces])
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] > max_cost:
                continue
            row = np.sort(C[i])
            if len(row) > 1 and row[1] - row[0] < ambig_margin:
                ambiguous.append(f)
            b = bmap[cands[j][0]]
            b["face_frames"].append(f); b["face_tracks"].append(faces[i][0])
    return sorted(set(ambiguous))


# ------------------------------------------------------------------ body re-ID
def link_bodies(bodies: list[dict], hists: dict[int, np.ndarray], fps: float, max_gap_s=3.0,
                min_sim=0.6, max_dist=2.5):
    """Link body tracks across short occlusions by torso appearance. Sets body["reid_prev"]
    (id of the earlier track this one continues) and "reid_sim". Cannot-link: tracks that
    coexist in time are never linked; each earlier track continues at most one later track."""
    max_gap = int(round(max_gap_s * fps))
    order = sorted(bodies, key=lambda b: b["frames"][0])
    used = set()
    for b in order:
        b.setdefault("reid_prev", None)
    for b in order:
        s0 = b["frames"][0]
        best, bsim = None, min_sim
        for a in order:
            if a is b or a["id"] in used:
                continue
            e = a["frames"][-1]
            if not (0 < s0 - e <= max_gap):
                continue
            # spatial plausibility: start of b near the end of a (in body widths)
            ab, bb = np.array(a["boxes"][-1]), np.array(b["boxes"][0])
            w = max(ab[2] - ab[0], bb[2] - bb[0], 1.0)
            d = np.linalg.norm((ab[:2] + ab[2:]) / 2 - (bb[:2] + bb[2:]) / 2) / w
            if d > max_dist * (1 + (s0 - e) / fps):
                continue
            s = hist_sim(hists.get(a["id"]), hists.get(b["id"]))
            if s > bsim:
                best, bsim = a, s
        if best is not None:
            b["reid_prev"] = best["id"]; b["reid_sim"] = round(bsim, 3)
            used.add(best["id"])


# ------------------------------------------------------------------ head model
def head_prior(body_box):
    x1, y1, x2, y2 = body_box[:4]
    bw, bh = x2 - x1, y2 - y1
    hw = float(np.clip(0.12 * bh, 0.22 * bw, 0.55 * bw))
    hh = 1.25 * hw
    return np.array([(x1 + x2) / 2, y1 + 0.55 * hh, hw, hh])


def rel_sample(face_box, body_box):
    fx, fy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
    bw, bh = max(body_box[2] - body_box[0], 1.0), max(body_box[3] - body_box[1], 1.0)
    return np.array([(fx - (body_box[0] + body_box[2]) / 2) / bw, (fy - body_box[1]) / bh,
                     (face_box[2] - face_box[0]) / bh, (face_box[3] - face_box[1]) / bh])


def apply_rel(rel, body_box):
    bw, bh = max(body_box[2] - body_box[0], 1.0), max(body_box[3] - body_box[1], 1.0)
    return np.array([(body_box[0] + body_box[2]) / 2 + rel[0] * bw, body_box[1] + rel[1] * bh,
                     rel[2] * bh, rel[3] * bh])


class HeadModel:
    """Per-body head position model: relative (face vs body) samples, nearest in time wins."""

    def __init__(self, body: dict, face_box_of, tracks: set | None = None):
        """tracks: only learn from these face tracks (one person's), None = all."""
        self.samples_f, self.samples_r = [], []
        for f, tid in zip(body.get("face_frames", []), body.get("face_tracks", [])):
            if tracks is not None and tid not in tracks:
                continue
            fb = face_box_of(tid, f)
            bb = body_box_at(body, f)
            if fb is None or bb is None:
                continue
            r = rel_sample(fb, bb)
            if 0.0 < r[2] < 0.6 and abs(r[0]) < 0.8:
                self.samples_f.append(int(f)); self.samples_r.append(r)
        self.samples_f = np.array(self.samples_f, np.int64)
        self.samples_r = np.array(self.samples_r, np.float64).reshape(-1, 4)

    def head(self, body_box, f, fps):
        """(cx, cy, w, h) of the head; grows with distance (in time) from the nearest sample."""
        prior = head_prior(body_box)
        if len(self.samples_f) == 0:
            return prior * np.array([1, 1, 1.15, 1.15])
        d = np.abs(self.samples_f - f)
        idx = np.argsort(d)[:6]
        if d[idx[0]] > 3 * fps:
            return prior * np.array([1, 1, 1.15, 1.15])
        r = np.median(self.samples_r[idx], 0)
        h = apply_rel(r, body_box)
        dt = d[idx[0]] / fps
        grow = 1.0 + 0.2 * min(dt, 2.0)
        h[2:] *= grow
        # never smaller than ~90 % of the generic head prior: covers hair and the back of the head,
        # and protects against a bad sample
        h[2] = max(h[2], 0.9 * prior[2]); h[3] = max(h[3], 0.9 * prior[3])
        # The face-vs-body offset is pose dependent: a close person facing the camera with an arm
        # out has the face far off the box centre, the same person turned away has the head centred
        # (real clip: blur sat beside the back of the head 2 s after the last frontal face). Away from
        # the last face sample the head may be anywhere between the learnt spot and the generic spot
        # (top centre of the body) -> cover the whole span (uncertain => blur), growing with time.
        span = float(np.clip((dt - 0.15) / 0.6, 0.0, 1.0))
        if span > 0:
            g = h.copy()
            g[0] += span * (prior[0] - h[0]); g[1] += span * (prior[1] - h[1])
            x1 = min(h[0] - h[2] / 2, g[0] - g[2] / 2); x2 = max(h[0] + h[2] / 2, g[0] + g[2] / 2)
            y1 = min(h[1] - h[3] / 2, g[1] - g[3] / 2); y2 = max(h[1] + h[3] / 2, g[1] + g[3] / 2)
            h = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        return h


# ------------------------------------------------------------------ orphan face tracks
def adopt_orphan_tracks(proj, min_share=0.7) -> list[tuple[int, int]]:
    """A face track without embeddings (tiny, cut by the frame edge, blurry) gets its own person from
    clustering. If the body that carries it carries one person's faces otherwise (>= min_share of its
    other associations), the track is that person: assign it. Returns [(track, person)] changes.
    Uses the current association (bodies[*].face_tracks)."""
    bodies = proj.data.get("bodies") or []
    tp = {t["id"]: t["person_id"] for t in proj.tracks}
    changes = []
    for t in proj.tracks:
        if t.get("manual"):
            continue
        e, _ = proj.track_embs(t["id"])
        if e is not None and len(e) >= 2:
            continue
        votes: dict = {}
        for b in bodies:
            ft = b.get("face_tracks", [])
            if t["id"] not in ft:
                continue
            others = [tp[x] for x in ft if x != t["id"] and tp.get(x) is not None]
            if not others:
                continue
            for p in set(others):
                share = others.count(p) / len(others)
                if share >= min_share:
                    votes[p] = votes.get(p, 0) + ft.count(t["id"])
        if votes:
            p = max(votes, key=votes.get)
            if p != t["person_id"]:
                changes.append((t["id"], p))
    if changes:
        for tid, p in changes:
            proj.track(tid)["person_id"] = p
            proj.track(tid)["adopted_by_body"] = True
        names = {p["id"]: p["name"] for p in proj.persons}
        proj.rebuild_persons(keep_names=names, renumber=True)
    return changes
