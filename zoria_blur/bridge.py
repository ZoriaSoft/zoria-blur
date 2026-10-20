"""Bridge detection gaps inside a track by following the face with optical flow.

Linear interpolation across a gap is wrong when a face goes behind someone and comes out
elsewhere, or is partly hidden and missed by the detector (two faces overlapping). Here the face
is followed pixel-wise from the last detection forward and from the next detection backward;
the two paths are blended, and where they disagree both are kept (fail-safe: more blur, never
less). Results go to track["bridge"] = {"frames": [...], "boxes": [...]} and are blurred like
detections.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_LK = dict(winSize=(15, 15), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def _step(prev_gray, gray, box, exclude=()):
    """Move box (x1,y1,x2,y2) from prev_gray to gray. Returns new box or None if lost."""
    H, W = prev_gray.shape
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 4:
        return None
    mx1, my1 = int(max(0, x1 + 0.1 * w)), int(max(0, y1 + 0.1 * h))
    mx2, my2 = int(min(W, x2 - 0.1 * w)), int(min(H, y2 - 0.1 * h))
    if mx2 - mx1 < 3 or my2 - my1 < 3:
        return None
    mask = np.zeros_like(prev_gray)
    mask[my1:my2, mx1:mx2] = 255
    # never pick features on another detected face (prevents latching onto the face in front)
    for e in exclude:
        ex1, ey1, ex2, ey2 = (int(max(0, e[0])), int(max(0, e[1])), int(min(W, e[2])), int(min(H, e[3])))
        if ex2 > ex1 and ey2 > ey1:
            mask[ey1:ey2, ex1:ex2] = 0
    full = (mx2 - mx1) * (my2 - my1)
    if np.count_nonzero(mask) < 0.4 * full:
        return None   # mostly covered by another face: this face is (going) behind it
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.01, minDistance=2, mask=mask)
    if pts is None or len(pts) < 4:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **_LK)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    if good.sum() < 4:
        return None
    p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
    d = np.median(p1 - p0, axis=0)
    # scale from pairwise distance ratios
    i, j = np.triu_indices(len(p0), 1)
    d0 = np.linalg.norm(p0[i] - p0[j], axis=1); d1 = np.linalg.norm(p1[i] - p1[j], axis=1)
    ok = d0 > 2
    s = float(np.clip(np.median(d1[ok] / d0[ok]), 0.9, 1.1)) if ok.any() else 1.0
    cx, cy = (x1 + x2) / 2 + d[0], (y1 + y2) / 2 + d[1]
    nw, nh = w * s, h * s
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


def _follow(grays, start_box, excludes):
    """Returns [(box, lost_frames)]; lost_frames counts consecutive frames the face was not found."""
    out = [(list(start_box), 0)]
    box = list(start_box)
    lost = 0
    for k in range(1, len(grays)):
        nb = _step(grays[k - 1], grays[k], box, excludes[k - 1])
        if nb is not None and lost <= MAX_LOST:
            box = nb; lost = 0
        else:
            lost += 1   # hold briefly; a long hold means the face went behind something
        out.append((list(box), lost))
    return out


MAX_LOST = 6


def bridge_project(proj, video: Path, *, max_gap_s=2.0, min_gap=3, progress=None):
    fps = proj.data["video"]["fps_float"]
    max_gap = int(round(max_gap_s * fps))
    jobs = []
    obs: dict[int, list] = {}
    for t in proj.tracks:
        for f, b in zip(t["frames"], t["boxes"]):
            obs.setdefault(int(f), []).append((t["id"], b))
    for t in proj.tracks:
        fr = t["frames"]
        for a, b in zip(range(len(fr) - 1), range(1, len(fr))):
            g = fr[b] - fr[a]
            if min_gap <= g <= max_gap:
                jobs.append((t, fr[a], fr[b], t["boxes"][a], t["boxes"][b]))
        t["bridge"] = {"frames": [], "boxes": []}
    if not jobs:
        return 0
    cap = cv2.VideoCapture(str(video))
    cache: dict[int, np.ndarray] = {}

    def gray(i):
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            cache[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None
        return cache[i]

    n_boxes = 0
    jobs.sort(key=lambda j: j[1])
    for k, (t, f0, f1, b0, b1) in enumerate(jobs):
        gs = [gray(i) for i in range(f0, f1 + 1)]
        if any(g is None for g in gs):
            continue
        exc = [[b for tid, b in obs.get(i, []) if tid != t["id"]] for i in range(f0, f1 + 1)]
        fwd = _follow(gs, b0, exc)
        bwd = _follow(gs[::-1], b1, exc[::-1])[::-1]
        L = f1 - f0
        for i in range(1, L):
            w = i / L
            (fa, fl), (ba, bl) = fwd[i], bwd[i]
            fa, ba = np.array(fa), np.array(ba)
            f_ok, b_ok = fl <= MAX_LOST, bl <= MAX_LOST
            if f_ok and b_ok:
                fw, bw_ = fa[2] - fa[0], ba[2] - ba[0]
                dc = np.linalg.norm((fa[:2] + fa[2:]) / 2 - (ba[:2] + ba[2:]) / 2)
                boxes = [list((1 - w) * fa + w * ba)] if dc < 0.5 * max(fw, bw_) else [list(fa), list(ba)]
            elif f_ok:
                boxes = [list(fa)]
            elif b_ok:
                boxes = [list(ba)]
            else:
                boxes = []   # hidden from both sides: nothing visible to blur
            for bx in boxes:
                t["bridge"]["frames"].append(f0 + i)
                t["bridge"]["boxes"].append([round(float(v), 1) for v in bx])
                n_boxes += 1
        # keep the cache small (jobs are sorted by start frame)
        for key in [x for x in cache if x < f0]:
            del cache[key]
        if progress and k % 5 == 0:
            progress({"event": "progress", "phase": "bridge", "frame": k, "total": len(jobs)})
    cap.release()
    return n_boxes
