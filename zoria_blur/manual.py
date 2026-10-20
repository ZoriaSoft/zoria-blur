"""Manual fixes: the user draws one box on a frame where a face was missed.

    python -m zoria_blur addbox VIDEO -p PROJECT --person 2 --frame 340 --box 210,220,250,280

The box is followed with optical flow forward and backward until the person's next / previous
detection (or `max_s` seconds, or until the patch is lost). The result is stored as a manual track of
that person (track["manual"] = true, also logged in data["manual"]) and face<->body association is
re-run, so the person-level layer continues from it. Renders blur it like any detection of the person.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .bridge import MAX_LOST, _step
from .video_io import FrameReader, probe


def _follow(grays, box, excludes, stop_at):
    out, lost, cur = [], 0, list(box)
    for k in range(1, len(grays)):
        if k in stop_at:
            break
        nb = _step(grays[k - 1], grays[k], cur, excludes[k - 1])
        if nb is None:
            lost += 1
            if lost > MAX_LOST:
                break
        else:
            cur, lost = nb, 0
        out.append((k, list(cur)))
    return out


def add_box(video: Path, proj, ffmpeg: str, *, person: int, frame: int, box, max_s=1.5):
    pids = {p["id"] for p in proj.persons}
    if person not in pids:
        if person != 0:
            raise ValueError(f"unknown person {person}; known: {sorted(pids)} (use 0 for a new person)")
        person = max(pids | {0}) + 1
    fps = proj.data["video"]["fps_float"]
    n = proj.data["video"]["frame_count"]
    if not 0 <= frame < n:
        raise ValueError(f"frame {frame} out of range 0..{n - 1}")
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must be x1,y1,x2,y2 with x2>x1, y2>y1")
    span = int(round(max_s * fps))
    lo, hi = max(0, frame - span), min(n - 1, frame + span)
    own = sorted({int(f) for t in proj.tracks if t["person_id"] == person for f in t["frames"]})
    nxt = next((f for f in own if f > frame), None)
    prv = next((f for f in reversed(own) if f < frame), None)
    if nxt is not None:
        hi = min(hi, nxt - 1)
    if prv is not None:
        lo = max(lo, prv + 1)
    others: dict[int, list] = {}
    for t in proj.tracks:
        if t["person_id"] != person:
            for f, b in zip(t["frames"], t["boxes"]):
                others.setdefault(int(f), []).append(b)
    grays = {}
    for k, fr in enumerate(FrameReader(probe(Path(video), ffmpeg), ffmpeg, start_frame=lo)):
        i = lo + k                 # input-side seek lands exactly on frame `lo` (tests/test_cli_ui.py)
        if i > hi:
            break
        if i >= lo:
            grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    fw = [grays[i] for i in range(frame, hi + 1)]
    bw = [grays[i] for i in range(frame, lo - 1, -1)]
    exf = [others.get(i, []) for i in range(frame, hi + 1)]
    exb = [others.get(i, []) for i in range(frame, lo - 1, -1)]
    res = {frame: [x1, y1, x2, y2]}
    for k, b in _follow(fw, res[frame], exf, set()):
        res[frame + k] = b
    for k, b in _follow(bw, res[frame], exb, set()):
        res[frame - k] = b
    frames = sorted(res)
    tid = max([t["id"] for t in proj.tracks] + [0]) + 1
    proj.tracks.append({
        "id": tid, "person_id": person, "manual": True,
        "frames": frames, "boxes": [[round(float(v), 1) for v in res[f]] for f in frames],
        "scores": [1.0] * len(frames), "mean_score": 1.0, "crops": [], "confidence": 1.0,
    })
    proj.data.setdefault("manual", []).append({"person": person, "frame": frame, "box": [x1, y1, x2, y2],
                                               "track": tid, "frames": [frames[0], frames[-1]]})
    names = {p["id"]: p["name"] for p in proj.persons}
    proj.rebuild_persons(keep_names=names)
    for t in proj.tracks:          # rebuild_persons zeroes confidence of embedding-less tracks
        if t.get("manual"):
            t["confidence"] = 1.0
    if proj.data.get("bodies"):
        from .bodies import associate_faces
        associate_faces(proj.tracks, proj.data["bodies"])
    proj._log("addbox", {"person": person, "frame": frame, "box": [x1, y1, x2, y2], "track": tid})
    return {"track": tid, "person": person, "frames": [frames[0], frames[-1]], "num_frames": len(frames)}
