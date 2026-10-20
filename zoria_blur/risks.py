"""Risky moments of an analysis, as time ranges a UI timeline can show.

types:
  track_break    a person's face is not detected for a while inside their timeline, and no body that carries
                 them is detected either (with a body seen, the head is blurred from the body: not a guess)
  occlusion      a person's body is lost (coasting on prediction / gap inside its track) while that predicted
                 box is mostly behind another detected body (a lost body with nobody in front of it walked out of
                 the frame or was missed by the detector: covered by the coast / bridge blur, not shown). Not
                 counted: frames where every person the body carried is seen anyway (their face, or another body
                 that carries them: the track just changed id), and bodies seen on fewer than 3 detection frames
                 (one-off boxes; each used to add a 1 s coast range)
  face_merge     a face box looked like two merged faces (split attempted)
  heads_overlap  two people's heads overlap on screen
  ambiguous      a face could belong to two bodies
  low_confidence a face track whose identity match is weak
"""
from __future__ import annotations

import numpy as np

from .bodies import body_box_at, head_prior


def _ranges(frames, tol):
    frames = sorted(set(int(f) for f in frames))
    out = []
    for f in frames:
        if out and f - out[-1][1] <= tol:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def body_persons(body: dict, track_person: dict) -> list[int]:
    ps = []
    for tid in body.get("face_tracks", []):
        p = track_person.get(tid)
        if p is not None and p not in ps:
            ps.append(p)
    return ps


def _covered(box, others, frac=0.3) -> bool:
    """Is at least `frac` of `box` covered by one of the `others` boxes?"""
    a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    for o in others:
        w = min(box[2], o[2]) - max(box[0], o[0])
        h = min(box[3], o[3]) - max(box[1], o[1])
        if w > 0 and h > 0 and w * h / a >= frac:
            return True
    return False


def compute_risks(proj, min_break_s=0.25) -> list[dict]:
    fps = proj.data["video"]["fps_float"]
    every = proj.data["analysis"]["params"].get("every", 2)
    tol = max(every + 1, int(round(0.2 * fps)))
    tp = {t["id"]: t["person_id"] for t in proj.tracks}
    risks = []

    def add(kind, frames, persons, extra=None):
        for a, b in _ranges(frames, tol):
            r = {"type": kind, "start_frame": a, "end_frame": b, "start_s": round(a / fps, 2),
                 "end_s": round((b + 1) / fps, 2), "persons": sorted(set(persons))}
            if extra:
                r.update(extra)
            risks.append(r)

    bodies = proj.data.get("bodies") or []
    # detected body frames per person (a body that carried their face)
    body_frames: dict = {}
    for b in bodies:
        for pid in body_persons(b, tp):
            body_frames.setdefault(pid, set()).update(int(f) for f in b["frames"])
    # face gaps inside a person's timeline
    for p in proj.persons:
        fr = sorted({int(f) for t in proj.tracks if t["person_id"] == p["id"] for f in t["frames"]})
        seen = body_frames.get(p["id"], set())
        for a, b in zip(fr, fr[1:]):
            if b - a > max(2 * every, min_break_s * fps) and b - a <= 10 * fps:
                # body detections every `every` frames: a frame counts as body-covered if one is within every
                gap = [f for f in range(a + 1, b) if not any(f + d in seen for d in range(-every, every + 1))]
                if len(gap) >= max(2 * every, min_break_s * fps):
                    add("track_break", gap, [p["id"]])
    det_boxes: dict = {}
    for b in bodies:
        for f, bx in zip(b["frames"], b["boxes"]):
            det_boxes.setdefault(int(f), []).append((b["id"], bx))

    def hidden(bid, f, box):
        near = [int(f) + d for d in range(-every, every + 1)]
        others = [bx for g in near for (ob, bx) in det_boxes.get(g, []) if ob != bid]
        return _covered(box, others)

    face_frames: dict = {}
    for t in proj.tracks:
        face_frames.setdefault(t["person_id"], set()).update(int(f) for f in t["frames"])
    carried: dict = {}                  # person -> {frame: {body ids detected carrying them}}
    for b in bodies:
        for pid in body_persons(b, tp):
            for f in b["frames"]:
                carried.setdefault(pid, {}).setdefault(int(f), set()).add(b["id"])

    def seen_elsewhere(bid, ps, f):
        if not ps:
            return False
        near = range(int(f) - every, int(f) + every + 1)
        for pid in ps:
            if any(g in face_frames.get(pid, ()) for g in near):
                continue
            if any(carried.get(pid, {}).get(g, set()) - {bid} for g in near):
                continue
            return False
        return True

    for b in bodies:
        if len(b["frames"]) < 3:
            continue
        ps = body_persons(b, tp)
        cand = list(zip(b.get("coast_frames") or [], b.get("coast_boxes") or []))
        # interior gaps of a body track (lost then re-found)
        for a, c in zip(b["frames"], b["frames"][1:]):
            if c - a > max(3 * every, 0.3 * fps):
                cand += [(f, body_box_at(b, f)) for f in range(a + 1, c)]
        occ = [int(f) for f, bx in cand if hidden(b["id"], f, bx) and not seen_elsewhere(b["id"], ps, f)]
        if occ:
            add("occlusion", occ, ps, {"body": b["id"]})
    mg: dict = {}
    for e in proj.data["analysis"].get("merge_events", []):
        f = e["frame"]
        ps = tuple(sorted({tp[t["id"]] for t in proj.tracks if f in t["frames"]}))
        mg.setdefault(ps, []).append(f)
    for ps, frs in mg.items():
        add("face_merge", frs, list(ps))
    amb = proj.data["analysis"].get("ambiguous_assoc_frames") or []
    if amb:
        add("ambiguous", amb, [])
    # heads overlapping (from body boxes)
    if len(bodies) >= 2:
        over: dict = {}
        n = proj.data["video"]["frame_count"]
        for f in range(0, n, max(1, every)):
            hs = []
            for b in bodies:
                bb = body_box_at(b, f)
                if bb is not None:
                    h = head_prior(bb)
                    hs.append((b, h))
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    (b1, h1), (b2, h2) = hs[i], hs[j]
                    if abs(h1[0] - h2[0]) < 0.5 * (h1[2] + h2[2]) * 0.8 and abs(h1[1] - h2[1]) < 0.5 * (h1[3] + h2[3]):
                        key = tuple(sorted(set(body_persons(b1, tp) + body_persons(b2, tp))))
                        over.setdefault(key, []).append(f)
        for key, frs in over.items():
            add("heads_overlap", frs, list(key))
    for t in proj.tracks:
        if t.get("manual"):
            continue
        if t.get("confidence", 1.0) < 0.3 and len(t["frames"]) >= 3:
            add("low_confidence", range(t["frames"][0], t["frames"][-1] + 1), [t["person_id"]],
                {"track": t["id"]})
    risks.sort(key=lambda r: (r["start_frame"], r["type"]))
    return risks
