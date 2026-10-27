"""Phase 2: build smoothed per-frame face regions from the project and burn in the blur."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from .numeric import gaussian_filter1d_nearest

from .project import Project
from .video_io import FrameReader, FrameWriter, pick_encoder, probe


# ---------------------------------------------------------------- box sequences
def _segments(frames, boxes, max_gap, max_jump):
    """Split an observation sequence where the gap is too long or the face jumps (scene cut)."""
    segs, cur = [], [0]
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        b0, b1 = boxes[i - 1], boxes[i]
        w = max(b0[2] - b0[0], b1[2] - b1[0], 1.0)
        c0 = np.array([(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2])
        c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
        jump = np.linalg.norm(c1 - c0) / w
        if gap > max_gap or (gap > 1 and jump > max_jump * gap ** 0.5):
            segs.append(cur); cur = [i]
        else:
            cur.append(i)
    segs.append(cur)
    return segs


def dense_sequence(frames, boxes, fps, n_frames, *, max_gap_s=1.0, smooth_s=0.12, extend_s=0.3,
                   max_jump=1.5):
    """Observations at detection frames -> {frame: (cx, cy, w, h)} for every covered frame.

    * interpolates linearly across gaps <= max_gap_s (short detection misses),
    * zero-phase Gaussian smoothing (offline, so no lag) removes jitter,
    * extrapolates `extend_s` before the first / after the last observation (fail-safe).
    """
    out: dict[int, np.ndarray] = {}
    if not frames:
        return out
    order = np.argsort(frames)
    frames = [int(frames[i]) for i in order]
    boxes = [boxes[i] for i in order]
    # dedupe frames (person-level merges can share a frame within the overlap tolerance)
    fr_u, bx_u = [], []
    for f, b in zip(frames, boxes):
        if fr_u and fr_u[-1] == f:
            p = bx_u[-1]; bx_u[-1] = [min(p[0], b[0]), min(p[1], b[1]), max(p[2], b[2]), max(p[3], b[3])]
        else:
            fr_u.append(f); bx_u.append(list(b))
    frames, boxes = fr_u, bx_u
    max_gap = max(2, int(round(max_gap_s * fps)))
    ext = int(round(extend_s * fps))
    sigma = max(0.5, smooth_s * fps)
    for seg in _segments(frames, boxes, max_gap, max_jump):
        f = np.array([frames[i] for i in seg])
        b = np.array([boxes[i] for i in seg], dtype=np.float64)
        cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                           b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
        full_f = np.arange(f[0], f[-1] + 1)
        dense = np.stack([np.interp(full_f, f, cxcywh[:, k]) for k in range(4)], axis=1)
        if len(full_f) > 3:
            sm = gaussian_filter1d_nearest(dense, sigma, axis=0)
            # never let smoothing shrink the box below the raw size (keeps faces covered)
            sm[:, 2:] = np.maximum(sm[:, 2:], dense[:, 2:] * 0.97)
            dense = sm
        for i, fi in enumerate(full_f):
            out[int(fi)] = dense[i]
        # extrapolate ends with damped velocity, growing the box slightly (uncertainty)
        k = min(len(full_f) - 1, max(1, int(fps * 0.2)))
        v_end = (dense[-1] - dense[-1 - k]) / k if k > 0 else np.zeros(4)
        v_start = (dense[k] - dense[0]) / k if k > 0 else np.zeros(4)
        for s in range(1, ext + 1):
            damp = 0.85 ** s
            grow = 1.0 + 0.02 * s
            fe, fs = int(full_f[-1] + s), int(full_f[0] - s)
            if fe < n_frames and fe not in out:
                e = dense[-1].copy(); e[:2] += v_end[:2] * s * damp; e[2:] *= grow
                out[fe] = e
            if fs >= 0 and fs not in out:
                e = dense[0].copy(); e[:2] -= v_start[:2] * s * damp; e[2:] *= grow
                out[fs] = e
    return out


# ---------------------------------------------------------------- blur kernels
def _shrink(img, sw, sh):
    """Area-average down to (sw, sh). Big factors halve with pyrDown first (8x faster than one INTER_AREA step on a
    large head; at most 2 grey levels apart), then the last step is INTER_AREA."""
    while img.shape[1] // 2 >= sw * 2 and img.shape[0] // 2 >= sh * 2:
        img = cv2.pyrDown(img)
    return cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)


def _blur_roi(roi, face_w, style, strength):
    h, w = roi.shape[:2]
    strength = max(0.1, float(strength))
    if style == "pixelate":
        blocks = max(3.0, 9.0 / strength)                # blocks across the face width
        sc = blocks / max(face_w, 1.0)
        sw, sh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        small = _shrink(roi, sw, sh)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # gaussian: downscale -> blur -> 4x up -> blur -> full size. Strong, cheap, no blockiness:
    # all real filtering happens on small images, the full-res step is a bilinear resize.
    target = max(4.0, 11.0 / strength)                   # px across face in the small image
    sc = min(1.0, target / max(face_w, 1.0))
    sw, sh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    small = _shrink(roi, sw, sh)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    if sc < 0.25:
        mid = cv2.resize(small, (sw * 4, sh * 4), interpolation=cv2.INTER_LINEAR)
        mid = cv2.GaussianBlur(mid, (0, 0), 3.0)
        return cv2.resize(mid, (w, h), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(up, (0, 0), max(1.0, 0.6 / max(sc, 1e-3)))


def _soft_ellipse(shape, center, axes, sigma):
    """Filled anti-aliased ellipse blurred with `sigma` (the feathered blur mask). A big face has a big feather: a
    full-size Gaussian with sigma 40-120 px cost 15-90 ms per face (2.9 fps on a 2560x1080 clip with ~19 regions),
    so a wide sigma is computed on a grid `ds` times coarser (sigma there >= 4 px) and resized back. Measured against
    the full-size mask: wherever the full-size alpha is above 0.5 the coarse one is never lower (it only grows the
    feather by up to ~0.06 at its outer edge), so it never blurs less of a face. 85 ms -> 0.7 ms for a 500 px face."""
    h, w = shape
    ds = int(sigma // 4.0)
    if ds <= 1:
        m = np.zeros(shape, np.float32)
        cv2.ellipse(m, (int(round(center[0])), int(round(center[1]))), (int(round(axes[0])), int(round(axes[1]))),
                    0, 0, 360, 1.0, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(m, (0, 0), sigma)
    sh, sw = -(-h // ds), -(-w // ds)
    m = np.zeros((sh, sw), np.float32)
    s = 16                                          # 4 fractional bits: sub-pixel centre / axes on the coarse grid
    cx, cy = (center[0] + 0.5) / ds - 0.5, (center[1] + 0.5) / ds - 0.5
    cv2.ellipse(m, (int(round(cx * s)), int(round(cy * s))), (int(round(axes[0] / ds * s)), int(round(axes[1] / ds * s))),
                0, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(m, (0, 0), sigma / ds)
    return cv2.resize(m, (sw * ds, sh * ds), interpolation=cv2.INTER_LINEAR)[:h, :w]


def apply_regions(frame, regions, style="gaussian", strength=1.0, padding=0.3, feather=0.25, keep=None):
    """regions: iterable of (cx, cy, w, h[, carve]). Feathered ellipse slightly larger than the face/head.
    keep: optional list of x1,y1,x2,y2[,strong] face boxes that must stay visible; the padded blur mask of
    a neighbouring face is carved out over them.
    carve (5th region value): True / missing = carve every keep box, but never into the region's own raw
    box; False = never carve (fail-safe paths); "k" = an uncertain predicted head: carve only the
    strong keep boxes (confidently identified kept faces), including inside its own box - the head is a
    guess, the kept face an identity-checked observation."""
    H, W = frame.shape[:2]
    keep = keep or []
    for reg in regions:
        cx, cy, w, h = reg[:4]
        flag = reg[4] if len(reg) >= 5 else True
        guess = isinstance(flag, str)
        carve = bool(flag)
        kl = [kb for kb in keep if len(kb) >= 5 and kb[4]] if guess else keep
        if w < 2 or h < 2:
            continue
        ax = 0.5 * w * (1.0 + padding)
        ay = 0.5 * h * (1.0 + padding) * 1.12          # a bit taller: forehead + chin
        cy2 = cy - 0.06 * h                             # shift up toward hair line
        f = feather * max(ax, ay)
        x1, y1 = int(max(0, cx - ax - 2 * f)), int(max(0, cy2 - ay - 2 * f))
        x2, y2 = int(min(W, cx + ax + 2 * f + 1)), int(min(H, cy2 + ay + 2 * f + 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        # solid core grown by half the feather so alpha is ~1 over the whole padded face
        mask = _soft_ellipse(roi.shape[:2], (round(cx - x1), round(cy2 - y1)),
                             (round(ax + f * 0.5), round(ay + f * 0.5)), max(0.8, f / 2.0))
        if kl and carve:
            km = np.zeros_like(mask)
            for kb in kl:
                kcx, kcy = (kb[0] + kb[2]) / 2 - x1, (kb[1] + kb[3]) / 2 - y1
                kax, kay = 0.5 * (kb[2] - kb[0]) * 1.05, 0.5 * (kb[3] - kb[1]) * 1.1
                if kcx + kax < 0 or kcy + kay < 0 or kcx - kax > mask.shape[1] or kcy - kay > mask.shape[0]:
                    continue
                cv2.ellipse(km, (int(round(kcx)), int(round(kcy))), (int(round(kax)), int(round(kay))),
                            0, 0, 360, 1.0, -1, cv2.LINE_AA)
            if km.any():
                km = cv2.GaussianBlur(km, (0, 0), max(0.8, f / 3.0))
                # never carve into the selected face's own raw box
                own = np.zeros_like(mask)
                if not guess:
                    cv2.ellipse(own, (int(round(cx - x1)), int(round(cy - y1))),
                                (int(round(0.5 * w)), int(round(0.5 * h))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
                mask = mask * (1.0 - km * (1.0 - own))
        blurred = _blur_roi(roi, w, style, strength)
        # roi * (1 - mask) + blurred * mask in one native pass (8x faster than the float32 numpy expression; the
        # same result within 1 grey level of rounding)
        frame[y1:y2, x1:x2] = cv2.blendLinear(roi, blurred, 1.0 - mask, mask)
    return frame


# ---------------------------------------------------------------- planning
STRONG_KEEP_SIM = 0.30   # blur-except: a kept detection this similar to the person is "confirmed"
GROUP_PERSONS = 3        # a body box that carried faces of >= this many persons is a group box
def plan_regions(proj: Project, mode: str, persons: list[int], *, keep_track_conf=0.30,
                 keep_det_sim=0.18, min_keep_embs=2, max_gap_s=1.0, smooth_s=0.12, extend_s=0.3):
    """Returns ({frame: [(cx,cy,w,h), ...]}, report dict)."""
    fps = proj.data["video"]["fps_float"]
    n = proj.data["video"]["frame_count"]
    sel = set(persons)
    known = {p["id"] for p in proj.persons}
    missing = sel - known
    if missing:
        raise ValueError(f"unknown person id(s): {sorted(missing)}; known: {sorted(known)}")
    groups: dict[str, tuple[list, list]] = {}
    report = {"blurred_tracks": [], "kept_tracks": [], "failsafe_tracks": [], "failsafe_detections": 0}

    def add(key, f, b):
        g = groups.setdefault(key, ([], []))
        g[0].append(f); g[1].append(b)

    on_kept_body = (lambda f, box: False)
    if mode == "blur-except" and proj.data.get("bodies"):
        pre_kept = set()
        for t in proj.tracks:
            if t["person_id"] in sel and not t.get("manual") and t.get("confidence", 0.0) >= keep_track_conf:
                e, _ = proj.track_embs(t["id"])
                if e is not None and len(e) >= min_keep_embs:
                    pre_kept.add(t["id"])
        on_kept_body = _kept_body_test(proj, pre_kept, fps)

    for t in proj.tracks:
        pid = t["person_id"]
        if mode == "blur-selected":
            if pid in sel:
                report["blurred_tracks"].append(t["id"])
                for f, b in zip(t["frames"], t["boxes"]):
                    add(f"p{pid}", f, b)
        elif mode == "blur-except":
            if pid in sel and t.get("manual"):
                continue      # a user-drawn box of a kept person: nothing to blur
            if pid not in sel:
                report["blurred_tracks"].append(t["id"])
                # unknown/other people: group per person so short gaps between their tracks are bridged
                for f, b in zip(t["frames"], t["boxes"]):
                    add(f"p{pid}", f, b)
                continue
            e, _ = proj.track_embs(t["id"])
            n_e = 0 if e is None else len(e)
            if t.get("confidence", 0.0) < keep_track_conf or n_e < min_keep_embs:
                # FAIL-SAFE: kept person's track not confidently this person -> blur it
                report["failsafe_tracks"].append(t["id"])
                for f, b in zip(t["frames"], t["boxes"]):
                    add(f"t{t['id']}", f, b)
                continue
            report["kept_tracks"].append(t["id"])
            # FAIL-SAFE per detection: an embedded detection that does not look like the person
            sims = t.get("det_sims") or [None] * len(t["frames"])
            for f, b, s in zip(t["frames"], t["boxes"], sims):
                if s is not None and s < keep_det_sim:
                    if on_kept_body(int(f), b):
                        # a profile / turned-away view of the kept person on their own body
                        report["lowsim_kept_on_body"] = report.get("lowsim_kept_on_body", 0) + 1
                        continue
                    report["failsafe_detections"] += 1
                    add(f"x{t['id']}_{f}", f, b)
        else:
            raise ValueError("mode must be blur-selected or blur-except")

    # Observed faces of people who must stay visible (blur-selected: anyone not selected).
    # A synthesized box (gap fill / extrapolation) must never slide over such a face:
    # the real detection owns that face in that frame.
    # blur-except: the confidently kept faces (kept tracks, minus single detections that did not look
    # like the person). Without this a neighbour's padded ellipse covered the kept face whenever two
    # faces were close (real group selfie: kept face blurred in 100 % of frames). Carving never touches
    # the blurred face's own raw box, and uncertain regions (gap paths, unseen heads) are never carved.
    protect: dict[int, list] = {}
    if mode == "blur-selected":
        for t in proj.tracks:
            if t["person_id"] not in sel:
                for f, b in zip(t["frames"], t["boxes"]):
                    protect.setdefault(int(f), []).append(b)
    else:
        # strong = identity confirmed (det sim >= STRONG_KEEP_SIM, or an unembeddable detection next to
        # a confirmed one on the same track): uncertain predicted heads of other people are carved there.
        kept_ids = set(report["kept_tracks"])
        every = proj.data["analysis"]["params"].get("every", 2)
        for t in proj.tracks:
            if t["id"] in kept_ids and not t.get("manual"):
                sims = t.get("det_sims") or [None] * len(t["frames"])
                conf_f = np.array([int(f) for f, s in zip(t["frames"], sims) if s is not None and s >= STRONG_KEEP_SIM],
                                  np.int64)
                for f, b, s in zip(t["frames"], t["boxes"], sims):
                    if s is None or s >= keep_det_sim:
                        strong = (s is not None and s >= STRONG_KEEP_SIM) or (
                            s is None and len(conf_f) > 0 and int(np.abs(conf_f - int(f)).min()) <= 2 * every)
                        protect.setdefault(int(f), []).append(list(b[:4]) + [bool(strong)])
    report["dropped_synth_boxes"] = 0
    report["_protect"] = protect

    per_frame: dict[int, list] = {}
    for key, (fs, bs) in groups.items():
        single = key.startswith("x")
        dense = dense_sequence(fs, bs, fps, n, max_gap_s=max_gap_s, smooth_s=smooth_s,
                               extend_s=(extend_s if not single else 0.25))
        if single:  # isolated suspicious detection: cover the whole detection interval around it
            every = proj.data["analysis"]["params"]["every"]
            f0 = fs[0]; b = dense.get(f0)
            if b is not None:
                for ff in range(max(0, f0 - every - 2), min(n, f0 + every + 3)):
                    dense.setdefault(ff, b)
        observed = set(int(x) for x in fs)
        lo, hi = min(observed), max(observed)
        for f, r in dense.items():
            if f not in observed and lo < f < hi:
                # Gap INSIDE a blurred person's timeline: the face went behind someone (often the
                # very person who must stay visible). Fail-safe: blur the path uncarved, even if
                # that briefly blurs the visible face too (user's call, 2026-10-14).
                if f in protect and _touches(r, protect[f]):
                    report["failsafe_overlap_boxes"] = report.get("failsafe_overlap_boxes", 0) + 1
                per_frame.setdefault(f, []).append(tuple(float(v) for v in r) + (False,))
                continue
            if f not in observed and f in protect and _covers_other(r, protect[f]):
                report["dropped_synth_boxes"] += 1
                continue
            per_frame.setdefault(f, []).append(tuple(float(v) for v in r))
    # Optical-flow bridged boxes of blurred tracks: the face was there but the detector missed it
    # (occluded / overlapping another face). Blur as-is and never carve them (fail-safe).
    blurred = set(report["blurred_tracks"]) | set(report["failsafe_tracks"])
    report["bridged_boxes_used"] = 0
    if proj.data.get("bodies"):
        _plan_body_heads(proj, mode, sel, report, per_frame, set(report["kept_tracks"]))
    if mode == "blur-except" and proj.data.get("weak_faces"):
        _plan_weak_faces(proj, per_frame, set(report["kept_tracks"]), report, on_kept_body)
    for t in proj.tracks:
        if t["id"] not in blurred:
            continue
        br = t.get("bridge") or {}
        for f, b in zip(br.get("frames", []), br.get("boxes", [])):
            per_frame.setdefault(int(f), []).append(
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], False))
            report["bridged_boxes_used"] += 1
    return per_frame, report


def _plan_body_heads(proj, mode, sel, report, per_frame, kept_tracks, link_gap_s=2.0,
                     strong_frac=0.2, strong_min=6):
    """Person-level fail-safe: wherever the body of a person who must be blurred is visible, blur
    that person's head derived from the body box - whether or not the face was detected (hidden,
    turned away, merged with another face; and the face box alone misses hair / back of the head).

    Candidate persons of a body at frame f (uncertain => blur all of them):
      * the persons whose faces it carried nearest before and after f (an ID switch can sit between),
      * every person it carried a lot ("strong": >= 20 % of its faces): a face of someone else
        wrongly associated for a while must not unblur the real owner,
      * no faces at all: the persons of the body it continues / is continued by (torso re-ID),
        else "unknown".
    A candidate's head is placed from that person's own face-vs-body samples on this body.
      blur-selected: blur candidates that are selected persons.
      blur-except:   blur candidates that are not confidently kept, and unknown bodies.
    blur-selected: an unseen head is never carved (a selected person's head over a kept face blurs both).
    blur-except: an unseen head is a guess about someone else; it is carved out over a kept face whose
    identity is confirmed (strong protect box), never over an unconfirmed one (real street clip: guessed
    heads of people behind the kept passer-by blurred him in 42 % of frames).
    Group boxes (blur-except): a body box that carried faces of >= GROUP_PERSONS persons, or holds >= 2
    other body boxes at that frame, is several people in one box; its per-person head offsets drift with
    the box shape. An unseen head predicted from it is dropped where an individual body inside the group
    has its head at that spot (that body predicts the head itself: unknown -> blurred, the kept person's
    own body -> visible); elsewhere it stays (fail-safe for people seen only as part of the group)."""
    from collections import Counter
    from .bodies import HeadModel, dense_body, head_prior
    fps = proj.data["video"]["fps_float"]
    every = proj.data["analysis"]["params"].get("every", 2)
    tp = {t["id"]: t["person_id"] for t in proj.tracks}
    fbox = {t["id"]: dict(zip((int(f) for f in t["frames"]), t["boxes"])) for t in proj.tracks}
    bodies = proj.data["bodies"]
    bmap = {b["id"]: b for b in bodies}
    nxt = {b["reid_prev"]: b["id"] for b in bodies if b.get("reid_prev")}
    UNKNOWN = -1

    def face_box_of(tid, f):
        return fbox.get(tid, {}).get(int(f))

    def person_blurred(p, tids):
        if mode == "blur-selected":
            return p in sel
        if p == UNKNOWN or p not in sel:
            return True
        return not all(t in kept_tracks for t in tids)     # kept person, but a weak track

    def linked_persons(b):
        out, seen, cur = [], set(), b
        for forward in (False, True):
            cur, seen = b, set()
            while cur is not None and cur["id"] not in seen:
                seen.add(cur["id"])
                cur = bmap.get(nxt.get(cur["id"])) if forward else bmap.get(cur.get("reid_prev"))
                if cur is not None and cur.get("face_tracks"):
                    t = cur["face_tracks"][0] if forward else cur["face_tracks"][-1]
                    out.append((tp.get(t), [t]))
                    break
        return out

    span_margin = int(round(link_gap_s * fps))
    report["body_head_boxes"] = 0
    report["body_gap_boxes"] = 0
    report["body_heads_skipped_seen_elsewhere"] = 0
    models: dict = {}
    # frames where each person's face is detected, and on which body (None = on no body)
    on_body: dict = {}
    for bb in bodies:
        for f, t in zip(bb.get("face_frames", []), bb.get("face_tracks", [])):
            on_body[(int(f), t)] = bb["id"]
    det_of: dict = {}
    for t in proj.tracks:
        if t.get("manual"):
            continue
        for f in t["frames"]:
            det_of.setdefault(t["person_id"], {})[int(f)] = on_body.get((int(f), t["id"]))

    def seen_elsewhere(p, f, body_id):
        """p's face is detected within `every` frames of f, and not on this body: the face track
        (and the body that does carry it) cover p there. A head predicted from THIS body is then an
        old/ambiguous association (two people in one body box, e.g. one behind the other) and would
        only blur a neighbour (real clip: a kept face under a stale head of the man behind her)."""
        d = det_of.get(p)
        if not d:
            return False
        for g in range(f - every, f + every + 1):
            if g in d:
                return d[g] != body_id
        return False

    # per frame: the body boxes present (for group-box tests)
    at_frame: dict = {}
    dense_all = {}
    for bb in bodies:
        dense_all[bb["id"]] = dense_body(bb)
        for f, (box, _c) in dense_all[bb["id"]].items():
            at_frame.setdefault(f, []).append((bb["id"], box))
    n_persons = {bb["id"]: len({tp.get(t) for t in bb.get("face_tracks", [])}) for bb in bodies}
    report["group_heads_dropped"] = 0

    def inside(a, g, frac=0.7):
        ix = max(0.0, min(a[2], g[2]) - max(a[0], g[0])); iy = max(0.0, min(a[3], g[3]) - max(a[1], g[1]))
        return ix * iy >= frac * max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)

    def members(b, box, f):
        """individual (non-group) bodies inside group box `box` at frame f"""
        return [ob for oid, ob in at_frame.get(f, []) if oid != b["id"] and n_persons.get(oid, 0) < GROUP_PERSONS
                and inside(ob, box) and (ob[2] - ob[0]) * (ob[3] - ob[1]) < 0.8 * (box[2] - box[0]) * (box[3] - box[1])]

    def explained(h, mem):
        """an individual body's head zone holds the centre of guessed head h"""
        for ob in mem:
            hp = head_prior(ob)
            if abs(h[0] - hp[0]) < 0.5 * max(hp[2], ob[2] - ob[0]) and abs(h[1] - hp[1]) < 1.0 * hp[3]:
                return True
        return False

    def model(b, p, tids):
        key = (b["id"], p)
        if key not in models:
            models[key] = HeadModel(b, face_box_of, set(tids) if p != UNKNOWN else set())
        return models[key]

    for b in bodies:
        ff = np.array(b.get("face_frames", []), np.int64)
        ft = b.get("face_tracks", [])
        fp = [tp.get(t) for t in ft]
        tids_of: dict = {}
        for t, p in zip(ft, fp):
            tids_of.setdefault(p, set()).add(t)
        cnt = Counter(fp)
        strong = {p for p, c in cnt.items() if c >= max(strong_min, strong_frac * len(fp))}
        # per person: frames where that person's face was seen on this body
        seen_p = {p: np.array([f for f, q in zip(ff, fp) if q == p], np.int64) for p in tids_of}
        static = []
        if not ft:
            static = linked_persons(b) or [(UNKNOWN, [])]
        dense = dense_all[b["id"]]
        for f in sorted(dense):
            box, coast = dense[f]
            mem = None
            if len(ff):
                k = int(np.searchsorted(ff, f))
                # a strong person counts only near the span where their face rode this body: a long body
                # track can switch owner (real clip: the kept man's body track later carried a selected
                # person's face 4 s on, and his head was blurred from frame 0)
                cands = {p for p in strong if seen_p[p][0] - span_margin <= f <= seen_p[p][-1] + span_margin}
                if k > 0:
                    cands.add(fp[k - 1])
                if k < len(ff):
                    cands.add(fp[k])
                cl = [(p, tids_of[p]) for p in cands]
            else:
                cl = static
            for p, tids in cl:
                if p is None or not person_blurred(p, tids):
                    continue
                # also when the face IS detected: the face box misses hair / the back of the head.
                # Face seen here -> position is certain, kept faces next to it are carved out;
                # face not seen -> uncertain, never carved (fail-safe).
                sp = seen_p.get(p)
                seen = False
                if sp is not None and len(sp) and not coast:
                    j = int(np.searchsorted(sp, f))
                    seen = (j < len(sp) and sp[j] - f <= every) or (j > 0 and f - sp[j - 1] <= every)
                if not seen and p != UNKNOWN and seen_elsewhere(p, f, b["id"]):
                    report["body_heads_skipped_seen_elsewhere"] += 1
                    continue
                h = model(b, p, tids).head(box, f, fps)
                flag = bool(seen)
                if not seen and mode == "blur-except":
                    flag = "k"
                    if mem is None:
                        mem = members(b, box, f)
                    if (n_persons[b["id"]] >= GROUP_PERSONS or len(mem) >= 2) and explained(h, mem):
                        report["group_heads_dropped"] += 1
                        continue
                per_frame.setdefault(f, []).append(tuple(float(v) for v in h) + (flag,))
                report["body_head_boxes"] += 1

    # gaps between body tracks of the same person (lost -> new track), or linked by torso re-ID
    pairs = {}
    for b in bodies:
        if b.get("reid_prev"):
            pairs[(b["reid_prev"], b["id"])] = None
    by_person: dict = {}
    for b in bodies:
        for tid in set(b.get("face_tracks", [])):
            by_person.setdefault(tp.get(tid), set()).add(b["id"])
    for pid, bids in by_person.items():
        bl = sorted((bmap[i] for i in bids), key=lambda x: x["frames"][0])
        for a, c in zip(bl, bl[1:]):
            pairs[(a["id"], c["id"])] = pid
    max_gap = int(round(link_gap_s * fps))
    for (ia, ic), pid in pairs.items():
        a, c = bmap[ia], bmap[ic]
        ea, sc_ = a["frames"][-1], c["frames"][0]
        if not (1 < sc_ - ea <= max_gap):
            continue
        if pid is not None:
            persons = [pid]
        else:
            persons = list({tp.get(t) for t in (a.get("face_tracks", [])[-1:] + c.get("face_tracks", [])[:1])})
            if not persons and mode == "blur-except":
                persons = [UNKNOWN]
        for p in persons:
            ta = [t for t in a.get("face_tracks", []) if tp.get(t) == p]
            tc = [t for t in c.get("face_tracks", []) if tp.get(t) == p]
            if not person_blurred(p, set(ta) | set(tc)):
                continue
            ba = dense_body(a, with_coast=False).get(ea)
            bc = dense_body(c, with_coast=False).get(sc_)
            if ba is None or bc is None:
                continue
            h0 = model(a, p, ta).head(ba[0], ea, fps)
            h1 = model(c, p, tc).head(bc[0], sc_, fps)
            for f in range(ea + 1, sc_):
                w = (f - ea) / (sc_ - ea)
                h = (1 - w) * h0 + w * h1
                h[2:] = np.maximum(h0[2:], h1[2:]) * 1.1
                per_frame.setdefault(f, []).append(tuple(float(v) for v in h) + ("k" if mode == "blur-except" else False,))
                report["body_gap_boxes"] += 1


def _kept_body_test(proj, kept_ids, fps, near_s=1.0):
    """blur-except: f(frame, face_box) -> True when the face sits in the head zone of a body that
    carries ONLY confidently kept faces and carried one within `near_s` of that frame - and no other
    body claims that head zone better. Such a face is the kept person turned away / in profile (their
    embedding is weak), not a stranger. Bodies that ever carried anyone else never qualify."""
    from .bodies import _assoc_cost, dense_body
    near = int(round(near_s * fps))
    info = []
    for b in proj.data.get("bodies") or []:
        ft = b.get("face_tracks", [])
        ff = np.array(b.get("face_frames", []), np.int64)
        kept_only = bool(ft) and all(t in kept_ids for t in ft)
        info.append((b, kept_only, ff, dense_body(b, with_coast=False)))

    def test(f, box):
        best = None
        for b, kept_only, ff, dense in info:
            bb = dense.get(f)
            if bb is None:
                continue
            c = _assoc_cost(box, bb[0])
            if c < 9.0 and (best is None or c < best[0]):
                best = (c, kept_only, ff)
        if best is None or not best[1]:
            return False
        ff = best[2]
        k = int(np.searchsorted(ff, f))
        return (k < len(ff) and ff[k] - f <= near) or (k > 0 and f - ff[k - 1] <= near)
    return test


def _plan_weak_faces(proj, per_frame, kept_tracks, report, on_kept_body=None, link_s=0.5, extend_s=0.25):
    """blur-except fail-safe: faces the detector saw only weakly (score 0.3-0.55, mostly tiny / far /
    dark faces) never started a track, so nothing blurred them. Chain them over time (IoU / nearby
    centre, gaps <= link_s) and blur each chain, except where it sits on a kept person's face."""
    fps = proj.data["video"]["fps_float"]
    n = proj.data["video"]["frame_count"]
    kept: dict[int, list] = {}
    for t in proj.tracks:
        if t["id"] in kept_tracks:
            d = dense_sequence(t["frames"], t["boxes"], fps, n, max_gap_s=0.3, smooth_s=0.0, extend_s=0.1)
            for f, r in d.items():
                kept.setdefault(f, []).append(r)
    link = max(2, int(round(link_s * fps)))
    chains: list = []                       # [frames, boxes]
    for w in sorted(proj.data["weak_faces"], key=lambda x: x[0]):
        f, b = int(w[0]), w[1:5]
        bw = b[2] - b[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best, bd = None, None
        for c in chains:
            if not (0 < f - c[0][-1] <= link):
                continue
            pb = c[1][-1]
            d = np.hypot(cx - (pb[0] + pb[2]) / 2, cy - (pb[1] + pb[3]) / 2) / max(bw, pb[2] - pb[0], 1.0)
            if d < 1.0 and (bd is None or d < bd):
                best, bd = c, d
        if best is None:
            chains.append([[f], [b]])
        else:
            best[0].append(f); best[1].append(b)
    report["weak_face_chains"] = len(chains)
    report["weak_face_boxes"] = 0
    report["weak_faces_on_kept_body"] = 0
    for fs, bs in chains:
        if on_kept_body is not None and all(on_kept_body(f, b) for f, b in zip(fs, bs)):
            report["weak_faces_on_kept_body"] += 1     # the kept person seen weakly (turned away)
            continue
        dense = dense_sequence(fs, bs, fps, n, max_gap_s=link_s, smooth_s=0.04, extend_s=extend_s)
        for f, r in dense.items():
            cx, cy = r[0], r[1]
            if any(abs(cx - k[0]) < 0.6 * k[2] and abs(cy - k[1]) < 0.6 * k[3] for k in kept.get(f, [])):
                continue                    # a kept face seen weakly: it stays visible
            per_frame.setdefault(f, []).append(tuple(float(v) for v in r))
            report["weak_face_boxes"] += 1


def _touches(r, others):
    cx, cy, w, h = r
    for b in others:
        if cx - w / 2 < b[2] and cx + w / 2 > b[0] and cy - h / 2 < b[3] and cy + h / 2 > b[1]:
            return True
    return False


def _covers_other(r, others, dist_frac=0.5):
    """True when a synthesized box sits on top of a visible face (centers nearly coincide).
    Partial overlaps are handled by carving the visible face out of the mask instead, so a
    hidden-then-reappearing face next to a kept face still gets blurred."""
    cx, cy, w, h = r
    for b in others:
        bw, bh = b[2] - b[0], b[3] - b[1]
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if abs(cx - bx) < dist_frac * bw and abs(cy - by) < dist_frac * bh:
            return True
    return False


def render(video: Path, proj: Project, out_path: Path, ffmpeg: str, *, mode: str, persons: list[int],
           style="gaussian", strength=1.0, padding=0.3, crf=18, preset="veryfast", progress=None,
           qa=True, qa_every=3, backend="cpu", threads=0, models_dir=None, encoder: str | None = None,
           **plan_kw) -> dict:
    t0 = time.perf_counter()
    info = probe(Path(video), ffmpeg)
    pv = proj.data["video"]
    if (info.width, info.height) != (pv["width"], pv["height"]) or abs(info.fps_float - pv["fps_float"]) > 0.01:
        raise ValueError("video does not match the analyzed project (size/fps differ)")
    per_frame, report = plan_regions(proj, mode, persons, **plan_kw)
    protect = report.pop("_protect", {})
    total = pv["frame_count"]
    log = Path(out_path).with_suffix(".ffmpeg.log")
    enc = encoder or pick_encoder(ffmpeg)
    fallback = None

    def encode_all(enc_name: str):
        writer = FrameWriter(info, out_path, ffmpeg, crf=crf, preset=preset, log_path=log, encoder=enc_name)
        n, t_blur = 0, 0.0
        broken = False
        try:
            for idx, frame in enumerate(FrameReader(info, ffmpeg)):
                regs = per_frame.get(idx)
                if regs:
                    ta = time.perf_counter()
                    frame = apply_regions(frame.copy(), regs, style, strength, padding, keep=protect.get(idx))
                    t_blur += time.perf_counter() - ta
                try:
                    writer.write(frame)
                except (BrokenPipeError, OSError):
                    broken = True          # the encoder quit: its exit code below says why
                    break
                n += 1
                if progress and idx % 15 == 0:
                    progress({"event": "progress", "phase": "render", "frame": idx, "total": total})
        finally:
            rc = writer.close()
        return (rc if rc != 0 or not broken else 1), n, t_blur

    rc, n, t_blur = encode_all(enc)
    if rc != 0 and enc != "x264":
        # the graphics card's encoder failed on the real video (driver, size limit): redo the whole file with x264,
        # never ship a half-written file
        fallback = {"from": enc, "exit": rc}
        enc = "x264"
        if progress:
            progress({"event": "progress", "phase": "render", "frame": 0, "total": total})
        rc, n, t_blur = encode_all(enc)
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (exit {rc}); see {log}")
    if log.is_file() and log.stat().st_size == 0:
        log.unlink()
    el = time.perf_counter() - t0
    res = {"output": str(out_path), "frames": n, "seconds": round(el, 2), "fps": round(n / el, 2) if el else None,
           "seconds_blur": round(t_blur, 2), "encoder": enc, "mode": mode, "persons": persons, **report}
    if fallback:
        res["encoder_fallback"] = fallback
    proj.data["last_render"] = {"output": str(out_path), "mode": mode, "persons": list(persons),
                                "plan_kw": dict(plan_kw)}
    proj.save()
    if qa:
        from .qa import run_qa
        res["qa"] = run_qa(out_path, proj, ffmpeg, mode=mode, persons=persons, original=Path(video),
                           every=qa_every, backend=backend, threads=threads, models_dir=models_dir,
                           plan_kw=plan_kw, progress=progress)
    if progress:
        progress({"event": "done", "phase": "render", **res})
    return res


def preview_frame(video: Path, proj: Project, frame_idx: int, ffmpeg: str, *, mode, persons, style="gaussian",
                  strength=1.0, padding=0.3, **plan_kw):
    """Returns (original, blurred) BGR images for one frame (for the UI preview).

    Seeks close to the frame (decoding from the start costs seconds on long videos) and checks the
    landing point against a sequential decode only when the seek is not trusted (VFR sources)."""
    info = probe(Path(video), ffmpeg)
    if frame_idx < 0:
        raise IndexError("frame must be >= 0")
    per_frame, rep = plan_regions(proj, mode, persons, **plan_kw) if persons or mode == "blur-except" else ({}, {})
    protect = rep.get("_protect", {})
    orig = None
    # seek a little before the frame and walk forward: input-side -ss lands on the first frame at/after t
    start = max(0, frame_idx - 3)
    for k, frame in enumerate(FrameReader(info, ffmpeg, start_frame=start)):
        if start + k == frame_idx:
            orig = frame.copy()
            break
    if orig is None:
        for idx, frame in enumerate(FrameReader(info, ffmpeg)):
            if idx == frame_idx:
                orig = frame.copy()
                break
    if orig is None:
        raise IndexError(f"frame {frame_idx} beyond end of video")
    return orig, apply_regions(orig.copy(), per_frame.get(frame_idx, []), style, strength, padding,
                               keep=protect.get(frame_idx))


def preview_plan(proj: Project, mode: str, persons: list[int], *, padding=0.6, feather=0.25, **plan_kw) -> dict:
    """Blur regions for the app's live (playing) preview, from the same plan_regions() the renderer uses.

    Each region is the ellipse apply_regions() blurs, grown to the full feathered extent (radius + 2 feather
    + 2 px; the mask's Gaussian falloff, reflected at frame edges, reaches that far): [cx, cy, rx, ry] in video
    pixels, cy already shifted. tests/eval_preview_parity.py checks that no exported blur pixel falls outside. Fail-safe
    simplifications: keep-face carving is NOT applied (the preview blurs a little more around kept faces than
    the export, never less). blur-selected with nobody selected = no regions (as preview_frame)."""
    if mode == "blur-selected" and not persons:
        per_frame = {}
    else:
        per_frame, _ = plan_regions(proj, mode, persons, **plan_kw)
    frames = {}
    for f, regs in per_frame.items():
        out = []
        for reg in regs:
            cx, cy, w, h = (float(v) for v in reg[:4])
            if w < 2 or h < 2:
                continue
            ax = 0.5 * w * (1.0 + padding)
            ay = 0.5 * h * (1.0 + padding) * 1.12
            fe = feather * max(ax, ay)
            out.append([round(cx, 1), round(cy - 0.06 * h, 1), round(ax + 2 * fe + 2, 1), round(ay + 2 * fe + 2, 1)])
        if out:
            frames[str(int(f))] = out
    v = proj.data["video"]
    return {"fps": v["fps_float"], "frame_count": v["frame_count"], "width": v["width"], "height": v["height"],
            "mode": mode, "persons": list(persons), "frames": frames}
