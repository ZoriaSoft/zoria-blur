"""Automatic QA of a rendered video: is any face that should be blurred still sharp?

Two independent checks, on every `every`-th frame:

1. detector pass  - SCRFD at a LOW threshold on the rendered frame. Each face found is compared
   with the ORIGINAL pixels at the same place (mid-band projection, `sharp_ratio`). A face counts as sharp when
   it kept most of its original detail. Faces of kept persons are excluded by their (interpolated)
   boxes, unless the face embedding says it is a person who must be blurred.
     leak kinds: "in_mask"   sharp face inside a planned blur region (blur too weak / too small)
                 "identity"  sharp face that looks like a person who must be blurred
                 "not_kept"  (blur-except) sharp face that is not a confidently kept person
   A sharp face that matches nobody is reported as "unknown_face" (risk, not a leak) in
   blur-selected mode: it is somebody the analysis never saw, not a selected person.
   Every candidate is then re-detected in the ORIGINAL frame, in an upscaled crop around it (face ~80 px at the
   detector input, the best view the detector gets): a leak is a face of the source video that stays sharp.
     score >= src_face (0.5, the analysis' face threshold) -> a source face: the candidate is a leak as above;
     score >= src_min (0.2)                                -> "unconfirmed_face": a risk on the timeline, not a leak
                                                              (a face-like emblem on a glass door scores 0.12-0.29);
     below src_min                                         -> counted in `render_artifacts` (clasped hands next to
                                                              a blurred head, codec texture: no face in the source).
   Real faces of 25-35 px confirm at 0.81-0.87.
2. mask pass      - every real detection of a person who must be blurred (from the analysis) is
   checked directly: if the rendered pixels there are still as sharp as the original -> "mask_miss".

Risky moments (track breaks, occlusions, merges, low confidence) from the analysis are added as
time ranges so a UI timeline can show them next to the leaks.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .backend import default_models_dir
from .detector import SCRFD
from .embedder import ArcFace
from .renderer import dense_sequence, plan_regions
from .risks import _ranges, compute_risks
from .video_io import FrameReader, probe


def _gray(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float32)


def _lap(img):
    return cv2.Laplacian(_gray(img), cv2.CV_32F)


def _lapvar(img):
    return float(_lap(img).var())


DOG_SIGMAS = (0.8, 2.0)   # mid band: features (eyes, nose, mouth edges) of 15-60 px faces, below codec noise
FLAT_DOG_VAR = 1.0        # band energy of the original below this: no detail to lose, cannot judge
MIN_FACE_PX = 12          # narrower faces cannot be judged with a fixed band (see sharp_ratio)
SHARP_THRESH = 0.35       # sharp_ratio >= this: the face kept its detail (a leak where it must be blurred)


def _dog(img, s1=DOG_SIGMAS[0], s2=DOG_SIGMAS[1]):
    g = _gray(img)
    return cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2)


def sharp_ratio(orig, out, box, core=0.7):
    """Share of the ORIGINAL face detail still in `out` inside the core of box: the regression coefficient of the
    rendered mid-band image on the original's, <Do, Dr> / <Do, Do>, with D = difference of Gaussians (sigma 0.8 -
    2.0). 1 = untouched, ~0 = blurred. Returns (ratio or None, band variance of the original).

    Why this band and a projection (measured, README QA step 4):
    * a variance ratio counts detail that is not the face's (codec noise and the blur's own edges): a fully blurred
      dark head read 0.47-0.57, 8x pixelation 1.9-2.3. Uncorrelated detail adds nothing to a projection.
    * the Laplacian (finest band) projection loses the face to the re-encode: an untouched crf-18 re-encode of a
      small / dark / moving face read 0.23-0.52, a kept face in a real render 0.12-0.17 (counted as blurred).
      The mid band survives the codec: untouched re-encode 0.84-0.93, kept faces in real renders 0.57-0.80,
      real blurred faces 0.00-0.03, mild 3x3 blur 0.79-0.83, sigma-1.5 blur 0.45-0.53, sigma-3 0.14-0.24.
    Test set (tests/eval_sharpness.py, 7 clips, faces >= 12 px): real blurred faces max 0.26 (749), kept faces in
    real renders min 0.54, 3x3 blur min 0.62, sigma-1.5 98 % >= SHARP_THRESH.
    Faces narrower than MIN_FACE_PX are not judged (None): their core is < 9 px, where a renderer-blurred 6 px face
    reads up to 0.81 and a 3x3 blur as low as 0.0 (the band is the whole face, not its features)."""
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = box[:4]
    if x2 - x1 < MIN_FACE_PX:
        return None, 0.0
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * core, (y2 - y1) * core
    X1, Y1, X2, Y2 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return None, 0.0
    da, db = _dog(orig[Y1:Y2, X1:X2]), _dog(out[Y1:Y2, X1:X2])
    da = da - da.mean()
    db = db - db.mean()
    va = float((da * da).mean())
    if va < FLAT_DOG_VAR:
        return None, va
    return max(0.0, float((da * db).sum() / (da * da).sum())), va


def _inside(pt, reg, grow=1.0):
    cx, cy, w, h = reg[:4]
    return abs(pt[0] - cx) <= 0.5 * w * grow and abs(pt[1] - cy) <= 0.5 * h * grow


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def source_face_score(det, orig, box) -> float:
    """Best face score in the ORIGINAL frame at `box` (IoU > 0.3 or the box centre inside it), from an upscaled
    crop around the box (det.detect_region: crop = 2x the box, run at 160 px -> the face is ~80 px)."""
    d, _ = det.detect_region(orig, box, thresh=0.1, expand=0.5, det_size=160, nms_thresh=0.4)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best = 0.0
    for x in d:
        if _iou(box, x[:4]) > 0.3 or (x[0] <= cx <= x[2] and x[1] <= cy <= x[3]):
            best = max(best, float(x[4]))
    return best


def run_qa(rendered: Path, proj, ffmpeg: str, *, mode=None, persons=None, original: Path | None = None,
           every=3, det_thresh=0.3, sharp_thresh=SHARP_THRESH, id_sim=0.35, src_face=0.5, src_min=0.2, backend="cpu", threads=0,
           models_dir=None, plan_kw=None, progress=None) -> dict:
    t0 = time.perf_counter()
    if sharp_thresh is None:
        sharp_thresh = SHARP_THRESH
    lr = proj.data.get("last_render") or {}
    mode = mode or lr.get("mode")
    persons = persons if persons is not None else lr.get("persons")
    if plan_kw is None:
        plan_kw = lr.get("plan_kw") or {}
    if not mode or persons is None:
        raise ValueError("qa needs --mode/--persons (or a project that was rendered before)")
    original = Path(original or proj.data["video"]["path"])
    if not original.is_absolute() and not original.is_file():
        original = Path.cwd() / original
    fps = proj.data["video"]["fps_float"]
    n = proj.data["video"]["frame_count"]
    sel = set(persons)
    per_frame, rep = plan_regions(proj, mode, list(persons), **plan_kw)
    rep.pop("_protect", None)
    kept_tracks = set(rep["kept_tracks"]) if mode == "blur-except" else \
        {t["id"] for t in proj.tracks if t["person_id"] not in sel}
    blur_pids = sel if mode == "blur-selected" else {p["id"] for p in proj.persons if p["id"] not in sel}
    # kept face boxes per frame (short interpolation only: a kept face is excluded where it was seen)
    kept_boxes: dict[int, list] = {}
    for t in proj.tracks:
        if t["id"] in kept_tracks:
            d = dense_sequence(t["frames"], t["boxes"], fps, n, max_gap_s=0.3, smooth_s=0.0, extend_s=0.05)
            for f, r in d.items():
                cx, cy, w, h = r
                kept_boxes.setdefault(f, []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    # real detections of faces that must be blurred (mask pass)
    must = {}
    for t in proj.tracks:
        if t["id"] in kept_tracks or t.get("manual"):
            continue
        if mode == "blur-selected" and t["person_id"] not in sel:
            continue
        for f, b in zip(t["frames"], t["boxes"]):
            must.setdefault(int(f), []).append((t["person_id"], b))
    kept_dets = {}
    for t in proj.tracks:
        if t["id"] in kept_tracks and not t.get("manual"):
            for f, b in zip(t["frames"], t["boxes"]):
                kept_dets.setdefault(int(f), []).append((t["person_id"], b))
    kept_checked, kept_blurred = 0, []
    cents_blur = {p: proj.emb.get(f"p{p}") for p in blur_pids if proj.emb.get(f"p{p}") is not None}
    cents_keep = {p["id"]: proj.emb.get(f"p{p['id']}") for p in proj.persons
                  if p["id"] not in blur_pids and proj.emb.get(f"p{p['id']}") is not None}

    md = Path(models_dir) if models_dir else default_models_dir()
    # a video analyzed in small-face mode is checked at a higher detector scale too (tiny faces the
    # 640 pass cannot see would otherwise never show up as leaks)
    small = bool(proj.data.get("analysis", {}).get("detection", {}).get("small_mode"))
    det = SCRFD(md / "det_10g.onnx", backend, 960 if small else 640, threads)
    emb = ArcFace(md / "w600k_mbf.onnx", backend, threads)
    info_o, info_r = probe(original, ffmpeg), probe(Path(rendered), ffmpeg)
    leaks, unknown, n_checked, n_faces, n_kept_excl, n_mask_checked = [], [], 0, 0, 0, 0
    n_artifacts, unconfirmed, n_small = 0, [], 0
    ro = iter(FrameReader(info_o, ffmpeg))
    for i, out in enumerate(FrameReader(info_r, ffmpeg)):
        try:
            orig = next(ro)
        except StopIteration:
            break
        check_mask = i in must
        if i in kept_dets:     # how often is a kept face blurred (collateral of the fail-safe)?
            for pid, b in kept_dets[i]:
                r, _ = sharp_ratio(orig, out, b)
                if r is not None:
                    kept_checked += 1
                    if r < sharp_thresh:
                        kept_blurred.append(i)
        if i % every and not check_mask:
            continue
        if check_mask:
            for pid, b in must[i]:
                r, va = sharp_ratio(orig, out, b)
                if r is None:
                    n_small += b[2] - b[0] < MIN_FACE_PX
                    continue
                n_mask_checked += 1
                if r >= sharp_thresh:
                    leaks.append({"frame": i, "kind": "mask_miss", "person": pid,
                                  "box": [round(float(v)) for v in b], "sharp": round(r, 3)})
        if i % every:
            continue
        n_checked += 1
        d, k = det.detect(out, det_thresh)
        if not len(d):
            continue
        E = emb.embed(out, list(k))
        for j in range(len(d)):
            b = d[j, :4]
            n_faces += 1
            r, _ = sharp_ratio(orig, out, b)
            if r is None:
                n_small += b[2] - b[0] < MIN_FACE_PX
                continue
            if r < sharp_thresh:
                continue                     # blurred (or cannot judge): fine
            sb = max([(float(E[j] @ c), p) for p, c in cents_blur.items()] or [(-1.0, None)])
            sk = max([float(E[j] @ c) for c in cents_keep.values()] or [-1.0])
            looks_blur = sb[0] >= id_sim and sb[0] > sk
            is_kept_box = any(_iou(b, kb) > 0.3 for kb in kept_boxes.get(i, []))
            # on a kept person's tracked box, only a clear identity match overrides the box
            # (profile / back-of-head embeddings are weak and confuse similar views)
            strong_blur = sb[0] >= 0.45 and sb[0] >= sk + 0.1
            if is_kept_box and not strong_blur:
                n_kept_excl += 1
                continue
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            src = source_face_score(det, orig, b)
            if src < src_face:
                if src >= src_min:
                    unconfirmed.append({"frame": i, "box": [round(float(v)) for v in b], "src_score": round(src, 2)})
                else:
                    n_artifacts += 1         # no face there in the source video: not a leak
                continue
            in_mask = any(_inside(c, reg) for reg in per_frame.get(i, []))
            item = {"frame": i, "box": [round(float(v)) for v in b], "score": round(float(d[j, 4]), 2),
                    "sharp": round(r, 3), "sim_blur": round(sb[0], 3), "sim_keep": round(sk, 3),
                    "person": sb[1] if looks_blur else None}
            if in_mask and not is_kept_box:
                leaks.append(item | {"kind": "in_mask"})
            elif is_kept_box:
                leaks.append(item | {"kind": "identity"})
            elif looks_blur:
                leaks.append(item | {"kind": "identity"})
            elif mode == "blur-except":
                leaks.append(item | {"kind": "not_kept"})
            elif sk >= id_sim:
                n_kept_excl += 1             # a kept person somewhere we had no box for
            else:
                unknown.append(item)
        if progress and n_checked % 10 == 0:
            progress({"event": "progress", "phase": "qa", "frame": i, "total": n})
    tol = max(every + 1, int(round(0.2 * fps)))
    leak_ranges = [{"start_frame": a, "end_frame": b, "start_s": round(a / fps, 2), "end_s": round((b + 1) / fps, 2),
                    "count": sum(1 for x in leaks if a <= x["frame"] <= b)}
                   for a, b in _ranges([x["frame"] for x in leaks], tol)]
    risks = compute_risks(proj)
    for a, b in _ranges([x["frame"] for x in unconfirmed], tol):
        risks.append({"type": "unconfirmed_face", "start_frame": a, "end_frame": b, "start_s": round(a / fps, 2),
                      "end_s": round((b + 1) / fps, 2), "persons": []})
    for a, b in _ranges([x["frame"] for x in unknown], tol):
        risks.append({"type": "unknown_face", "start_frame": a, "end_frame": b, "start_s": round(a / fps, 2),
                      "end_s": round((b + 1) / fps, 2), "persons": []})
    for r in risks:
        r["relevant"] = (not r["persons"]) or bool(set(r["persons"]) & blur_pids)
    res = {
        "rendered": str(rendered), "mode": mode, "persons": list(persons),
        "frames_checked": n_checked, "faces_detected": n_faces, "kept_excluded": n_kept_excl,
        "mask_checks": n_mask_checked, "render_artifacts": n_artifacts,
        "faces_too_small": int(n_small), "sharp_thresh": sharp_thresh,
        "unconfirmed_faces": len(unconfirmed), "leak_count": len(leaks), "leak_frames": len({x["frame"] for x in leaks}),
        "leak_ranges": leak_ranges, "leaks": leaks[:200], "unknown_faces": len(unknown),
        "kept_face_checks": kept_checked, "kept_face_blurred": len(kept_blurred),
        "kept_blurred_ranges": [{"start_frame": a, "end_frame": b, "start_s": round(a / fps, 2),
                                 "end_s": round((b + 1) / fps, 2)} for a, b in _ranges(kept_blurred, tol)],
        "risky": risks, "seconds": round(time.perf_counter() - t0, 2),
        "passed": len(leaks) == 0,
    }
    if progress:
        for lr_ in leak_ranges:
            progress({"event": "qa_issue", "kind": "leak", **lr_})
        for r in risks:
            if r["relevant"]:
                progress({"event": "qa_issue", "kind": "risk", **r})
    return res
