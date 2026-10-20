"""Phase 1: detect + track + embed + cluster -> project directory."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .backend import default_models_dir
from .clustering import cluster_tracks, track_descriptor
from .detector import SCRFD
from .embedder import ArcFace, face_quality
from .project import Project, SCHEMA_VERSION
from .tracker import FaceTracker, iou_matrix
from .bodies import BodyTracker, torso_hist, associate_faces, link_bodies
from .person_detector import PERSON_MODELS, YOLOXPerson
from .video_io import FrameReader, probe

DET_MODELS = {"det_10g": "det_10g.onnx", "det_2.5g": "det_2.5g.onnx", "det_500m": "det_500m.onnx"}
EMB_MODELS = {"w600k_mbf": "w600k_mbf.onnx"}


class _SubsampledReader(FrameReader):
    """Only decodes-to-pipe every Nth frame (ffmpeg select filter after fps): far less IO."""

    def __init__(self, info, ffmpeg, every):
        import subprocess
        from .video_io import _NOWIN
        self.info, self.w, self.h = info, info.width, info.height
        vf = f"fps={info.fps}" + (f",select='not(mod(n\\,{every}))'" if every > 1 else "")
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", info.path, "-map", "0:v:0", "-an", "-sn",
               "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     bufsize=self.w * self.h * 3 * 2, **_NOWIN)
        self.nbytes = self.w * self.h * 3


def _crop_for_thumb(img, box, size=160):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(w, h) * 1.5
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    X2, Y2 = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
    c = img[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    sc = size / max(c.shape[:2])
    return cv2.resize(c, (max(1, int(c.shape[1] * sc)), max(1, int(c.shape[0] * sc))), interpolation=cv2.INTER_AREA)


def _split_identity_switches(obs, embs, frames_e, min_side=3, drop=0.15):
    """Split a track where its appearance changes abruptly (ID switch after people cross).
    Returns list of index ranges into obs."""
    if len(embs) < 2 * min_side:
        return [(0, len(obs))]
    best_k, best_s = None, 1.0
    for k in range(min_side, len(embs) - min_side + 1):
        a = embs[max(0, k - 6):k].mean(0); b = embs[k:k + 6].mean(0)
        s = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if s < best_s:
            best_s, best_k = s, k
    if best_k is None or best_s >= drop:
        return [(0, len(obs))]
    split_frame = frames_e[best_k]
    cut = next(i for i, o in enumerate(obs) if o[0] >= split_frame)
    return [(0, cut), (cut, len(obs))]


def _drop_nested(d, k, inside=0.7):
    """The looser NMS of the re-detection also keeps PART of a face as a second box (real clip: a big
    close-up face came back as the whole face + its eyes/sunglasses band; the band became a separate
    person and blur-except blurred the kept face with it). Two real faces never have one box mostly
    inside the other (the hidden one is occluded, not visible through the front face), so a box with
    >= `inside` of its area inside a larger box is the same face: drop it - but only when the larger
    box is about as confident (a big weak false positive must not swallow a real small face)."""
    if len(d) < 2:
        return d, k
    area = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    keep = np.ones(len(d), bool)
    for a in range(len(d)):
        for b in range(len(d)):
            if a == b or not keep[b] or area[b] <= area[a] or d[b, 4] < d[a, 4] - 0.15:
                continue
            iw = min(d[a, 2], d[b, 2]) - max(d[a, 0], d[b, 0])
            ih = min(d[a, 3], d[b, 3]) - max(d[a, 1], d[b, 1])
            if iw > 0 and ih > 0 and iw * ih >= inside * max(area[a], 1e-6):
                keep[a] = False
                break
    return d[keep], k[keep]


def _split_merged_faces(frame, det, dets, kps, bodies, tracker, low_thresh, events, idx):
    """Find face boxes that probably hold two faces (abnormally wide, sudden size jump vs. the
    track they continue, or sitting where two people's heads overlap) and re-detect inside an
    upscaled crop with a looser NMS. Replaces a merged box by the faces recovered from it."""
    if len(dets) == 0:
        return dets, kps
    trig = []
    tboxes = [(t, t.box()) for t in tracker.tracks if t.state == "tracked" and t.obs]
    for i in range(len(dets)):
        x1, y1, x2, y2, sc = dets[i]
        w, h = x2 - x1, y2 - y1
        reason = None
        if w < 20:
            continue    # too small to hold two separable faces (and crowds of tiny faces made this 10x slower)
        if w / max(h, 1.0) > 0.95:
            reason = "wide"
        if reason is None and tboxes:
            ious = iou_matrix(np.array([b for _, b in tboxes]), dets[i:i + 1, :4])[:, 0]
            j = int(np.argmax(ious))
            if ious[j] > 0.2:
                ws = [o[1][2] - o[1][0] for o in tboxes[j][0].obs[-6:]]
                if w > 1.45 * float(np.median(ws)):
                    reason = "size_jump"
        if reason is None and len(bodies) >= 2:
            fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
            n_in = 0
            for b in bodies:
                bw, bh = b[2] - b[0], b[3] - b[1]
                if b[0] - 0.1 * bw <= fx <= b[2] + 0.1 * bw and b[1] - h <= fy <= b[1] + 0.3 * bh:
                    n_in += 1
            if n_in >= 2:
                reason = "heads_overlap"
        if reason:
            trig.append((i, reason))
    if not trig:
        return dets, kps
    new_d, new_k, drop = [], [], set()
    for i, reason in trig:
        rd, rk = det.detect_region(frame, dets[i], thresh=low_thresh)
        rd, rk = _drop_nested(rd, rk)
        found = 0
        for j in range(len(rd)):
            # keep only faces that are not already in the full-frame result
            others = np.delete(dets[:, :4], i, 0)
            if len(others) and iou_matrix(others, rd[j:j + 1, :4]).max() > 0.4:
                continue
            if new_d and iou_matrix(np.array([d[:4] for d in new_d]), rd[j:j + 1, :4]).max() > 0.4:
                continue
            new_d.append(rd[j]); new_k.append(rk[j]); found += 1
        if found >= 1 and len(rd) >= 2:
            drop.add(i)                       # replaced by the recovered faces
            if found >= 2:                    # one box became two (or more): a real split
                events.append({"frame": idx, "reason": reason, "recovered": found})
        elif found >= 1:
            # nothing split; we found the same face again -> drop the duplicate re-detection
            new_d = new_d[:-found]; new_k = new_k[:-found]
            if reason != "heads_overlap":
                events.append({"frame": idx, "reason": reason, "recovered": 0})
    keep = [i for i in range(len(dets)) if i not in drop]
    if not new_d:
        return dets[keep], kps[keep]
    return (np.concatenate([dets[keep], np.array(new_d, np.float32)]),
            np.concatenate([kps[keep], np.array(new_k, np.float32)]))


def analyze(video: Path, out_dir: Path, ffmpeg: str, *, backend="cpu", threads=0, every=2,
            det_model="det_10g", det_size=640, det_thresh=0.5, low_thresh=0.25, weak_thresh=0.3,
            min_emb_face=28, cluster_thresh=0.40, models_dir: Path | None = None,
            person_model="yolox_tiny", person_every=1, person_thresh=0.3, split_faces=True,
            det_mode="balanced", progress=None) -> Project:
    t0 = time.perf_counter()
    models_dir = Path(models_dir) if models_dir else default_models_dir()
    info = probe(Path(video), ffmpeg)
    out_dir = Path(out_dir)
    if out_dir.exists():
        for sub in ("crops", "thumbs"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    det = SCRFD(models_dir / DET_MODELS[det_model], backend, det_size, threads)
    emb_model = ArcFace(models_dir / EMB_MODELS["w600k_mbf"], backend, threads)
    tracker = FaceTracker(high_thresh=det_thresh, low_thresh=low_thresh, new_thresh=det_thresh + 0.05,
                          buffer_frames=int(round(info.fps_float * 2.0)))

    use_bodies = bool(person_model) and person_model != "none"
    if use_bodies:
        pdet = YOLOXPerson(models_dir / PERSON_MODELS[person_model], backend, threads)
        btracker = BodyTracker(buffer_frames=int(round(info.fps_float * 1.5)),
                               coast_frames=int(round(info.fps_float * 1.0)))
    from .smallfaces import SmallFaceDetector
    sfd = SmallFaceDetector(det, pdet if use_bodies else None, det_mode, info.width, info.height,
                            full_size=det_size, person_thresh=person_thresh)
    body_hists: dict[int, list] = {}
    merge_events: list = []
    weak_raw: list = []
    t_person = t_split = 0.0
    det_embs: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    crop_heaps: dict[int, dict] = {}   # track id -> {second bucket: (quality, frame, jpeg)}
    total_est = max(1, info.frame_count)
    n_det_frames, t_det, t_emb = 0, 0.0, 0.0
    last_frame = -1
    reader = _SubsampledReader(info, ffmpeg, every)
    for k, frame in enumerate(reader):
        idx = k * every
        last_frame = idx
        cur_bodies = []
        if use_bodies:
            tp = time.perf_counter()
            if k % max(1, person_every) == 0:
                pd = sfd.bodies(frame, k)
                hs = [torso_hist(frame, b) for b in pd]
                btracker.step(idx, pd, hs)
                cur_bodies = [b[:4] for b in pd]
            else:
                cur_bodies = [t.box() for t in btracker.tracks if t.state == "tracked"]
            t_person += time.perf_counter() - tp
        ta = time.perf_counter()
        dets, kps = sfd.faces(frame, low_thresh, k, np.array(cur_bodies) if len(cur_bodies) else None)
        t_det += time.perf_counter() - ta
        if split_faces:
            ts_ = time.perf_counter()
            dets, kps = _split_merged_faces(frame, det, dets, kps, cur_bodies, tracker, low_thresh,
                                            merge_events, idx)
            t_split += time.perf_counter() - ts_
        dets, kps = _drop_nested(dets, kps)
        tb = time.perf_counter()
        embs: list = [None] * len(dets)
        sel = [i for i in range(len(dets))
               if dets[i, 4] >= det_thresh and (dets[i, 2] - dets[i, 0]) >= min_emb_face]
        if sel:
            feats = emb_model.embed(frame, [kps[i] for i in sel])
            for j, i in enumerate(sel):
                q = face_quality(dets[i], kps[i]) * float(dets[i, 4])
                embs[i] = feats[j]
                det_embs[(idx, i)] = (feats[j], q)
        t_emb += time.perf_counter() - tb
        tracker.step(idx, dets, embs)
        # weak faces (score below what starts a track): kept so blur-except can blur them (fail-safe)
        for i in range(len(dets)):
            if weak_thresh <= dets[i, 4] < det_thresh + 0.05 and dets[i, 2] - dets[i, 0] >= 5:
                weak_raw.append((idx, [round(float(v), 1) for v in dets[i, :4]], round(float(dets[i, 4]), 3)))
        # thumbnail candidates: best crop per ~1 s bucket per track (keeps time diversity)
        bucket = int(idx // max(1, int(round(info.fps_float))))
        for t in tracker.tracks:
            if t.obs and t.obs[-1][0] == idx:
                key = (idx, t.obs[-1][3])
                if key in det_embs:
                    q = det_embs[key][1]
                    b = crop_heaps.setdefault(t.id, {})
                    if bucket not in b or q > b[bucket][0]:
                        c = _crop_for_thumb(frame, t.obs[-1][1])
                        if c is not None:
                            ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            b[bucket] = (q, idx, buf.tobytes())
        n_det_frames += 1
        if progress and k % 10 == 0:
            progress({"event": "progress", "phase": "analyze", "frame": idx, "total": total_est})
    # the subsampled reader only sees every Nth frame: up to every-1 frames follow the last one
    frame_count = last_frame + every
    if last_frame < info.frame_count <= last_frame + every:
        frame_count = info.frame_count
    raw_tracks = tracker.finalize()
    t_loop = time.perf_counter() - t0

    # ---- build track records (+ identity-switch split) ----
    tracks, emb_store = [], {}
    next_id = 1
    for t in raw_tracks:
        obs = t.obs
        e_list = [(o[0], det_embs[(o[0], o[3])]) for o in obs if (o[0], o[3]) in det_embs]
        E = np.stack([x[1][0] for x in e_list]) if e_list else np.zeros((0, 512), np.float32)
        Fe = [x[0] for x in e_list]
        parts = _split_identity_switches(obs, E, Fe) if len(E) else [(0, len(obs))]
        for a, b in parts:
            sub = obs[a:b]
            if not sub:
                continue
            fr = {o[0] for o in sub}
            sel = [i for i, f in enumerate(Fe) if f in fr]
            tid = next_id; next_id += 1
            if sel:
                emb_store[f"t{tid}"] = E[sel].astype(np.float16).astype(np.float32)
                emb_store[f"t{tid}_q"] = np.array([e_list[i][1][1] for i in sel], np.float32)
                emb_store[f"t{tid}_f"] = np.array([Fe[i] for i in sel], np.int32)
            crops = []
            for q, f, jpg in sorted(crop_heaps.get(t.id, {}).values(), key=lambda x: -x[0])[:24]:
                if f in fr:
                    name = f"track_{tid:04d}_f{f:06d}.jpg"
                    (out_dir / "crops" / name).write_bytes(jpg)
                    crops.append({"frame": f, "quality": round(q, 4), "file": name})
            tracks.append({
                "id": tid,
                "person_id": 0,
                "frames": [o[0] for o in sub],
                "boxes": [[round(v, 1) for v in o[1]] for o in sub],
                "scores": [round(o[2], 3) for o in sub],
                "mean_score": round(float(np.mean([o[2] for o in sub])), 3),
                "crops": crops,
            })

    # weak detections that no track took (tiny / far / dark faces scoring 0.3-0.55): blur-except
    # blurs them, since "everyone except X" must not leave an unknown face sharp just because the
    # detector was unsure (real street clip: dozens of 10-25 px faces stayed sharp).
    tb_by_f: dict = {}
    for t in tracks:
        for f, b in zip(t["frames"], t["boxes"]):
            tb_by_f.setdefault(int(f), []).append(b)
    weak_faces = []
    for f, b, sc in weak_raw:
        tb = tb_by_f.get(int(f))
        if tb and iou_matrix(np.array(tb, np.float64), np.array([b], np.float64)).max() > 0.3:
            continue
        weak_faces.append([int(f)] + b + [sc])

    data = {
        "schema_version": SCHEMA_VERSION,
        "engine": f"zoria-blur {__version__}",
        "video": info.to_dict() | {"frame_count": frame_count},
        "analysis": {
            "params": {"every": every, "det_model": det_model, "det_size": det_size,
                       "det_thresh": det_thresh, "low_thresh": low_thresh, "min_emb_face": min_emb_face,
                       "weak_thresh": weak_thresh,
                       "cluster_thresh": cluster_thresh, "embedder": "w600k_mbf", "backend": backend},
        },
        "persons": [],
        "tracks": tracks,
        "weak_faces": weak_faces,
        "edits": [],
    }
    # ---- person (body) tracks ----
    bodies = []
    if use_bodies:
        W, H = info.width, info.height
        for bt in btracker.finalize():
            if len(bt.obs) < 3 and max(o[2] for o in bt.obs) < 0.6:
                continue                      # flicker
            bid = len(bodies) + 1
            cf = [c[0] for c in bt.coast if c[0] > bt.obs[-1][0]]
            cb = [[round(float(np.clip(v, 0, W if i % 2 == 0 else H)), 1) for i, v in enumerate(c[1])]
                  for c in bt.coast if c[0] > bt.obs[-1][0]]
            bodies.append({"id": bid, "frames": [o[0] for o in bt.obs],
                           "boxes": [[round(v, 1) for v in o[1]] for o in bt.obs],
                           "scores": [round(o[2], 3) for o in bt.obs],
                           "coast_frames": cf, "coast_boxes": cb})
            if bt.hists:
                hm = np.mean(np.stack(bt.hists), 0)
                emb_store[f"b{bid}_h"] = (hm / max(hm.sum(), 1e-9)).astype(np.float32)
    data["bodies"] = bodies
    data["analysis"]["params"].update({"person_model": person_model if use_bodies else None,
                                       "person_every": person_every, "person_thresh": person_thresh,
                                       "split_faces": split_faces, "det_mode": det_mode})
    data["analysis"]["detection"] = sfd.info()
    data["analysis"]["merge_events"] = merge_events
    proj = Project(out_dir, data, emb_store)
    proj.recluster(cluster_thresh)
    data["edits"] = []
    if bodies:
        amb = associate_faces(proj.tracks, bodies)
        link_bodies(bodies, {b["id"]: emb_store.get(f"b{b['id']}_h") for b in bodies}, info.fps_float)
        data["analysis"]["ambiguous_assoc_frames"] = amb
        from .bodies import adopt_orphan_tracks
        data["analysis"]["adopted_tracks"] = adopt_orphan_tracks(proj)
    from .bridge import bridge_project
    tb = time.perf_counter()
    data["analysis"]["bridge_boxes"] = bridge_project(proj, Path(video), progress=progress)
    t_bridge = time.perf_counter() - tb
    elapsed = time.perf_counter() - t0
    data["analysis"]["timing"] = {
        "seconds_total": round(elapsed, 2),
        "seconds_loop": round(t_loop, 2),
        "seconds_detector": round(t_det, 2),
        "seconds_embedder": round(t_emb, 2),
        "seconds_bridge": round(t_bridge, 2),
        "seconds_person_detector": round(t_person, 2),
        "seconds_face_split": round(t_split, 2),
        "person_tracks": len(bodies),
        "video_frames": frame_count,
        "detection_frames": n_det_frames,
        "video_fps_processed": round(frame_count / elapsed, 2) if elapsed else None,
    }
    from .risks import compute_risks
    data["analysis"]["risks"] = compute_risks(proj)
    proj.save()
    if progress:
        for r in data["analysis"]["risks"]:
            progress({"event": "risk", **r})
        progress({"event": "done", "phase": "analyze", "project": str(out_dir / "project.json")})
    return proj
