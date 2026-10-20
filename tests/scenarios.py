"""Scenario tests on synthetic clips built from REAL people cut out of a local test clip (not included).

Each scenario composes two people (A = must be blurred, B = must stay visible) on a textured
background, moves them, then runs the full pipeline: analyze -> render blur-selected A -> QA.
Pass = QA reports no leak AND a ground-truth check (we know where A's face is in every frame)
finds no frame where A's visible face kept its detail.

    PYTHONPATH=. python tests/scenarios.py                 # all scenarios
    PYTHONPATH=. python tests/scenarios.py behind merge    # some
    options: --src VIDEO (required: a clip of your own)  --out DIR (default work/scenarios)

Scenarios: behind (A passes behind B's head), leave_return (A leaves the frame and comes back),
fast (A crosses the frame fast), small_far (A is small/far), merge (A's and B's faces overlap).
Runtime on 2 CPU cores: ~30-45 s per scenario.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zoria_blur.analyze import analyze  # noqa: E402
from zoria_blur.qa import SHARP_THRESH, run_qa, sharp_ratio  # noqa: E402
from zoria_blur.renderer import render  # noqa: E402
from zoria_blur.video_io import find_ffmpeg  # noqa: E402

W, H, FPS = 640, 480, 30
# Frame 180 of your test clip (adjust the boxes to it): (body box, face box) of two different men, frontal.
SRC_FRAME = 180
PEOPLE = {
    "A": ((220, 198, 353, 560), (252, 223, 290, 271)),   # man in sunglasses
    "B": ((114, 210, 230, 560), (147, 223, 182, 269)),   # middle man
}


def load_people(src: Path):
    cap = cv2.VideoCapture(str(src))
    frame = None
    for i in range(SRC_FRAME + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {SRC_FRAME} of {src}")
    cap.release()
    out = {}
    for k, (bb, fb) in PEOPLE.items():
        x1, y1, x2, y2 = bb
        crop = frame[y1:y2, x1:x2].copy()
        face = np.array([fb[0] - x1, fb[1] - y1, fb[2] - x1, fb[3] - y1], np.float64)
        out[k] = (crop, face)
    return out


def background(seed=0):
    rng = np.random.RandomState(seed)
    small = rng.randint(90, 200, (12, 16, 3)).astype(np.uint8)
    bg = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    noise = rng.randint(-12, 12, (H, W, 3))
    return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)


def lerp(a, b, t):
    return a + (b - a) * t


def path(keys, n):
    """keys: [(t0..1, x)] piecewise linear -> list of n values"""
    ts = np.linspace(0, 1, n)
    kt = [k[0] for k in keys]; kx = [k[1] for k in keys]
    return np.interp(ts, kt, kx)


SCENARIOS = {
    # name: (n_frames, {person: (xs path keys, y, scale, z)})
    "behind": (90, {"A": ([(0, 40), (1, 500)], 70, 0.85, 0), "B": ([(0, 270), (1, 270)], 80, 0.85, 1)}),
    "leave_return": (105, {"A": ([(0, 300), (0.3, 700), (0.6, 700), (1, 330)], 70, 0.85, 0),
                           "B": ([(0, 40), (1, 40)], 80, 0.85, 1)}),
    "fast": (60, {"A": ([(0, -40), (0.5, 560), (1, -40)], 70, 0.85, 1), "B": ([(0, 260), (1, 260)], 90, 0.85, 0)}),
    "small_far": (75, {"A": ([(0, 380), (1, 560)], 170, 0.45, 0), "B": ([(0, 60), (1, 120)], 70, 0.85, 1)}),
    "merge": (90, {"A": ([(0, 80), (0.5, 250), (1, 80)], 70, 0.85, 1), "B": ([(0, 420), (0.5, 262), (1, 420)], 72, 0.85, 0)}),
}


def person_mask(shape, face):
    """Rough silhouette: head ellipse around the face + a torso/legs block below it, so people
    overlap like people (faces can sit side by side), not like opaque rectangles."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    fw, fh = face[2] - face[0], face[3] - face[1]
    cv2.ellipse(m, (int(fx), int(fy - 0.1 * fh)), (int(0.8 * fw), int(0.75 * fh)), 0, 0, 360, 1.0, -1)
    cv2.rectangle(m, (int(0.1 * w), int(face[3] + 0.1 * fh)), (int(0.9 * w), h - 1), 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 1.5)


def make_clip(name, people, out_dir: Path):
    n, spec = SCENARIOS[name]
    bg = background(hash(name) % 1000)
    xs = {k: path(v[0], n) for k, v in spec.items()}
    vid = out_dir / f"{name}.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = {k: [] for k in spec}
    keys = sorted(spec, key=lambda k: spec[k][3])                 # back to front
    for i in range(n):
        fr = bg.astype(np.float32)
        owner = np.full((H, W), -1, np.int32)
        for li, k in enumerate(keys):
            crop, face = people[k]
            _, y, sc, _ = spec[k]
            c = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
            m = person_mask(c.shape, face * sc)
            x = int(round(xs[k][i]))
            X1, Y1 = max(0, x), max(0, y)
            X2, Y2 = min(W, x + c.shape[1]), min(H, y + c.shape[0])
            if X2 > X1 and Y2 > Y1:
                cc = c[Y1 - y:Y2 - y, X1 - x:X2 - x]; mm = m[Y1 - y:Y2 - y, X1 - x:X2 - x][..., None]
                fr[Y1:Y2, X1:X2] = fr[Y1:Y2, X1:X2] * (1 - mm) + cc * mm
                ow = owner[Y1:Y2, X1:X2]; ow[mm[..., 0] > 0.5] = li
            gt[k].append([x + face[0] * sc, y + face[1] * sc, x + face[2] * sc, y + face[3] * sc])
        vw.write(np.clip(fr + 0.5, 0, 255).astype(np.uint8))
        for li, k in enumerate(keys):
            fb = gt[k][-1]
            area = max(1.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
            X1, Y1 = int(max(0, fb[0])), int(max(0, fb[1]))
            X2, Y2 = int(min(W, fb[2])), int(min(H, fb[3]))
            vis = float((owner[Y1:Y2, X1:X2] == li).sum()) / area if X2 > X1 and Y2 > Y1 else 0.0
            gt[k][-1] = gt[k][-1] + [vis]
    vw.release()
    return vid, gt


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def person_of(proj, gtk):
    votes = {}
    for t in proj.tracks:
        for f, b in zip(t["frames"], t["boxes"]):
            if f < len(gtk) and _iou(b, gtk[f][:4]) > 0.3:
                votes[t["person_id"]] = votes.get(t["person_id"], 0) + 1
    return max(votes, key=votes.get) if votes else None


def run(name, people, out_dir: Path, ff):
    t0 = time.perf_counter()
    vid, gt = make_clip(name, people, out_dir)
    proj = analyze(vid, out_dir / f"{name}_zoria", ff, every=2)
    pa, pb = person_of(proj, gt["A"]), person_of(proj, gt["B"])
    res = {"scenario": name, "persons": len(proj.persons), "A": pa, "B": pb}
    if pa is None:
        res.update(ok=False, why="A was never detected")
        return res
    sel = [pa]
    out = out_dir / f"{name}_blur.mp4"
    r = render(vid, proj, out, ff, mode="blur-selected", persons=sel, strength=2.0, padding=0.6,
               extend_s=0.5, smooth_s=0.04, max_gap_s=2.0, qa=True, qa_every=2)
    q = r["qa"]
    # ground truth: A's visible face must never keep its detail
    gt_leaks, b_blur, b_chk = [], 0, 0
    c1, c2 = cv2.VideoCapture(str(vid)), cv2.VideoCapture(str(out))
    for i in range(len(gt["A"])):
        ok1, o = c1.read(); ok2, b = c2.read()
        if not (ok1 and ok2):
            break
        fa = gt["A"][i]
        if fa[4] >= 0.6 and fa[2] - fa[0] >= 8:
            s, _ = sharp_ratio(o, b, fa[:4])
            if s is not None and s >= SHARP_THRESH:
                gt_leaks.append((i, round(s, 2)))
        fb = gt["B"][i]
        if fb[4] >= 0.9:
            s, _ = sharp_ratio(o, b, fb[:4])
            if s is not None:
                b_chk += 1
                b_blur += s < SHARP_THRESH
    res.update(qa_leaks=q["leak_count"], gt_leaks=len(gt_leaks), gt_leak_frames=gt_leaks[:10],
               kept_B_blurred=f"{b_blur}/{b_chk}", risky=len([x for x in q["risky"] if x["relevant"]]),
               seconds=round(time.perf_counter() - t0, 1))
    res["ok"] = q["leak_count"] == 0 and not gt_leaks
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "work/scenarios")
    a = ap.parse_args()
    if not a.src.is_file():
        print(json.dumps({"skipped": f"source video {a.src} not found"}))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    ff = find_ffmpeg()
    people = load_people(a.src)
    results = []
    for nm in a.names:
        r = run(nm, people, a.out, ff)
        results.append(r)
        print(json.dumps(r), flush=True)
    bad = [r["scenario"] for r in results if not r["ok"]]
    print(json.dumps({"passed": len(results) - len(bad), "failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
