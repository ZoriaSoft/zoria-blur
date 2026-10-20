"""Calibration of qa.sharp_ratio on the real-video test set (work/testset).

For every clip with an analysis (proj/<clip>) and renders (renders/<clip>_{sel,exc}_p<P>.mp4) it collects the
sharpness of real face detections, every `--every`-th frame:
  untouched   original vs. a crf-18 / veryfast libx264 re-encode of the original (what a kept face can at best be)
  kept        kept persons' detections in the blur-except render (want: sharp)
  blurred     detections of persons that must be blurred, in both renders (want: ~0)
  k3, s1.5, s3, pix_app  synthetic blurs of the must-blur faces in the original: 3x3 Gaussian, sigma 1.5, sigma 3,
              the renderer's own pixelate style at strength 1 / 2 (9 / 4.5 blocks across the face)
and prints min / p5 / median / p95 / max and the share at or above the QA threshold per group.

    PYTHONPATH=. python tests/eval_sharpness.py [clips...] [--every 3] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zoria_blur.project import Project  # noqa: E402
from zoria_blur.qa import sharp_ratio  # noqa: E402
from zoria_blur.renderer import _blur_roi, plan_regions  # noqa: E402
from zoria_blur.video_io import FrameReader, probe  # noqa: E402


def frames(path):
    return iter(FrameReader(probe(Path(path), "ffmpeg"), "ffmpeg"))


def reencode(src: Path, dst: Path):
    if not dst.is_file():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-c:v", "libx264", "-crf", "18",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dst)], check=True)


def pix_app(img, box, strength=1.0):
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    w = x2 - x1
    H, W = img.shape[:2]
    X1, Y1, X2, Y2 = max(0, x1 - w // 3), max(0, y1 - w // 3), min(W, x2 + w // 3), min(H, y2 + w // 3)
    out = img.copy()
    out[Y1:Y2, X1:X2] = _blur_roi(img[Y1:Y2, X1:X2], max(w, 1), "pixelate", strength)
    return out


def eval_clip(base: Path, name: str, every: int, tmp: Path) -> dict[str, list]:
    pdir = base / "proj" / name
    proj = Project.load(pdir)
    rend = {tag: sorted((base / "renders").glob(f"{name}_{tag}_p*.mp4")) for tag in ("sel", "exc")}
    if not rend["sel"] or not rend["exc"]:
        return {}
    pid = int(rend["exc"][0].stem.rsplit("_p", 1)[1])
    _, rep = plan_regions(proj, "blur-except", [pid])
    kept_exc = set(rep["kept_tracks"])
    must = {"sel": {}, "exc": {}}
    kept = {}
    allf = {}
    for t in proj.tracks:
        if t.get("manual"):
            continue
        for f, b in zip(t["frames"], t["boxes"]):
            f = int(f)
            if f % every:
                continue
            allf.setdefault(f, []).append(b)
            if t["person_id"] == pid:
                must["sel"].setdefault(f, []).append(b)
            if t["id"] in kept_exc:
                kept.setdefault(f, []).append(b)
            else:
                must["exc"].setdefault(f, []).append(b)
    src = base / f"{name}.mp4"
    re = tmp / f"{name}_re18.mp4"
    reencode(src, re)
    it = {"o": frames(src), "re": frames(re), "sel": frames(rend["sel"][0]), "exc": frames(rend["exc"][0])}
    out = {k: [] for k in ("untouched", "kept", "blurred", "k3", "s1.5", "s3", "pix_app", "pix_app2")}
    for i in range(proj.data["video"]["frame_count"]):
        try:
            fr = {k: next(v) for k, v in it.items()}
        except StopIteration:
            break
        if i not in allf:
            continue
        o = fr["o"]
        for b in kept.get(i, []):
            r, va = sharp_ratio(o, fr["re"], b)
            if r is not None:
                out["untouched"].append((r, b[2] - b[0], va))
            r, va = sharp_ratio(o, fr["exc"], b)
            if r is not None:
                out["kept"].append((r, b[2] - b[0], va))
        syn = None
        for tag in ("sel", "exc"):
            for b in must[tag].get(i, []):
                r, va = sharp_ratio(o, fr[tag], b)
                if r is not None:
                    out["blurred"].append((r, b[2] - b[0], va, i, tag == "exc", *b[:4]))
        for b in must["exc"].get(i, []) + must["sel"].get(i, []):
            if syn is None:
                syn = {"k3": cv2.GaussianBlur(o, (3, 3), 0), "s1.5": cv2.GaussianBlur(o, (0, 0), 1.5),
                       "s3": cv2.GaussianBlur(o, (0, 0), 3.0)}
            for k, img in syn.items():
                r, va = sharp_ratio(o, img, b)
                if r is not None:
                    out[k].append((r, b[2] - b[0], va))
            for k, st in (("pix_app", 1.0), ("pix_app2", 2.0)):
                r, va = sharp_ratio(o, pix_app(o, b, st), b)
                if r is not None:
                    out[k].append((r, b[2] - b[0], va))
    return out


def stats(v, thr):
    if not v:
        return "n=0"
    a = np.array([x[0] for x in v])
    q = np.percentile(a, [0, 5, 50, 95, 100])
    return (f"n={len(a):4d}  min {q[0]:.2f}  p5 {q[1]:.2f}  med {q[2]:.2f}  p95 {q[3]:.2f}  max {q[4]:.2f}  "
            f">=thr {100.0 * (a >= thr).mean():5.1f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--dir", type=Path, default=ROOT / "work" / "testset")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--thr", type=float, default=None, help="QA threshold (default: qa.run_qa's)")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/zb_eval"))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.thr is None:
        import inspect
        from zoria_blur.qa import run_qa
        a.thr = inspect.signature(run_qa).parameters["sharp_thresh"].default
    a.tmp.mkdir(parents=True, exist_ok=True)
    names = a.clips or sorted(p.name for p in (a.dir / "proj").iterdir() if (p / "project.json").is_file())
    allres = {}
    for n in names:
        res = eval_clip(a.dir, n, a.every, a.tmp)
        allres[n] = res
        print(f"== {n}", flush=True)
        for k, v in res.items():
            print(f"  {k:9s} {stats(v, a.thr)}", flush=True)
    print(f"== ALL (threshold {a.thr})")
    for k in ("untouched", "kept", "blurred", "k3", "s1.5", "s3", "pix_app", "pix_app2"):
        v = [x for r in allres.values() for x in r.get(k, [])]
        print(f"  {k:9s} {stats(v, a.thr)}")
    if a.json:
        a.json.write_text(json.dumps({n: {k: [[round(float(x), 4) for x in t] for t in v] for k, v in r.items()}
                                      for n, r in allres.items()}))


if __name__ == "__main__":
    main()
