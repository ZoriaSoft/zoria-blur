"""Semi-ground-truth face recall on real test-clip frames.

    PYTHONPATH=. python tests/semigt.py propose   # strongest detector -> numbered boxes + crop sheets to verify by eye
    PYTHONPATH=. python tests/semigt.py eval      # recall of each det mode vs the verified GT (gt.json)

Ground truth lives in work/testset/semigt/gt.json: {"<clip>_f<N>": {"keep": [proposal ids that are real
faces], "add": [[x1,y1,x2,y2], ...] faces the detector missed, drawn by hand}}. A GT face counts as found when
a detection overlaps it with IoU >= 0.4 (one detection per face) (hand-drawn boxes on 8-15 px faces are not pixel exact) or
the detection centre lies inside a hand-added box.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from zoria_blur.backend import default_models_dir  # noqa: E402
from zoria_blur.detector import SCRFD, detect_tiled, merge_dets  # noqa: E402

D = ROOT / "work" / "testset" / "semigt"


def strongest(det, img, t=0.2):
    a = detect_tiled(det, img, t, (3, 2), 640, 1280)
    b = detect_tiled(det, img, t, (4, 3), 640, None)
    return merge_dets([a, b], 0.4)


def propose():
    det = SCRFD(default_models_dir() / "det_10g.onnx", "cpu", 640, 2)
    out = {}
    for p in sorted((D / "frames").glob("*.png")):
        img = cv2.imread(str(p))
        d, _ = strongest(det, img)
        key = p.stem
        out[key] = [[round(float(v), 1) for v in r[:4]] + [round(float(r[4]), 3)] for r in d]
        vis = img.copy()
        for i, r in enumerate(d):
            x1, y1, x2, y2 = [int(v) for v in r[:4]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0) if r[4] >= 0.5 else (0, 200, 255), 1)
            cv2.putText(vis, str(i), (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imwrite(str(D / f"{key}_boxes.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # crop sheet: each proposal with 1x context around it, upscaled to 112 px
        tiles = []
        for i, r in enumerate(d):
            x1, y1, x2, y2 = r[:4]; w, h = x2 - x1, y2 - y1; s = max(w, h) * 2.0
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            X2, Y2 = int(min(img.shape[1], cx + s / 2)), int(min(img.shape[0], cy + s / 2))
            c = cv2.resize(img[Y1:Y2, X1:X2], (112, 112), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"{i} {r[4]:.2f} {int(max(w, h))}px", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1)
            tiles.append(c)
        if tiles:
            cols = 10
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imwrite(str(D / f"{key}_crops.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(key, len(d), flush=True)
    (D / "proposals.json").write_text(json.dumps(out, indent=1))


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def gt_boxes():
    prop = json.loads((D / "proposals.json").read_text())
    gt = json.loads((D / "gt.json").read_text())
    out = {}
    for k, g in gt.items():
        if k.startswith("_"):
            continue
        boxes = [(prop[k][i][:4], False) for i in g["keep"]] + [(b, True) for b in g.get("add", [])]
        ign = [prop[k][i][:4] for i in range(len(prop[k])) if i not in g["keep"] and i not in g.get("reject", [])]
        out[k] = (boxes, ign)
    return out


MODES = {
    "s640": lambda d, im, t: d.detect(im, t, det_size=640),
    "s960": lambda d, im, t: d.detect(im, t, det_size=960),
    "s1280": lambda d, im, t: d.detect(im, t, det_size=1280),
    "t2x2": lambda d, im, t: detect_tiled(d, im, t, (2, 2), 640, 640),
    "t3x2": lambda d, im, t: detect_tiled(d, im, t, (3, 2), 640, 640),
    "t3x2_1280": lambda d, im, t: detect_tiled(d, im, t, (3, 2), 640, 1280),
}


def evaluate():
    det = SCRFD(default_models_dir() / "det_10g.onnx", "cpu", 640, 2)
    from zoria_blur.person_detector import YOLOXPerson
    from eval_wider import balanced_still
    MODES["bal"] = balanced_still(det, YOLOXPerson(default_models_dir() / "yolox_tiny.onnx", "cpu", 2))
    gts = gt_boxes()
    res = {}
    for m, fn in MODES.items():
        for k, (boxes, ign) in gts.items():
            img = cv2.imread(str(D / "frames" / f"{k}.png"))
            t0 = time.perf_counter(); d, _ = fn(det, img, 0.3); dt = time.perf_counter() - t0
            for th in (0.5, 0.3):
                dd = d[d[:, 4] >= th]
                found = 0; sizes_missed = []
                used = set()
                for b, manual in boxes:
                    hit = False
                    for j, r in enumerate(dd):
                        if j in used:
                            continue
                        cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
                        if _iou(b, r[:4]) >= 0.4 or (manual and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]):
                            hit = True; used.add(j); break
                    found += hit
                    if not hit:
                        sizes_missed.append(int(max(b[2] - b[0], b[3] - b[1])))
                fp = sum(1 for j, r in enumerate(dd) if j not in used
                         and not any(_iou(b, r[:4]) >= 0.4 for b in ign))
                res.setdefault(m, {}).setdefault(str(th), {})[k] = {"found": found, "total": len(boxes), "fp": fp,
                                                                    "missed_sizes": sizes_missed, "sec": round(dt, 3)}
        print(m, {k: (v["found"], v["total"]) for k, v in res[m]["0.5"].items()}, flush=True)
    (D / "semigt_results.json").write_text(json.dumps(res, indent=1))
    clips = sorted({k.rsplit("_f", 1)[0] for k in gts})
    lines = []
    for th in ("0.5", "0.3"):
        lines.append(f"\n### score >= {th} (found / hand-counted faces in 3 frames; FP = detections on no GT face)\n")
        lines.append("| mode | " + " | ".join(clips) + " | total | FP (9 frames) | s/frame |")
        lines.append("|---" * (len(clips) + 4) + "|")
        for m in MODES:
            r = res[m][th]; cells = []; F = T = FP = 0; S = []
            for c in clips:
                f = sum(v["found"] for k, v in r.items() if k.startswith(c)); t = sum(v["total"] for k, v in r.items() if k.startswith(c))
                cells.append(f"{f}/{t} ({100 * f / max(1, t):.0f}%)"); F += f; T += t
            FP = sum(v["fp"] for v in r.values()); S = [v["sec"] for v in r.values()]
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {F}/{T} ({100 * F / max(1, T):.0f}%) | {FP} | {np.mean(S):.2f} |")
    txt = "\n".join(lines); print(txt)
    (D / "semigt_results.md").write_text(txt + "\n")


if __name__ == "__main__":
    {"propose": propose, "eval": evaluate}[sys.argv[1]]()
