"""Face-detector recall on WIDER FACE validation (internal evaluation only; the dataset is for
research use and is never shipped or committed).

    PYTHONPATH=. python tests/eval_wider.py run   --wider work/wider --every 8 --settings s640,s960,...
    PYTHONPATH=. python tests/eval_wider.py report --wider work/wider [--md out.md]

Data layout under --wider (download: Hugging Face dataset CUHK-CSE/wider_face data/WIDER_val.zip +
data/wider_face_split.zip; eval_tools.zip from shuoyang1213.me/WIDERFACE for the easy/medium/hard lists):
    WIDER_val/images/<event>/<file>.jpg
    wider_face_split/wider_face_val_bbx_gt.txt
    eval_tools/ground_truth/wider_{easy,medium,hard}_val.mat

`run` stores raw detections (score >= 0.3) per setting in <wider>/dets_<setting>.npz plus timing;
`report` matches them to ground truth (greedy by score, IoU >= 0.5) and prints recall per face-size
bucket (size = max(w, h) of the GT box in original pixels), recall on the easy/medium/hard face lists,
and false positives per image (a detection matching no GT box - valid or invalid - at IoU >= 0.5;
WIDER leaves many tiny crowd faces unannotated, so this is an upper bound).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zoria_blur.backend import default_models_dir  # noqa: E402
from zoria_blur.detector import SCRFD, detect_tiled, merge_dets  # noqa: E402

BUCKETS = [(0, 10), (10, 16), (16, 24), (24, 32), (32, 64), (64, 1e9)]
BUCKET_NAMES = ["<10", "10-16", "16-24", "24-32", "32-64", ">64"]

SETTINGS = {
    # name: (description, fn(det, img, thresh))
    "s640": ("full frame, det-size 640", lambda d, im, t: d.detect(im, t, det_size=640)),
    "s960": ("full frame, det-size 960", lambda d, im, t: d.detect(im, t, det_size=960)),
    "s1280": ("full frame, det-size 1280", lambda d, im, t: d.detect(im, t, det_size=1280)),
    "t2x2": ("640 full + 2x2 tiles @640", lambda d, im, t: detect_tiled(d, im, t, (2, 2), 640, 640)),
    "t3x2": ("640 full + 3x2 tiles @640", lambda d, im, t: detect_tiled(d, im, t, (3, 2), 640, 640)),
    "t3x2_1280": ("1280 full + 3x2 tiles @640", lambda d, im, t: detect_tiled(d, im, t, (3, 2), 640, 1280)),
    "s640_500m": ("det_500m, det-size 640", None),
    "bal": ("balanced small mode: 640 full + head ROIs (YOLOX-tiny bodies on 2x2 tiles)", None),
}


def balanced_still(det, pdet):
    """balanced small mode on a still image: full pass, bodies on tiles, head-ROI pass. In a video the
    ROI also takes the previous detection frame's small faces; a still image stands in for a static
    scene by taking the full pass's own small faces. The periodic tile refresh (1 of 5 detection frames)
    is NOT included here."""
    from zoria_blur.smallfaces import SmallFaceDetector

    def fn(_d, im, t):
        h, w = im.shape[:2]
        sf = SmallFaceDetector(det, pdet, "balanced", w, h)
        sf.small_mode, sf.small_since = True, 0
        bodies = sf.bodies(im, 1)
        full = det.detect(im, t, det_size=640)
        sf._remember_small(full[0])
        roi = sf._roi_pass(im, t, bodies, full)
        return merge_dets([full, roi], 0.4)
    return fn


def load_gt(wider: Path):
    lines = (wider / "wider_face_split" / "wider_face_val_bbx_gt.txt").read_text().split("\n")
    gt, i = {}, 0
    while i < len(lines) and lines[i].strip():
        name = lines[i].strip(); n = int(lines[i + 1]); i += 2
        rows = []
        for j in range(max(n, 1)):
            v = [int(x) for x in lines[i + j].split()]
            if n:
                rows.append(v)
        i += max(n, 1)
        a = np.array(rows, np.float32).reshape(-1, 10)
        gt[name] = a  # x y w h blur expr illum invalid occl pose
    return gt


def load_splits(wider: Path):
    from scipy.io import loadmat
    out = {}
    for sp in ("easy", "medium", "hard"):
        m = loadmat(str(wider / "eval_tools" / "ground_truth" / f"wider_{sp}_val.mat"))
        ev = m["event_list"]; fl = m["file_list"]; gl = m["gt_list"]
        d = {}
        for e in range(len(ev)):
            event = str(ev[e][0][0])
            for f in range(len(fl[e][0])):
                fname = str(fl[e][0][f][0][0])
                idx = gl[e][0][f][0].reshape(-1).astype(int) - 1
                d[f"{event}/{fname}.jpg"] = idx
        out[sp] = d
    return out


def subset(gt, every):
    names = sorted(gt)
    by_ev = {}
    for n in names:
        by_ev.setdefault(n.split("/")[0], []).append(n)
    sel = []
    for ev in sorted(by_ev):
        sel += by_ev[ev][::every]
    return sel


def iou(a, b):
    """a: Nx4, b: Mx4 (x1 y1 x2 y2) -> NxM"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def cmd_run(args):
    wider = Path(args.wider)
    gt = load_gt(wider)
    names = subset(gt, args.every)
    if args.limit:
        names = names[:args.limit]
    models = Path(args.models) if args.models else default_models_dir()
    dets = {
        "det_10g": SCRFD(models / "det_10g.onnx", "cpu", 640, args.threads),
    }
    for s in args.settings.split(","):
        out = wider / f"dets_{s}.npz"
        if out.exists() and not args.force:
            print(f"{s}: exists, skip", flush=True); continue
        if s == "s640_500m":
            d500 = SCRFD(models / "det_500m.onnx", "cpu", 640, args.threads)
            fn = lambda d, im, t: d500.detect(im, t, det_size=640)  # noqa: E731
        elif s == "bal":
            from zoria_blur.person_detector import YOLOXPerson
            fn = balanced_still(dets["det_10g"], YOLOXPerson(models / "yolox_tiny.onnx", "cpu", args.threads))
        else:
            fn = SETTINGS[s][1]
        res, t_total = {}, 0.0
        for i, n in enumerate(names):
            img = cv2.imread(str(wider / "WIDER_val" / "images" / n))
            t0 = time.perf_counter()
            d, _ = fn(dets["det_10g"], img, 0.3)
            t_total += time.perf_counter() - t0
            res[n] = d[:, :5].astype(np.float32)
            if i % 50 == 0:
                print(f"{s}: {i}/{len(names)} {t_total / (i + 1):.3f} s/img", flush=True)
        np.savez_compressed(out, names=np.array(names), **{f"d{i}": res[n] for i, n in enumerate(names)},
                            seconds=np.array([t_total / len(names)]))
        print(f"{s}: done {t_total / len(names):.3f} s/img", flush=True)


def evaluate(wider, s, gt, splits, thresh):
    z = np.load(wider / f"dets_{s}.npz")
    names = [str(x) for x in z["names"]]
    hit_b = np.zeros(len(BUCKETS)); tot_b = np.zeros(len(BUCKETS))
    sp_hit = {k: 0 for k in splits}; sp_tot = {k: 0 for k in splits}
    fp = 0
    for i, n in enumerate(names):
        d = z[f"d{i}"]; d = d[d[:, 4] >= thresh]
        g = gt[n]
        gb = np.c_[g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]]
        valid = (g[:, 7] == 0) & (g[:, 2] > 0) & (g[:, 3] > 0)
        ov = iou(d[:, :4], gb)
        # greedy match by score (dets are sorted by score)
        matched = np.zeros(len(g), bool); det_ok = np.zeros(len(d), bool)
        for j in range(len(d)):
            if ov.shape[1] == 0:
                break
            cand = np.where(~matched & (ov[j] >= 0.5))[0]
            if len(cand):
                k = cand[np.argmax(ov[j, cand])]; matched[k] = True; det_ok[j] = True
        # a det overlapping any GT (incl. invalid / already matched duplicates of one face) is not counted FP
        any_ov = ov.max(1) >= 0.5 if ov.shape[1] else np.zeros(len(d), bool)
        fp += int((~det_ok & ~any_ov).sum())
        size = np.maximum(g[:, 2], g[:, 3])
        for bi, (lo, hi) in enumerate(BUCKETS):
            m = valid & (size >= lo) & (size < hi)
            tot_b[bi] += m.sum(); hit_b[bi] += (matched & m).sum()
        for k in splits:
            idx = splits[k].get(n, np.array([], int))
            sp_tot[k] += len(idx); sp_hit[k] += int(matched[idx].sum()) if len(idx) else 0
    return {
        "images": len(names), "seconds_per_image": float(z["seconds"][0]),
        "buckets": {BUCKET_NAMES[b]: [int(hit_b[b]), int(tot_b[b])] for b in range(len(BUCKETS))},
        "all": [int(hit_b.sum()), int(tot_b.sum())],
        "splits": {k: [sp_hit[k], sp_tot[k]] for k in splits},
        "fp_per_image": fp / max(1, len(names)),
    }


def cmd_report(args):
    wider = Path(args.wider)
    gt = load_gt(wider); splits = load_splits(wider)
    allres = {}
    lines = []
    for thresh in (0.5, 0.3):
        lines.append(f"\n### score >= {thresh}\n")
        lines.append("| setting | " + " | ".join(BUCKET_NAMES) + " | all | easy | medium | hard | FP/img | s/img |")
        lines.append("|---" * (len(BUCKET_NAMES) + 7) + "|")
        for s in SETTINGS:
            if not (wider / f"dets_{s}.npz").exists():
                continue
            r = evaluate(wider, s, gt, splits, thresh)
            allres[f"{s}@{thresh}"] = r
            pct = lambda hv: f"{100 * hv[0] / max(1, hv[1]):.1f}"  # noqa: E731
            lines.append(f"| {s} ({SETTINGS[s][0]}) | " + " | ".join(pct(r["buckets"][b]) for b in BUCKET_NAMES)
                         + f" | {pct(r['all'])} | {pct(r['splits']['easy'])} | {pct(r['splits']['medium'])} | "
                           f"{pct(r['splits']['hard'])} | {r['fp_per_image']:.2f} | {r['seconds_per_image']:.2f} |")
        if allres:
            any_r = next(iter(allres.values()))
            lines.append("\nGT faces per bucket: " + ", ".join(f"{b}: {any_r['buckets'][b][1]}" for b in BUCKET_NAMES)
                         + f"; images: {any_r['images']}; easy/medium/hard faces: "
                         + "/".join(str(any_r['splits'][k][1]) for k in ('easy', 'medium', 'hard')))
    txt = "\n".join(lines)
    print(txt)
    (wider / "wider_results.json").write_text(json.dumps(allres, indent=1))
    if args.md:
        Path(args.md).write_text(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--wider", default="work/wider"); r.add_argument("--every", type=int, default=8)
    r.add_argument("--settings", default="s640,s960,s1280,t2x2,t3x2,t3x2_1280"); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--threads", type=int, default=2); r.add_argument("--models"); r.add_argument("--force", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--wider", default="work/wider"); p.add_argument("--md")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)
