"""Live-preview parity: every pixel the export blurs (alpha > THRESH) must lie inside the live preview's
blur ellipses of the same frame. python tests/eval_preview_parity.py PROJECT_DIR [THRESH]
The preview never carves kept faces, so it may blur more than the export, never less."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zoria_blur import renderer as R  # noqa: E402
from zoria_blur.project import Project  # noqa: E402

KW = dict(keep_track_conf=0.30, keep_det_sim=0.18, extend_s=0.5, smooth_s=0.04, max_gap_s=2.0)


def check(proj, mode, pers, thresh=0.02):
    pv = proj.data["video"]; W, H = pv["width"], pv["height"]
    orig = R._blur_roi
    R._blur_roi = lambda roi, fw, st, s: np.full_like(roi, 255)     # output = 255 * export alpha
    try:
        if mode == "blur-selected" and not pers:
            per, prot = {}, {}
        else:
            per, rep = R.plan_regions(proj, mode, pers, **KW); prot = rep.get("_protect", {})
        plan = R.preview_plan(proj, mode, pers, padding=0.6, **KW)["frames"]
        bad, worst = 0, 0
        for f, regs in per.items():
            a = R.apply_regions(np.zeros((H, W, 3), np.uint8), regs, "gaussian", 2.0, 0.6, keep=prot.get(f))[..., 0] / 255.0
            m = np.zeros((H, W), np.uint8)
            for cx, cy, rx, ry in plan.get(str(f), []):
                cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1, -1)
            miss = int(((a > thresh) & (m == 0)).sum())
            if miss:
                bad += 1; worst = max(worst, miss)
        return len(per), bad, worst
    finally:
        R._blur_roi = orig


if __name__ == "__main__":
    proj = Project.load(Path(sys.argv[1]))
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    ids = [p["id"] for p in proj.persons]
    fail = 0
    for mode, pers in [("blur-selected", ids[:2]), ("blur-selected", ids[:1]), ("blur-except", ids[:1]),
                       ("blur-except", ids[:3]), ("blur-except", [])]:
        n, bad, worst = check(proj, mode, pers, th)
        fail += bad
        print(f"{mode:13s} {pers}: {n} frames with blur, {bad} with export blur outside the preview (max {worst} px)")
    sys.exit(1 if fail else 0)
