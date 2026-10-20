"""Leak check for blur-except output: re-detect faces in the RENDERED video and compare
each visible face to the kept persons' centroids. Any confidently detected, recognisable face that
is not a kept person = leak. Usage:
  python tests/verify_leaks.py RENDERED.mp4 PROJECT_DIR --persons 3 [--every 3]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zoria_blur.backend import default_models_dir
from zoria_blur.detector import SCRFD
from zoria_blur.embedder import ArcFace
from zoria_blur.project import Project
from zoria_blur.video_io import FrameReader, probe, find_ffmpeg

ap = argparse.ArgumentParser()
ap.add_argument("video", type=Path); ap.add_argument("project", type=Path)
ap.add_argument("--persons", default=""); ap.add_argument("--every", type=int, default=3)
ap.add_argument("--det-thresh", type=float, default=0.5); ap.add_argument("--leak-sim", type=float, default=0.25)
a = ap.parse_args()
proj = Project.load(a.project)
keep = [int(x) for x in a.persons.split(",") if x]
cents = {p: proj.emb[f"p{p}"] for p in keep}
others = {p["id"]: proj.emb.get(f"p{p['id']}") for p in proj.persons if p["id"] not in keep}
m = default_models_dir()
det = SCRFD(m / "det_10g.onnx"); emb = ArcFace(m / "w600k_mbf.onnx")
ff = find_ffmpeg(); info = probe(a.video, ff)
stats = {"frames_checked": 0, "faces_visible": 0, "kept_visible": 0, "leaks": []}
for i, fr in enumerate(FrameReader(info, ff)):
    if i % a.every: continue
    stats["frames_checked"] += 1
    d, k = det.detect(fr, a.det_thresh)
    if not len(d): continue
    E = emb.embed(fr, list(k))
    for j in range(len(d)):
        stats["faces_visible"] += 1
        sk = max([float(E[j] @ c) for c in cents.values()] or [-1])
        so = max([(float(E[j] @ c), pid) for pid, c in others.items() if c is not None] or [(-1, None)])
        if sk >= a.leak_sim and sk >= so[0]:
            stats["kept_visible"] += 1
        elif so[0] >= a.leak_sim:
            stats["leaks"].append({"frame": i, "box": [round(float(v)) for v in d[j, :4]], "score": round(float(d[j, 4]), 2),
                                   "looks_like_person": so[1], "sim": round(so[0], 3), "sim_kept": round(sk, 3)})
        else:
            stats.setdefault("unmatched_visible", []).append({"frame": i, "score": round(float(d[j, 4]), 2),
                                                              "box": [round(float(v)) for v in d[j, :4]], "sim_kept": round(sk, 3), "sim_other": round(so[0], 3)})
stats["leak_count"] = len(stats["leaks"])
print(json.dumps(stats, indent=1))
