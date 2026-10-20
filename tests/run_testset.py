"""Real-video regression suite: analyze -> render (blur-selected + blur-except) -> QA on every clip
of the stock-footage test set in work/testset (see work/testset/SOURCES.md), then print one table.

    PYTHONPATH=. python tests/run_testset.py                     # whole set (re-analyzes every clip)
    PYTHONPATH=. python tests/run_testset.py --reuse             # keep existing analyses, only render + QA
    PYTHONPATH=. python tests/run_testset.py b_turning_head h_close_faces --sheets
    options: --dir DIR (default work/testset)  --qa-every N (default 3)  --sheets (contact sheets for
             visual review: QA cannot see an unblurred back of the head)  --json FILE

Person choice (stable across re-runs, no ids hard-coded): the most prominent person = largest sum of
face areas over all detections. blur-selected blurs that person; blur-except keeps that person and
blurs everyone else. Pass = no QA leak in either mode.
Runtime on 2 CPU cores: ~1.5-3 min analyze + ~1 min render/QA per clip.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name, scenario, analyze --every
CLIPS = [
    ("a_occlusion_runners", "(a) occlusion: runners pass each other", 1),
    ("b_turning_head", "(b) head turns: back / profile / front", 2),
    ("c_crowd_selfie", "(c) crowd: 6-face group selfie", 2),
    ("d_handheld_selfie", "(d) fast handheld + backlit, 3 faces", 1),
    ("e_leave_frame", "(e) walks toward camera, leaves frame", 2),
    ("f_small_far", "(f) small far faces, street crowd", 2),
    ("g_night_street", "(g) low light night street", 2),
    ("h_close_faces", "(h) faces close: 4-person selfie", 2),
]


def run_cli(args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "zoria_blur", *args]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT),
                       **{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}})
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"{' '.join(args[:2])}: no JSON output\n{p.stderr[-2000:]}")
    res = json.loads(lines[-1])
    if not res.get("ok"):
        raise RuntimeError(f"{' '.join(args[:2])}: {res.get('error')}\n{p.stderr[-2000:]}")
    return res


def prominent_person(proj_dir: Path) -> int:
    d = json.loads((proj_dir / "project.json").read_text())
    area: dict[int, float] = {}
    for t in d["tracks"]:
        a = sum((b[2] - b[0]) * (b[3] - b[1]) for b in t["boxes"])
        area[t["person_id"]] = area.get(t["person_id"], 0.0) + a
    return max(area, key=area.get)


def contact_sheet(video: Path, out: Path, n=12, cols=3, width=480, label=""):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in range(n):
        f = int((i + 0.5) * total / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        h = int(fr.shape[0] * width / fr.shape[1])
        fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
        cv2.putText(fr, f"f{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    img = np.vstack(rows)
    if label:
        cv2.putText(img, label, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def persons_sheet(proj_dir: Path, out: Path, per=6, size=96):
    import cv2
    import numpy as np
    d = json.loads((proj_dir / "project.json").read_text())
    rows = []
    for p in d["persons"]:
        thumbs = []
        for rel in (p.get("thumbnails") or [])[:per]:
            im = cv2.imread(str(proj_dir / rel))
            if im is not None:
                thumbs.append(cv2.resize(im, (size, size)))
        while len(thumbs) < per:
            thumbs.append(np.zeros((size, size, 3), np.uint8))
        lab = np.zeros((size, 150, 3), np.uint8)
        cv2.putText(lab, f"P{p['id']}", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(lab, f"{p['num_detections']} det", (6, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(lab, f"{len(p['track_ids'])} trk", (6, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.hstack([lab] + thumbs))
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_clip(name, scenario, every, base: Path, reuse: bool, qa_every: int, sheets: bool,
             det_mode: str = "balanced") -> list[dict]:
    video = base / f"{name}.mp4"
    pdir = base / "proj" / name
    rdir = base / "renders"
    rdir.mkdir(parents=True, exist_ok=True)
    if reuse and (pdir / "project.json").is_file():
        d = json.loads((pdir / "project.json").read_text())
        a_s = d["analysis"]["timing"].get("seconds_total")
    else:
        t0 = time.perf_counter()
        res = run_cli(["analyze", str(video), "-p", str(pdir), "--det-model", "det_10g", "--every", str(every),
                       "--det-mode", det_mode])
        a_s = res["timing"].get("seconds_total", round(time.perf_counter() - t0, 1))
        d = json.loads((pdir / "project.json").read_text())
    n_persons, n_tracks = len(d["persons"]), len(d["tracks"])
    pid = prominent_person(pdir)
    if sheets:
        persons_sheet(pdir, base / "sheets" / f"{name}_persons.jpg")
    rows = []
    for mode, tag in (("blur-selected", "sel"), ("blur-except", "exc")):
        out = rdir / f"{name}_{tag}_p{pid}.mp4"
        r = run_cli(["render", str(video), "-p", str(pdir), "--mode", mode, "--persons", str(pid),
                     "-o", str(out), "--qa-every", str(qa_every)])
        qa = r.get("qa", {})
        out.with_suffix(".qa.json").write_text(json.dumps(qa, indent=1))   # full QA result for review
        kc, kb = qa.get("kept_face_checks", 0), qa.get("kept_face_blurred", 0)
        rel_risks = [x for x in qa.get("risky", []) if x.get("relevant")]
        rows.append({
            "clip": name, "scenario": scenario, "every": every, "persons": n_persons, "tracks": n_tracks,
            "bodies": len(d.get("bodies", [])), "person": pid, "mode": mode,
            "det_mode": d["analysis"].get("detection", {}).get("det_mode", "fast(v0.2)"),
            "small_mode": d["analysis"].get("detection", {}).get("small_mode"),
            "leaks": qa.get("leak_count"), "leak_frames": qa.get("leak_frames"),
            "leak_ranges": [(x["start_s"], x["end_s"]) for x in qa.get("leak_ranges", [])],
            "leak_kinds": sorted({x["kind"] for x in qa.get("leaks", [])}),
            "kept_blurred_pct": round(100.0 * kb / kc, 1) if kc else None,
            "kept_blurred_ranges": [(x["start_s"], x["end_s"]) for x in qa.get("kept_blurred_ranges", [])],
            "unknown_faces": qa.get("unknown_faces"), "relevant_risks": len(rel_risks),
            "risk_ranges": [(x["type"], x["start_s"], x["end_s"]) for x in rel_risks][:12],
            "analyze_s": a_s, "render_s": r.get("seconds"), "qa_s": qa.get("seconds"),
            "output": str(out),
        })
        if sheets:
            contact_sheet(out, base / "sheets" / f"{name}_{tag}.jpg", label=f"{name} {mode} P{pid}")
    return rows


def fmt_table(rows: list[dict]) -> str:
    hdr = ("| clip | scenario | persons/tracks | det mode (small) | mode (P) | leaks | kept blurred % | analyze s | "
           "render s | QA s | relevant risks |")
    out = [hdr, "|" + "---|" * 11]
    for r in rows:
        kp = "-" if r["kept_blurred_pct"] is None else f"{r['kept_blurred_pct']}"
        lk = f"{r['leaks']}" + (f" {r['leak_kinds']} @ {r['leak_ranges']}" if r["leaks"] else "")
        dm = f"{r.get('det_mode')} ({'on' if r.get('small_mode') else 'off'})"
        out.append(f"| {r['clip']} | {r['scenario']} | {r['persons']}/{r['tracks']} | {dm} | {r['mode']} (P{r['person']}) | "
                   f"{lk} | {kp} | {r['analyze_s']} | {r['render_s']} | {r['qa_s']} | {r['relevant_risks']} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", help="subset of clip names (default: all)")
    ap.add_argument("--dir", type=Path, default=ROOT / "work" / "testset")
    ap.add_argument("--reuse", action="store_true", help="reuse existing analyses (render + QA only)")
    ap.add_argument("--qa-every", type=int, default=3)
    ap.add_argument("--det-mode", default="balanced", choices=["fast", "balanced", "thorough"])
    ap.add_argument("--sheets", action="store_true", help="write contact sheets to DIR/sheets")
    ap.add_argument("--json", type=Path, help="write all rows as JSON (default DIR/results.json)")
    a = ap.parse_args(argv)
    todo = [c for c in CLIPS if not a.clips or c[0] in a.clips]
    missing = [c[0] for c in todo if not (a.dir / f"{c[0]}.mp4").is_file()]
    if missing:
        print(f"missing clips in {a.dir}: {missing} (see SOURCES.md for URLs and cuts)", file=sys.stderr)
        return 2
    rows = []
    for name, scen, every in todo:
        t0 = time.perf_counter()
        try:
            rows += run_clip(name, scen, every, a.dir, a.reuse, a.qa_every, a.sheets, a.det_mode)
        except Exception as e:  # keep going: one broken clip must not hide the others
            print(f"[{name}] ERROR {e}", file=sys.stderr)
            rows.append({"clip": name, "scenario": scen, "error": str(e)})
            continue
        print(f"[{name}] done in {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    ok_rows = [r for r in rows if "error" not in r]
    jp = a.json or (a.dir / "results.json")
    old = []
    if jp.is_file() and a.clips:          # partial run: merge into the existing results
        try:
            old = [r for r in json.loads(jp.read_text()) if r.get("clip") not in a.clips]
        except Exception:
            old = []
    order = {c[0]: i for i, c in enumerate(CLIPS)}
    allrows = sorted(old + rows, key=lambda r: (order.get(r["clip"], 99), r.get("mode", "")))
    jp.write_text(json.dumps(allrows, indent=1))
    print(fmt_table([r for r in allrows if "error" not in r]))
    failed = [r for r in ok_rows if r["leaks"]] + [r for r in rows if "error" in r]
    print(f"\n{len(ok_rows)} renders, {len(failed)} with leaks/errors", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
