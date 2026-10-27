"""zoria-blur command line.

  python -m zoria_blur analyze VIDEO [-p PROJECT_DIR]
  python -m zoria_blur persons PROJECT_DIR
  python -m zoria_blur render VIDEO -p PROJECT_DIR --mode blur-except --persons 1 [-o out.mp4]
  python -m zoria_blur merge PROJECT_DIR 1 3          (person 3 -> person 1)
  python -m zoria_blur split PROJECT_DIR --tracks 7,9 (tracks -> new person)
  python -m zoria_blur assign PROJECT_DIR --tracks 7 --person 2
  python -m zoria_blur rename PROJECT_DIR 2 "Ali"
  python -m zoria_blur recluster PROJECT_DIR --threshold 0.45
  python -m zoria_blur qa RENDERED -p PROJECT_DIR      (leak check; runs automatically after render)
  python -m zoria_blur addbox VIDEO -p PROJECT_DIR --person 2 --frame 340 --box x1,y1,x2,y2
  python -m zoria_blur preview VIDEO -p PROJECT_DIR --frame 120 --mode ... --persons ... -o prev.png
  python -m zoria_blur probe VIDEO                     (size, fps, duration, frame count)
  python -m zoria_blur tracks PROJECT_DIR [--person N] (tracks with a thumbnail each, for split)
  python -m zoria_blur devices [--only dml] [--no-test] (system check: CPU, memory, GPUs, CUDA / DirectML / CPU)

Every command prints a single JSON result on stdout (machine readable for the UI).
With --json-progress, progress events are printed as JSON lines on stdout before it.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

BACKENDS = ("cpu", "dml", "cuda", "auto")   # = backend.BACKENDS; backend (onnxruntime) is imported lazily so
                                            # light commands (frames, plan, probe) start fast


def used_summary() -> dict:
    from .backend import used_summary as us
    return us()
from .project import Project, default_project_dir
from .video_io import find_ffmpeg


def _ids(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def _emit(obj):
    try:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        # the app is gone (closed / cancelled the job): its pipe is closed. Nobody reads the result, so stop
        # quietly; an exception here would reach PyInstaller's windowed bootloader, which shows a modal error box.
        os._exit(3)


def _progress_printer(enabled):
    if not enabled:
        def p(ev):
            if ev.get("event") == "progress":
                tot = max(1, ev["total"])
                sys.stderr.write(f"\r[{ev['phase']}] {ev['frame']}/{tot} ({100 * ev['frame'] / tot:.0f}%)   ")
                sys.stderr.flush()
            elif ev.get("event") == "done":
                sys.stderr.write("\n")
        return p
    return _emit


def _current_risks(proj: Project) -> list[dict]:
    """Risky moments for the *current* persons. analysis.risks is written once by analyze; after merge / split /
    assign / addbox its person ids are stale, while the QA after an export recomputes them, so the app's timeline
    and its done panel could disagree. Recomputed here (cheap, no video) so both use the same list."""
    try:
        from .risks import compute_risks
        return compute_risks(proj)
    except Exception:  # noqa: BLE001 - an old project without what compute_risks needs: the stored list
        return proj.data.get("analysis", {}).get("risks", [])


def _project_payload(proj: Project) -> dict:
    """What the UI needs after any project command: persons (thumbnail paths relative to `project`),
    video facts and the risky time ranges for the timeline."""
    v = proj.data["video"]
    fps = v["fps_float"]
    return {"project": str(Path(proj.root).resolve()),
            "video": {"path": v["path"], "width": v["width"], "height": v["height"], "fps": fps,
                      "frame_count": v["frame_count"], "duration_s": round(v["frame_count"] / fps, 3) if fps else 0.0},
            "persons": proj.summary(), "risks": _current_risks(proj),
            "last_render": proj.data.get("last_render")}


def _tracks_payload(proj: Project, person: int | None) -> list[dict]:
    fps = proj.data["video"]["fps_float"]
    out = []
    for t in proj.tracks:
        if person is not None and t["person_id"] != person:
            continue
        crops = sorted(t.get("crops", []), key=lambda c: -c.get("quality", 0))
        thumb = None
        if crops and (Path(proj.root) / "crops" / crops[0]["file"]).is_file():
            thumb = f"crops/{crops[0]['file']}"
        fr = t["frames"]
        out.append({"id": t["id"], "person_id": t["person_id"], "manual": bool(t.get("manual")),
                    "start_s": round(fr[0] / fps, 2) if fr else None, "end_s": round(fr[-1] / fps, 2) if fr else None,
                    "num_detections": len(fr), "confidence": t.get("confidence", 0.0), "thumbnail": thumb})
    return out


def _render_opts(ap):
    ap.add_argument("--mode", required=True, choices=["blur-selected", "blur-except"])
    ap.add_argument("--persons", type=_ids, default=[], help="comma separated person ids, e.g. 1,3")
    ap.add_argument("--style", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=float, default=2.0, help="blur strength multiplier (1 = strong)")
    ap.add_argument("--padding", type=float, default=0.6, help="extra size around the face box (0.3 = +30%%)")
    ap.add_argument("--keep-conf", type=float, default=0.30,
                    help="blur-except: min track similarity to a kept person to leave it visible")
    ap.add_argument("--keep-det-sim", type=float, default=0.18,
                    help="blur-except: a single detection below this similarity is blurred")
    ap.add_argument("--extend", type=float, default=0.5, help="seconds to extend blur before/after a track")
    ap.add_argument("--smooth", type=float, default=0.04, help="temporal smoothing sigma in seconds")
    ap.add_argument("--max-gap", type=float, default=2.0, help="interpolate detection gaps up to N seconds")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):      # UTF-8 JSON on Windows pipes too (Turkish names)
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="zoria-blur", description="Offline face blur engine")
    ap.add_argument("--ffmpeg", help="path to ffmpeg executable")
    ap.add_argument("--backend", choices=BACKENDS, default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = auto)")
    ap.add_argument("--models", type=Path, help="models directory (default ./models or $ZORIA_MODELS)")
    ap.add_argument("--dml-device", type=int,
                    help="DirectML adapter index (see 'devices' dml_adapters; default: the high-performance GPU, "
                         "or $ZORIA_DML_DEVICE)")
    ap.add_argument("--json-progress", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("video", type=Path)
    a.add_argument("-p", "--project", type=Path)
    a.add_argument("--every", type=int, default=2, help="run the detector every N frames")
    a.add_argument("--det-model", default="det_10g", choices=["det_10g", "det_2.5g", "det_500m"])
    a.add_argument("--det-size", type=int, default=640, help="full-frame detector input size (frame is downscaled)")
    a.add_argument("--det-mode", default="balanced", choices=["fast", "balanced", "thorough"],
                   help="small faces: fast = one 640 pass; balanced = adds tiles/head ROIs automatically once small "
                        "people are seen (default); thorough = tiles on every detection frame")
    a.add_argument("--det-thresh", type=float, default=0.5)
    a.add_argument("--low-thresh", type=float, default=0.25)
    a.add_argument("--weak-thresh", type=float, default=0.3,
                   help="weak faces (this score up to det-thresh) that start no track are kept; blur-except blurs them")
    a.add_argument("--min-emb-face", type=int, default=28, help="min face width (px) to compute embedding")
    a.add_argument("--cluster-thresh", type=float, default=0.40, help="cosine similarity to merge tracks")
    a.add_argument("--person-model", default="yolox_tiny", choices=["yolox_tiny", "yolox_nano", "yolox_s", "none"],
                   help="person (body) detector for person-level tracking; none = face-only (old behaviour)")
    a.add_argument("--person-every", type=int, default=1,
                   help="run the person detector on every Nth face-detection frame (tracker predicts between)")
    a.add_argument("--person-thresh", type=float, default=0.3)
    a.add_argument("--no-split-faces", action="store_true", help="disable merged-face re-detection")

    r = sub.add_parser("render")
    r.add_argument("video", type=Path)
    r.add_argument("-p", "--project", type=Path)
    r.add_argument("-o", "--output", type=Path)
    r.add_argument("--crf", type=int, default=18)
    r.add_argument("--preset", default="veryfast")
    r.add_argument("--no-qa", action="store_true", help="skip the automatic QA pass after rendering")
    r.add_argument("--encoder", default="auto", choices=["auto", "x264", "nvenc", "qsv", "amf"],
                   help="H.264 encoder: auto = the graphics card's own encoder when it passes a test encode, else x264")
    r.add_argument("--qa-every", type=int, default=3, help="QA: check every Nth frame")
    _render_opts(r)

    q = sub.add_parser("qa", help="check a rendered video for sharp faces that should be blurred")
    q.add_argument("rendered", type=Path)
    q.add_argument("-p", "--project", type=Path, required=True)
    q.add_argument("--original", type=Path, help="original video (default: the analyzed one)")
    q.add_argument("--mode", choices=["blur-selected", "blur-except"], help="default: last render's")
    q.add_argument("--persons", type=_ids, default=None, help="default: last render's")
    q.add_argument("--every", type=int, default=3)
    q.add_argument("--det-thresh", type=float, default=0.3)
    q.add_argument("--sharp", type=float, default=None,
                   help="kept-detail ratio (qa.sharp_ratio) at or above which a face is sharp (default qa.SHARP_THRESH)")

    ab = sub.add_parser("addbox", help="add a manual face box; optical flow continues it until the next detection")
    ab.add_argument("video", type=Path)
    ab.add_argument("-p", "--project", type=Path)
    ab.add_argument("--person", type=int, required=True, help="person id (0 = new person)")
    ab.add_argument("--frame", type=int, required=True)
    ab.add_argument("--box", required=True, help="x1,y1,x2,y2 in original video pixels")
    ab.add_argument("--max-seconds", type=float, default=1.5)

    pv = sub.add_parser("preview")
    pv.add_argument("video", type=Path)
    pv.add_argument("-p", "--project", type=Path)
    pv.add_argument("--frame", type=int, required=True)
    pv.add_argument("-o", "--output", type=Path, required=True)
    pv.add_argument("--side-by-side", action="store_true")
    pv.add_argument("--original-out", type=Path, help="also write the untouched frame here")
    _render_opts(pv)

    pl = sub.add_parser("plan", help="blur regions per frame for the app's live preview (JSON file)")
    pl.add_argument("project", type=Path)
    pl.add_argument("-o", "--output", type=Path, required=True)
    _render_opts(pl)

    fr = sub.add_parser("frames", help="raw RGBA frames of the video on stdout (live preview player)")
    fr.add_argument("video", type=Path)
    fr.add_argument("--start", type=int, default=0, help="first frame (CFR index, as the renderer counts)")
    fr.add_argument("--width", type=int, required=True)
    fr.add_argument("--height", type=int, required=True)

    dv = sub.add_parser("devices", help="probe CUDA / DirectML / CPU with a test inference; prints the choice")
    dv.add_argument("--only", choices=("cuda", "dml", "cpu"), help="probe just this backend (quick start-up check)")
    dv.add_argument("--no-test", action="store_true", help="skip the test run of the real face detector")
    dv.add_argument("--gpu-test-only", action="store_true", help="test run only on a GPU backend (never the CPU)")
    pb = sub.add_parser("probe"); pb.add_argument("video", type=Path)
    tk = sub.add_parser("tracks"); tk.add_argument("project", type=Path); tk.add_argument("--person", type=int)
    s = sub.add_parser("persons"); s.add_argument("project", type=Path)
    m = sub.add_parser("merge"); m.add_argument("project", type=Path); m.add_argument("ids", type=int, nargs="+")
    sp = sub.add_parser("split"); sp.add_argument("project", type=Path); sp.add_argument("--tracks", type=_ids, required=True)
    asg = sub.add_parser("assign"); asg.add_argument("project", type=Path)
    asg.add_argument("--tracks", type=_ids, required=True); asg.add_argument("--person", type=int, required=True)
    rn = sub.add_parser("rename"); rn.add_argument("project", type=Path); rn.add_argument("id", type=int); rn.add_argument("name")
    br = sub.add_parser("bridge"); br.add_argument("video", type=Path); br.add_argument("-p", "--project", type=Path)
    rc = sub.add_parser("recluster"); rc.add_argument("project", type=Path); rc.add_argument("--threshold", type=float, required=True)

    args = ap.parse_args(argv)
    if args.dml_device is not None:
        os.environ["ZORIA_DML_DEVICE"] = str(args.dml_device)
    prog = _progress_printer(args.json_progress)
    try:
        if args.cmd == "analyze":
            from .analyze import analyze
            ff = find_ffmpeg(args.ffmpeg)
            pdir = args.project or default_project_dir(args.video)
            proj = analyze(args.video, pdir, ff, backend=args.backend, threads=args.threads, every=args.every,
                           det_model=args.det_model, det_size=args.det_size, det_thresh=args.det_thresh,
                           low_thresh=args.low_thresh, weak_thresh=args.weak_thresh, min_emb_face=args.min_emb_face,
                           cluster_thresh=args.cluster_thresh, models_dir=args.models,
                           person_model=args.person_model, person_every=args.person_every,
                           person_thresh=args.person_thresh, split_faces=not args.no_split_faces,
                           det_mode=args.det_mode, progress=prog)
            _emit({"ok": True, **_project_payload(proj), "timing": proj.data["analysis"]["timing"],
                   "device": used_summary()})
        elif args.cmd == "plan":
            from .renderer import preview_plan
            proj = Project.load(args.project)
            plan = preview_plan(proj, args.mode, args.persons, padding=args.padding, keep_track_conf=args.keep_conf,
                                keep_det_sim=args.keep_det_sim, extend_s=args.extend, smooth_s=args.smooth,
                                max_gap_s=args.max_gap)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, args.output)
            _emit({"ok": True, "output": str(args.output), "frames_with_regions": len(plan["frames"])})
        elif args.cmd == "frames":
            from .video_io import stream_rgba
            stream_rgba(args.video, find_ffmpeg(args.ffmpeg), args.start, args.width, args.height)
        elif args.cmd == "devices":
            from .backend import probe_devices
            _emit({"ok": True, **probe_devices(args.models, args.threads, only=args.only, test=not args.no_test,
                                               progress=prog if args.json_progress else None,
                                               gpu_test_only=args.gpu_test_only)})
        elif args.cmd == "probe":
            from .video_io import probe
            info = probe(args.video, find_ffmpeg(args.ffmpeg))
            _emit({"ok": True, "path": info.path, "width": info.width, "height": info.height, "fps": info.fps_float,
                   "duration_s": round(info.duration, 3), "frame_count": info.frame_count,
                   "audio_codec": info.audio_codec})
        elif args.cmd == "bridge":
            from .bridge import bridge_project
            pdir = args.project or default_project_dir(args.video)
            proj = Project.load(pdir)
            nb = bridge_project(proj, args.video, progress=prog)
            proj.data["analysis"]["bridge_boxes"] = nb
            proj.save()
            _emit({"ok": True, "bridge_boxes": nb})
        elif args.cmd in ("render", "preview"):
            from .renderer import render, preview_frame
            ff = find_ffmpeg(args.ffmpeg)
            pdir = args.project or default_project_dir(args.video)
            proj = Project.load(pdir)
            kw = dict(keep_track_conf=args.keep_conf, keep_det_sim=args.keep_det_sim, extend_s=args.extend,
                      smooth_s=args.smooth, max_gap_s=args.max_gap)
            if args.cmd == "render" and args.mode == "blur-selected" and not args.persons:
                raise ValueError("blur-selected needs --persons")   # preview: no persons = untouched frame
            if args.cmd == "render":
                out = args.output or args.video.with_name(args.video.stem + "_blurred.mp4")
                res = render(args.video, proj, out, ff, mode=args.mode, persons=args.persons, style=args.style,
                             strength=args.strength, padding=args.padding, crf=args.crf, preset=args.preset,
                             progress=prog, qa=not args.no_qa, qa_every=args.qa_every, backend=args.backend,
                             threads=args.threads, models_dir=args.models,
                             encoder=None if args.encoder == "auto" else args.encoder, **kw)
                _emit({"ok": True, **res, "device": used_summary()})
            else:
                import cv2, numpy as np
                o, b = preview_frame(args.video, proj, args.frame, ff, mode=args.mode, persons=args.persons,
                                     style=args.style, strength=args.strength, padding=args.padding, **kw)
                img = np.hstack([o, b]) if args.side_by_side else b
                args.output.parent.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(args.output.suffix or ".png", img)
                args.output.write_bytes(buf.tobytes())
                res = {"ok": True, "output": str(args.output), "frame": args.frame,
                       "time_s": round(args.frame / proj.data["video"]["fps_float"], 3)}
                if args.original_out:
                    args.original_out.parent.mkdir(parents=True, exist_ok=True)
                    ok, buf = cv2.imencode(args.original_out.suffix or ".png", o)
                    args.original_out.write_bytes(buf.tobytes())
                    res["original"] = str(args.original_out)
                _emit(res)
        elif args.cmd == "qa":
            from .qa import run_qa
            ff = find_ffmpeg(args.ffmpeg)
            proj = Project.load(args.project)
            res = run_qa(args.rendered, proj, ff, mode=args.mode, persons=args.persons, original=args.original,
                         every=args.every, det_thresh=args.det_thresh, sharp_thresh=args.sharp,
                         backend=args.backend, threads=args.threads, models_dir=args.models, progress=prog)
            _emit({"ok": True, **res, "device": used_summary()})
        elif args.cmd == "addbox":
            from .manual import add_box
            ff = find_ffmpeg(args.ffmpeg)
            pdir = args.project or default_project_dir(args.video)
            proj = Project.load(pdir)
            box = [float(v) for v in args.box.split(",")]
            if len(box) != 4:
                raise ValueError("--box needs x1,y1,x2,y2")
            res = add_box(args.video, proj, ff, person=args.person, frame=args.frame, box=box,
                          max_s=args.max_seconds)
            proj.save()
            _emit({"ok": True, **res, **_project_payload(proj)})
        else:
            proj = Project.load(args.project)
            if args.cmd == "tracks":
                _emit({"ok": True, "project": str(Path(proj.root).resolve()),
                       "tracks": _tracks_payload(proj, args.person)})
                return 0
            if args.cmd == "persons":
                pass
            elif args.cmd == "merge":
                proj.merge(args.ids); proj.save()
            elif args.cmd == "split":
                proj.split(args.tracks); proj.save()
            elif args.cmd == "assign":
                proj.assign(args.tracks, args.person); proj.save()
            elif args.cmd == "rename":
                proj.rename(args.id, args.name); proj.save()
            elif args.cmd == "recluster":
                proj.recluster(args.threshold); proj.save()
            _emit({"ok": True, **_project_payload(proj)})
    except Exception as e:  # one JSON error line for the UI; traceback to stderr
        import traceback
        try:
            traceback.print_exc(file=sys.stderr)
        except (OSError, ValueError, AttributeError):
            pass
        _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
