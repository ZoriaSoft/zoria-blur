"""UI contract checks (needs ffmpeg, no models): python tests/test_cli_ui.py

Covers the JSON commands the Windows app calls: probe, persons / tracks / rename / merge / split payloads,
preview (with --original-out, and blur-selected with no persons = untouched frame) and the seeking
FrameReader the preview and addbox use."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from zoria_blur.video_io import FrameReader, find_ffmpeg, probe  # noqa: E402

FF = find_ffmpeg(None)


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "zoria_blur", *map(str, args)], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines, r.stderr
    return json.loads(lines[-1]), r


def make_video(d: Path) -> Path:
    v = d / "clip.mp4"
    subprocess.run([FF, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "50", str(v)], check=True)
    return v


def make_project(d: Path, video: Path) -> Path:
    p = d / "clip_zoria"
    (p / "crops").mkdir(parents=True)
    import cv2
    for tid in (1, 2, 3):
        cv2.imwrite(str(p / "crops" / f"track_{tid:04d}_f000010.jpg"), np.full((40, 40, 3), 60 * tid, np.uint8))
    def track(tid, pid, f0, f1, x):
        fr = list(range(f0, f1, 2))
        return {"id": tid, "person_id": pid, "frames": fr, "boxes": [[x, 60, x + 60, 140]] * len(fr),
                "scores": [0.9] * len(fr), "mean_score": 0.9, "confidence": 0.0,
                "crops": [{"file": f"track_{tid:04d}_f000010.jpg", "frame": 10, "quality": 1.0}]}
    data = {"schema_version": 1,
            "video": {"path": str(video), "width": 320, "height": 240, "fps": "25/1", "fps_float": 25.0,
                      "duration": 4.0, "frame_count": 100, "audio_codec": None},
            "analysis": {"params": {"every": 2}, "risks": [{"type": "occlusion", "start_frame": 20, "end_frame": 30,
                                                           "start_s": 0.8, "end_s": 1.2, "persons": [1]}]},
            "persons": [], "edits": [],
            "tracks": [track(1, 1, 0, 50, 40), track(2, 1, 50, 100, 40), track(3, 2, 0, 100, 200)]}
    (p / "project.json").write_text(json.dumps(data), encoding="utf-8")
    from zoria_blur.project import Project
    pr = Project.load(p)
    pr.rebuild_persons()
    pr.save()
    return p


def test_seek_matches_sequential(video):
    info = probe(video, FF)
    allf = [f.copy() for f in FrameReader(info, FF)]
    for idx in (1, 13, 49, 50, 77, len(allf) - 1):
        st = max(0, idx - 3)
        got = next(f for k, f in enumerate(FrameReader(info, FF, start_frame=st)) if st + k == idx)
        assert np.array_equal(got, allf[idx]), f"seek landed off frame {idx}"


def test_probe(video):
    j, _ = cli("probe", video)
    assert j["ok"] and j["width"] == 320 and abs(j["duration_s"] - 4.0) < 0.1 and j["frame_count"] == 100, j


def _risks_current(j, proj):
    """Every project payload carries the risky moments of the *current* persons, the same list the QA after an
    export recomputes (0.1.7: the app's done panel and timeline count one list)."""
    from zoria_blur.project import Project
    from zoria_blur.risks import compute_risks
    want = compute_risks(Project.load(proj))
    assert j["risks"] == json.loads(json.dumps(want)), (j["risks"], want)
    ids = {p["id"] for p in j["persons"]}
    assert all(set(r["persons"]) <= ids for r in j["risks"]), "risk names a person that no longer exists"


def test_devices_system_check():
    """devices --json-progress: live steps for the app's system check, then the choice + test run + fingerprint;
    --only cpu --no-test is the quiet start-up check (no WMI, no test run)."""
    r = subprocess.run([sys.executable, "-m", "zoria_blur", "--json-progress", "devices"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    objs = [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    evs = [o for o in objs if o.get("event") == "probe"]
    res = objs[-1]
    assert res["ok"], r.stderr[-500:]
    items = [e["item"] for e in evs]
    for it in ("cpu", "memory", "gpus", "cuda", "dml", "cpu_backend", "test"):
        assert it in items, (it, items)
    assert items.index("cpu") < items.index("gpus") < items.index("cuda") < items.index("dml") < items.index("test")
    for it in ("cpu", "memory", "gpus", "cuda", "dml", "cpu_backend"):
        st = [e["status"] for e in evs if e["item"] == it]
        assert st[0] == "checking" and st[-1] != "checking", (it, st)
    assert res["chosen"] in ("cuda", "dml", "cpu")
    assert res["bench"] and res["bench"]["backend"] == res["chosen"] and res["bench"]["ms"] > 0, res["bench"]
    assert res["system"]["cpu"]["threads"] and res["system"]["fingerprint"]
    q, _ = cli("devices", "--only", "cpu", "--no-test")
    assert q["ok"] and q["quick"] and q["chosen"] == "cpu" and q["bench"] is None
    assert q["system"]["fingerprint"] == res["system"]["fingerprint"], "quick check must see the same hardware"
    g, _ = cli("devices", "--gpu-test-only")
    assert g["ok"] and (g["bench"] is None or g["bench"]["backend"] != "cpu"), g["bench"]


def test_project_commands(video, proj):
    j, _ = cli("persons", proj)
    assert j["ok"] and Path(j["project"]).is_dir()
    assert j["video"]["duration_s"] == 4.0
    _risks_current(j, proj)
    ps = {p["id"]: p for p in j["persons"]}
    assert set(ps) == {1, 2} and ps[1]["seconds_visible"] > 0
    assert all((Path(j["project"]) / t).is_file() for t in ps[1]["thumbnails"])
    j, _ = cli("tracks", proj, "--person", 1)
    assert [t["id"] for t in j["tracks"]] == [1, 2]
    assert all((Path(j["project"]) / t["thumbnail"]).is_file() for t in j["tracks"])
    j, _ = cli("rename", proj, 2, "Ayşe Yılmaz")
    assert {p["id"]: p["name"] for p in j["persons"]}[2] == "Ayşe Yılmaz"
    j, _ = cli("split", proj, "--tracks", 2)
    assert len(j["persons"]) == 3
    new = max(p["id"] for p in j["persons"])
    _risks_current(j, proj)
    j, _ = cli("merge", proj, 1, new)
    assert sorted(p["id"] for p in j["persons"]) == [1, 2]
    _risks_current(j, proj)
    assert {p["id"]: p["name"] for p in j["persons"]}[2] == "Ayşe Yılmaz"


def test_addbox_gets_a_picture(video, proj):
    """A person made only of a hand-marked face still gets a card picture (0.1.9 showed an empty card)."""
    j, _ = cli("addbox", video, "-p", proj, "--person", 0, "--frame", 40, "--box", "120,60,180,140")
    assert j["ok"], j
    j, _ = cli("persons", proj)
    new = max(j["persons"], key=lambda p: p["id"])
    assert new["thumbnails"], new
    assert all((Path(j["project"]) / t).is_file() for t in new["thumbnails"])


def test_render_encoder_fallback(video, proj, d):
    """A graphics-card encoder that fails on the real video never leaves a broken file: the export is redone with
    x264 (here NVENC is forced on whatever PC runs the test; without NVIDIA it fails at once)."""
    import cv2
    from zoria_blur.project import Project
    from zoria_blur.renderer import render
    from zoria_blur.video_io import encoder_works
    out = d / "enc_test.mp4"
    res = render(video, Project.load(proj), out, FF, mode="blur-except", persons=[], qa=False, encoder="nvenc")
    assert out.is_file() and res["frames"] == 100, res
    if encoder_works(FF, "nvenc"):
        assert res["encoder"] == "nvenc" and "encoder_fallback" not in res, res
    else:
        assert res["encoder"] == "x264" and res["encoder_fallback"]["from"] == "nvenc", res
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 100
    cap.release()
    j, _ = cli("render", video, "-p", proj, "-o", d / "enc_cli.mp4", "--mode", "blur-except", "--no-qa")
    assert j["ok"] and j["encoder"] in ("x264", "nvenc", "qsv", "amf"), j


def test_preview(video, proj, d):
    out, org = d / "prev.png", d / "orig.png"
    j, r = cli("preview", video, "-p", proj, "--frame", 60, "--mode", "blur-selected", "--persons", 2,
               "-o", out, "--original-out", org)
    assert j["ok"], r.stderr
    import cv2
    a, b = cv2.imread(str(org)), cv2.imread(str(out))
    assert a.shape == b.shape == (240, 320, 3)
    assert np.abs(a[60:140, 200:260].astype(int) - b[60:140, 200:260]).mean() > 3, "person 2 not blurred"
    assert np.abs(a[60:140, 40:100].astype(int) - b[60:140, 40:100]).mean() < 1, "person 1 blurred"
    j, r = cli("preview", video, "-p", proj, "--frame", 60, "--mode", "blur-selected", "-o", out)
    assert j["ok"], r.stderr
    assert np.array_equal(cv2.imread(str(out)), a), "no persons selected must give the untouched frame"
    j, _ = cli("preview", video, "-p", proj, "--frame", 999, "--mode", "blur-except", "-o", out)
    assert not j["ok"] and "frame" in j["error"]


def test_live_preview(video, proj, d):
    """plan (blur ellipses per frame) + frames (raw RGBA, same CFR indices as the renderer)."""
    out = d / "plan.json"
    j, r = cli("plan", proj, "-o", out, "--mode", "blur-selected", "--persons", 2)
    assert j["ok"], r.stderr
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["frame_count"] == 100 and plan["fps"] == 25.0 and plan["persons"] == [2]
    cx, cy, rx, ry = plan["frames"]["60"][0]
    assert 200 < cx < 260 and rx > 30 and ry > 40          # person 2's face box 200-260 x 60-140, padded
    assert all(200 - 80 < c[0] < 260 + 80 for regs in plan["frames"].values() for c in regs), "person 1 in plan"
    j, _ = cli("plan", proj, "-o", out, "--mode", "blur-selected")
    assert j["ok"] and json.loads(out.read_text(encoding="utf-8"))["frames"] == {}, "nobody selected = no blur"
    j, _ = cli("plan", proj, "-o", out, "--mode", "blur-except")
    assert len(json.loads(out.read_text(encoding="utf-8"))["frames"]["60"]) == 2, "blur-except none = everyone"
    sys.path.insert(0, str(ROOT / "tests"))
    from eval_preview_parity import check
    from zoria_blur.project import Project
    pr = Project.load(proj)
    for mode, pers in (("blur-selected", [2]), ("blur-except", [1]), ("blur-except", [])):
        n, bad, worst = check(pr, mode, pers)
        assert n > 0 and bad == 0, f"{mode} {pers}: export blurs outside the live preview ({worst} px)"
    # frames: start 60, 160x120 RGBA, 40 frames left; frame 0 of the stream = renderer frame 60
    r = subprocess.run([sys.executable, "-m", "zoria_blur", "frames", str(video), "--start", "60", "--width", "160",
                        "--height", "120"], cwd=ROOT, capture_output=True)
    fb = 160 * 120 * 4
    assert r.returncode == 0 and len(r.stdout) == 40 * fb, (len(r.stdout), r.stderr[-300:])
    import cv2
    info = probe(video, FF)
    ref = next(f for k, f in enumerate(FrameReader(info, FF)) if k == 60)
    got = np.frombuffer(r.stdout[:fb], np.uint8).reshape(120, 160, 4)[..., :3][..., ::-1]
    small = cv2.resize(ref, (160, 120), interpolation=cv2.INTER_AREA)
    assert np.abs(got.astype(int) - small).mean() < 8, "frames stream is not frame 60"


def test_closed_stdout_exits_quietly():
    """The app closed / cancelled the job: writing the result to its closed pipe must end the engine quietly
    (an escaping OSError made the windowed exe show PyInstaller's modal "Unhandled exception" box, 0.1.6)."""
    import os
    from zoria_blur import cli

    class Dead:
        def write(self, _):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

    class Exited(Exception):
        pass

    def fake_exit(code):
        raise Exited(code)

    old_out, old_exit = sys.stdout, os._exit
    sys.stdout, os._exit = Dead(), fake_exit
    try:
        cli._emit({"ok": True})
        raise AssertionError("_emit did not stop")
    except Exited as e:
        assert e.args[0] == 3
    finally:
        sys.stdout, os._exit = old_out, old_exit
    # and for real: the engine's stdout is closed before it writes its result
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.Popen([sys.executable, "packaging/engine_entry.py", "devices"], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    r.stdout.close()
    err = r.stderr.read().decode("utf-8", "replace")
    r.wait(120)
    assert "Traceback" not in err, err[-500:]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        v = make_video(d)
        p = make_project(d, v)
        test_seek_matches_sequential(v); print("ok seek")
        test_probe(v); print("ok probe")
        test_devices_system_check(); print("ok devices (system check events, test run, fingerprint)")
        test_project_commands(v, p); print("ok project commands")
        test_addbox_gets_a_picture(v, p); print("ok addbox picture")
        test_render_encoder_fallback(v, p, d); print("ok render encoder + fallback")
        test_preview(v, p, d); print("ok preview")
        test_live_preview(v, p, d); print("ok live preview (plan + frames)")
        test_closed_stdout_exits_quietly(); print("ok closed stdout = quiet exit")
    print("all UI contract tests passed")
