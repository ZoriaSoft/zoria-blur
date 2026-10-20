"""Video decode/encode through ffmpeg pipes (Windows-friendly, no shell=True).

ffmpeg lookup order: explicit path > ZORIA_FFMPEG env > <app>/bin/ffmpeg(.exe) (CUDA engine: also
../engine/_internal/bin)
> imageio-ffmpeg (if installed) > PATH. ffprobe is optional (looked up next to
ffmpeg or on PATH); without it OpenCV is used to probe.

Decoding always goes through ffmpeg's `fps` filter, so frames are constant
frame rate and frame N in `analyze` is exactly frame N in `render` even for
variable-frame-rate phone videos. ffmpeg also applies rotation metadata.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

_EXE = ".exe" if os.name == "nt" else ""
_NOWIN = {"creationflags": 0x08000000} if os.name == "nt" else {}   # CREATE_NO_WINDOW


def find_ffmpeg(explicit: str | None = None) -> str:
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("ZORIA_FFMPEG"):
        cands.append(os.environ["ZORIA_FFMPEG"])
    from .paths import resource_dirs        # <app>/bin, or the main engine's for the CUDA engine
    cands += [str(d / "bin" / f"ffmpeg{_EXE}") for d in resource_dirs()]
    for c in cands:
        if c and Path(c).is_file():
            return str(Path(c))
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    w = shutil.which("ffmpeg")
    if w:
        return w
    raise FileNotFoundError("ffmpeg not found. Pass --ffmpeg PATH, set ZORIA_FFMPEG, put ffmpeg in ./bin, "
                            "or `pip install imageio-ffmpeg`.")


def find_ffprobe(ffmpeg: str) -> str | None:
    p = Path(ffmpeg).with_name(f"ffprobe{_EXE}")
    if p.is_file():
        return str(p)
    return shutil.which("ffprobe")


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: str            # rational string e.g. "30000/1001"
    fps_float: float
    duration: float
    frame_count: int    # estimated (duration * fps); exact count is recorded after analyze
    audio_codec: str | None

    def to_dict(self):
        return asdict(self)


def probe(path: Path, ffmpeg: str) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ffprobe = find_ffprobe(ffmpeg)
    if ffprobe:
        out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
                              "-show_format", str(path)], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, **_NOWIN)
        if out.returncode == 0:
            j = json.loads(out.stdout)
            v = next(s for s in j["streams"] if s.get("codec_type") == "video")
            a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
            w, h = int(v["width"]), int(v["height"])
            rot = 0
            for sd in v.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(float(sd["rotation"]))
            if "rotate" in (v.get("tags") or {}):
                rot = int(v["tags"]["rotate"])
            if abs(rot) % 180 == 90:
                w, h = h, w
            fr = v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1"
            if fr in ("0/0", "0/1"):
                fr = v.get("r_frame_rate", "30/1")
            fps = Fraction(fr).limit_denominator(100000)
            dur = float(j.get("format", {}).get("duration") or v.get("duration") or 0)
            return VideoInfo(str(path), w, h, f"{fps.numerator}/{fps.denominator}", float(fps), dur,
                             int(round(dur * float(fps))), a.get("codec_name") if a else None)
    # Fallback: OpenCV (applies orientation by default on recent builds)
    import cv2
    cap = cv2.VideoCapture(str(path))
    ok, fr = cap.read()
    if not ok:
        raise RuntimeError(f"cannot decode {path}")
    fpsf = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    fps = Fraction(fpsf).limit_denominator(1001)
    return VideoInfo(str(path), fr.shape[1], fr.shape[0], f"{fps.numerator}/{fps.denominator}", float(fps),
                     n / fpsf if fpsf else 0.0, n, "unknown")


class FrameReader:
    """Iterates BGR uint8 frames at constant fps through an ffmpeg pipe."""

    def __init__(self, info: VideoInfo, ffmpeg: str, start_frame: int = 0):
        """start_frame > 0 seeks (input-side, frame-accurate on CFR sources): the first frame yielded is
        frame `start_frame` of the CFR sequence. Used by the one-frame preview; full passes start at 0."""
        self.info = info
        self.w, self.h = info.width, info.height
        seek = ["-ss", f"{start_frame / info.fps_float:.6f}"] if start_frame > 0 else []
        cmd = [ffmpeg, "-v", "error", "-nostdin", *seek, "-i", info.path, "-map", "0:v:0", "-an", "-sn",
               "-vf", f"fps={info.fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     bufsize=self.w * self.h * 3 * 2, **_NOWIN)
        self.nbytes = self.w * self.h * 3

    def __iter__(self):
        try:
            while True:
                buf = self.proc.stdout.read(self.nbytes)
                if len(buf) < self.nbytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        finally:
            self.close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def stream_rgba(video: Path, ffmpeg: str, start_frame: int, width: int, height: int):
    """Raw RGBA frames (width x height) of `video` to stdout, from CFR frame `start_frame` on, with the same
    fps filter and input-side seek as FrameReader, so frame k of the stream is frame start_frame + k of the
    renderer. Nothing else is written to stdout. Used by the app's live preview player."""
    info = probe(Path(video), ffmpeg)
    w, h = max(2, width // 2 * 2), max(2, height // 2 * 2)
    seek = ["-ss", f"{start_frame / info.fps_float:.6f}"] if start_frame > 0 else []
    cmd = [ffmpeg, "-v", "error", "-nostdin", *seek, "-i", info.path, "-map", "0:v:0", "-an", "-sn",
           "-vf", f"fps={info.fps},scale={w}:{h}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                            bufsize=w * h * 4 * 2, **_NOWIN)
    out = sys.stdout.buffer
    try:
        while True:
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            out.write(buf)
        out.flush()
    except (BrokenPipeError, OSError):
        pass                                  # the app stopped reading (pause / seek / close)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# audio codecs that can be stream-copied into mp4/mov
_MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}


class FrameWriter:
    """Encodes BGR frames with libx264 and muxes the ORIGINAL audio from the source file."""

    def __init__(self, info: VideoInfo, out_path: Path, ffmpeg: str, crf: int = 18,
                 preset: str = "veryfast", log_path: Path | None = None):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        acodec = info.audio_codec
        if acodec is None:
            a_args = []
        elif ext in (".mp4", ".m4v", ".mov") and acodec not in _MP4_AUDIO_OK and acodec != "unknown":
            a_args = ["-c:a", "aac", "-b:a", "256k"]     # codec cannot live in mp4: re-encode
        else:
            a_args = ["-c:a", "copy"]
        cmd = [ffmpeg, "-v", "error", "-y", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
               "-r", info.fps, "-i", "-",
               "-i", info.path,
               "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
               *a_args]
        if ext in (".mp4", ".m4v", ".mov"):
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(out_path))
        self._log = open(log_path, "wb") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log, **_NOWIN)
        self.out_path = out_path

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> int:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        return rc
