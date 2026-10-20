"""Where the bundled resources (models/, bin/ffmpeg) live.

Source tree: the repo root. PyInstaller build: its _internal folder (sys._MEIPASS). The optional CUDA engine
(<app>/engine-cuda) ships without models and ffmpeg and borrows them from the main engine next to it
(<app>/engine/_internal), so the CUDA package does not carry a second 250 MB copy of them."""
from __future__ import annotations

import sys
from pathlib import Path


def resource_dirs() -> list[Path]:
    """Folders to look in for models/ and bin/, in order."""
    own = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    out = [own]
    if getattr(sys, "frozen", False):
        app = Path(sys.executable).resolve().parent.parent       # <app>\engine-cuda\x.exe -> <app>
        sib = app / "engine" / "_internal"
        if sib.is_dir() and sib.resolve() != own.resolve():
            out.append(sib)
    return out


def find_resource(rel: str) -> Path | None:
    """First existing <dir>/<rel> over resource_dirs()."""
    for d in resource_dirs():
        p = d / rel
        if p.exists():
            return p
    return None
