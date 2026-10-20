"""Download the ONNX model weights into ./models (not committed to git).

    python scripts/download_models.py            # det_10g + det_500m + w600k_mbf (default set)
    python scripts/download_models.py --with-2.5g  # also det_2.5g (needs the 280 MB buffalo_m zip)

Person detector: YOLOX (Megvii) official ONNX exports, release 0.1.1rc0. Code AND weights are
Apache-2.0 (commercial use OK). yolox_tiny is the default, yolox_nano (~3x faster) is the CPU-lite option,
yolox_s is optional (--with-yolox-s).

LICENSE: InsightFace pretrained models are for NON-COMMERCIAL research use only.
See README.md before shipping a commercial product.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

# name -> (url, sha256, member-in-zip or None)
SOURCES = {
    "det_10g.onnx": ("https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
                     "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", None),
    "det_500m.onnx": ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
                      "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a", "det_500m.onnx"),
    "w600k_mbf.onnx": ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
                       "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f", "w600k_mbf.onnx"),
    "yolox_tiny.onnx": (_YOLOX + "yolox_tiny.onnx",
                        "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7", None),
    "yolox_nano.onnx": (_YOLOX + "yolox_nano.onnx",
                        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d", None),
}
OPTIONAL_YOLOX_S = {
    "yolox_s.onnx": (_YOLOX + "yolox_s.onnx",
                     "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063", None),
}
OPTIONAL = {
    "det_2.5g.onnx": ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
                      None, "det_2.5g.onnx"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_zip_cache: dict[str, bytes] = {}


def fetch(url: str) -> bytes:
    if url in _zip_cache:
        return _zip_cache[url]
    print(f"  downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "zoria-blur"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if url.endswith(".zip"):
        _zip_cache[url] = data
    return data


def get(name, url, digest, member):
    dst = MODELS / name
    if dst.is_file() and (digest is None or sha256(dst) == digest):
        print(f"ok   {name} (already present)", file=sys.stderr)
        return
    data = fetch(url)
    if member:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            src = next(n for n in z.namelist() if n.endswith(member))
            data = z.read(src)
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(data)
    if digest and sha256(tmp) != digest:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {name}")
    shutil.move(str(tmp), dst)
    print(f"got  {name} ({dst.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-2.5g", dest="with25", action="store_true")
    ap.add_argument("--with-yolox-s", dest="with_s", action="store_true")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    for n, (u, d, m) in SOURCES.items():
        get(n, u, d, m)
    if a.with25:
        for n, (u, d, m) in OPTIONAL.items():
            get(n, u, d, m)
    if a.with_s:
        for n, (u, d, m) in OPTIONAL_YOLOX_S.items():
            get(n, u, d, m)
    print("models ready in", MODELS, file=sys.stderr)


if __name__ == "__main__":
    main()
