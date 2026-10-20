"""Which CUDA / cuDNN DLLs does the engine really load? (input for the DLL list in build_cuda_engine.ps1)

Run in .venv-cuda (onnxruntime-gpu + nvidia-*-cu12 wheels) on an NVIDIA machine:
    .venv-cuda\\Scripts\\python.exe packaging\\cuda_dll_probe.py VIDEO.mp4 [WORKDIR]
It runs analyze + render with --backend cuda (all models the app uses), then prints every DLL loaded into the
process from the nvidia wheels or onnxruntime, with its size. Windows only."""
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def loaded_modules() -> list[str]:
    psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
    h = k32.GetCurrentProcess()
    arr = (wt.HMODULE * 4096)()
    need = wt.DWORD()
    psapi.EnumProcessModules(h, arr, ctypes.sizeof(arr), ctypes.byref(need))
    out = []
    for i in range(need.value // ctypes.sizeof(wt.HMODULE)):
        buf = ctypes.create_unicode_buffer(1024)
        psapi.GetModuleFileNameExW(h, arr[i], buf, 1024)
        out.append(buf.value)
    return out


def main():
    video = sys.argv[1]
    work = Path(sys.argv[2] if len(sys.argv) > 2 else "build/cuda_probe")
    work.mkdir(parents=True, exist_ok=True)
    from zoria_blur.cli import main as cli
    proj = work / "proj"
    rc1 = cli(["--backend", "cuda", "analyze", video, "-p", str(proj)])
    rc2 = cli(["--backend", "cuda", "render", video, "-p", str(proj), "--mode", "blur-except",
               "-o", str(work / "out.mp4")])
    mods = [m for m in loaded_modules() if "\\nvidia\\" in m.lower() or "onnxruntime" in m.lower()]
    rows = sorted({(Path(m).name, round(os.path.getsize(m) / 2**20, 1), m) for m in mods})
    print(json.dumps({"rc": [rc1, rc2], "dlls": [{"name": n, "mb": mb, "path": p} for n, mb, p in rows],
                      "total_mb": round(sum(r[1] for r in rows), 1)}, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
