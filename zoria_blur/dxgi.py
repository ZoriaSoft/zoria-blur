"""DXGI adapter list for DirectML (Windows only, ctypes, no extra packages).

ONNX Runtime's DmlExecutionProvider takes `device_id` = the adapter index of IDXGIFactory1::EnumAdapters1, and
uses adapter 0 when none is given. On a dual-GPU PC adapter 0 is usually the integrated GPU the monitor hangs on
(on an NVIDIA test PC the integrated Intel GPU came first), so the engine picks the adapter itself:
  1. ZORIA_DML_DEVICE=<index> (or the engine's --dml-device) when set: the user's choice;
  2. else the first hardware adapter in Windows' high-performance order (IDXGIFactory6::EnumAdapterByGpuPreference);
  3. else (older Windows) the hardware adapter with the most dedicated video memory.
Software adapters (Microsoft Basic Render Driver) are never picked.
"""
from __future__ import annotations

import os

_IID_FACTORY1 = "{770aae78-f26f-4dba-a829-253c83d1b387}"
_IID_FACTORY6 = "{c1b6694f-ff09-44a9-b03c-77900a0a1d17}"
_IID_ADAPTER1 = "{29038f61-3839-4626-91fd-086879011a05}"
_SOFTWARE = 2                      # DXGI_ADAPTER_FLAG_SOFTWARE
_HIGH_PERFORMANCE = 2              # DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
_NOT_FOUND = 0x887A0002            # DXGI_ERROR_NOT_FOUND

_cache: dict = {}


def adapters() -> list[dict]:
    """[{index, name, vendor_id, dedicated_mb, software, luid, perf_rank}] in EnumAdapters1 order; [] off Windows
    or on any error. perf_rank = position in the high-performance order (None if Windows can't tell)."""
    if "list" not in _cache:
        try:
            _cache["list"] = _enumerate() if os.name == "nt" else []
        except Exception:  # noqa: BLE001 - never let adapter discovery take the engine down
            _cache["list"] = []
    return _cache["list"]


def _enumerate() -> list[dict]:
    import ctypes
    from ctypes import POINTER, byref, c_uint, c_void_p, wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", c_uint), ("DeviceId", c_uint),
                    ("SubSysId", c_uint), ("Revision", c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("AdapterLuid", LUID), ("Flags", c_uint)]

    ole32 = ctypes.OleDLL("ole32")

    def guid(s):
        g = GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(s), byref(g))
        return g

    HR = ctypes.HRESULT

    def method(obj, index, *argtypes):
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, *argtypes)(vtbl[index])

    def release(obj):
        if obj:
            method(obj, 2)(obj)

    dxgi = ctypes.WinDLL("dxgi")
    dxgi.CreateDXGIFactory1.argtypes = [POINTER(GUID), POINTER(c_void_p)]
    dxgi.CreateDXGIFactory1.restype = HR
    fac = c_void_p()
    dxgi.CreateDXGIFactory1(byref(guid(_IID_FACTORY1)), byref(fac))

    def desc(adapter):
        d = DESC1()
        if method(adapter, 10, POINTER(DESC1))(adapter, byref(d)) != 0:       # IDXGIAdapter1::GetDesc1
            raise OSError("GetDesc1 failed")
        return d

    out = []
    try:
        i = 0
        enum1 = method(fac, 12, c_uint, POINTER(c_void_p))                    # IDXGIFactory1::EnumAdapters1
        while i < 16:
            a = c_void_p()
            if enum1(fac, i, byref(a)) & 0xFFFFFFFF == _NOT_FOUND or not a:
                break
            try:
                d = desc(a)
                out.append({"index": i, "name": d.Description.strip(), "vendor_id": d.VendorId,
                            "dedicated_mb": int(d.DedicatedVideoMemory // (1024 * 1024)),
                            "software": bool(d.Flags & _SOFTWARE),
                            "luid": (int(d.AdapterLuid.high), int(d.AdapterLuid.low)), "perf_rank": None})
            finally:
                release(a)
            i += 1
        # high-performance order (Windows 10 1803+): IDXGIFactory6::EnumAdapterByGpuPreference
        f6 = c_void_p()
        if method(fac, 0, POINTER(GUID), POINTER(c_void_p))(fac, byref(guid(_IID_FACTORY6)), byref(f6)) == 0 and f6:
            try:
                byp = method(f6, 29, c_uint, c_uint, POINTER(GUID), POINTER(c_void_p))
                iid_a1 = guid(_IID_ADAPTER1)
                for r in range(16):
                    a = c_void_p()
                    if byp(f6, r, _HIGH_PERFORMANCE, byref(iid_a1), byref(a)) != 0 or not a:
                        break
                    try:
                        l = desc(a).AdapterLuid
                        for e in out:
                            if e["luid"] == (int(l.high), int(l.low)) and e["perf_rank"] is None:
                                e["perf_rank"] = r
                    finally:
                        release(a)
            finally:
                release(f6)
    finally:
        release(fac)
    return out


def pick(adapter_list: list[dict] | None = None) -> dict | None:
    """The adapter DirectML should use (see module doc), or None = let ONNX Runtime use its default (adapter 0)."""
    ads = adapters() if adapter_list is None else adapter_list
    env = os.environ.get("ZORIA_DML_DEVICE", "").strip()
    if env:
        try:
            want = int(env)
        except ValueError:
            want = None
        hit = next((a for a in ads if a["index"] == want), None)
        if hit:
            return dict(hit, how="manual")
        if want is not None and not ads:
            return {"index": want, "name": None, "how": "manual"}       # can't enumerate: trust the user
    hw = [a for a in ads if not a["software"]]
    if not hw:
        return None
    ranked = [a for a in hw if a["perf_rank"] is not None]
    if ranked:
        return dict(min(ranked, key=lambda a: a["perf_rank"]), how="high_performance")
    return dict(max(hw, key=lambda a: (a["dedicated_mb"], -a["index"])), how="most_vram")
