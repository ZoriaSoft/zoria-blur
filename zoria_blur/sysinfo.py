"""Plain facts about this PC for the app's system check: processor (name, cores), memory, and a hardware
fingerprint the app keeps with its saved device choice (a changed fingerprint = re-check the devices).

No extra packages and no subprocess: Windows via winreg / ctypes, Linux via /proc. Never raises."""
from __future__ import annotations

import os
import platform


def cpu_info() -> dict:
    """{'name': str|None, 'cores': int|None (physical), 'threads': int|None (logical)}"""
    name, cores = None, None
    threads = os.cpu_count()
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            cores = _win_physical_cores()
        elif os.path.exists("/proc/cpuinfo"):
            ids = set()
            phys = core = None
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k == "model name" and not name:
                        name = v
                    elif k == "physical id":
                        phys = v
                    elif k == "core id":
                        core = v
                    elif not k and core is not None:
                        ids.add((phys, core)); phys = core = None
            if core is not None:
                ids.add((phys, core))
            cores = len(ids) or None
    except Exception:  # noqa: BLE001
        pass
    if not name:
        name = platform.processor() or None
    if name:
        name = " ".join(name.split())          # registry names often carry padding spaces
    return {"name": name, "cores": cores, "threads": threads}


def _win_physical_cores() -> int | None:
    import ctypes

    class SLPI(ctypes.Structure):           # SYSTEM_LOGICAL_PROCESSOR_INFORMATION (x64: 32 bytes)
        _fields_ = [("mask", ctypes.c_size_t), ("rel", ctypes.c_int), ("u", ctypes.c_ubyte * 16)]

    k32 = ctypes.windll.kernel32
    n = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
    if n.value == 0:
        return None
    count = n.value // ctypes.sizeof(SLPI)
    buf = (SLPI * count)()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
        return None
    return sum(1 for e in buf if e.rel == 0) or None       # RelationProcessorCore


def memory_gb() -> float | None:
    """Installed memory the OS sees, GB (1 decimal)."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            m = MS(); m.len = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return round(m.total / 2**30, 1)
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def hardware_adapters(gpus_fallback: list[str] | None = None) -> list[str]:
    """Names of the real (hardware) display adapters, sorted. Windows: DXGI (fast, no WMI); else the given names."""
    try:
        from .dxgi import adapters
        names = sorted({a["name"] for a in adapters() if not a.get("software") and a.get("name")})
        if names:
            return names
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(gpus_fallback or []))


# Oldest NVIDIA driver the shipped CUDA 12 runtime runs on (CUDA 12.0 GA, Windows; minor version compatibility).
MIN_NVIDIA_DRIVER = (527, 41)


def nvidia_driver_version(windows_version: str) -> str | None:
    """NVIDIA's own driver number from the Windows driver version: '32.0.15.6094' -> '560.94' (the last five digits
    of the third and fourth parts). None if it does not look like one."""
    parts = windows_version.strip().split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    digits = (parts[2] + parts[3].zfill(4))[-5:]
    if len(digits) < 5:
        return None
    return f"{int(digits[:3])}.{digits[3:]}"


def nvidia_driver() -> str | None:
    """Installed NVIDIA display driver as NVIDIA numbers it ('560.94'), from the display adapter class in the registry
    (no admin, no subprocess). None when there is no NVIDIA driver or it can't be read."""
    if os.name != "nt":
        return None
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        best = None
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as cls:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(cls, i)
                except OSError:
                    break
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(cls, sub) as k:
                        prov = str(winreg.QueryValueEx(k, "ProviderName")[0])
                        ver = str(winreg.QueryValueEx(k, "DriverVersion")[0])
                except OSError:
                    continue
                if "nvidia" not in prov.lower():
                    continue
                v = nvidia_driver_version(ver)
                if v and (best is None or driver_tuple(v) > driver_tuple(best)):
                    best = v
        return best
    except Exception:  # noqa: BLE001
        return None


def driver_tuple(v: str) -> tuple[int, int]:
    a, _, b = v.partition(".")
    try:
        return int(a), int(b or 0)
    except ValueError:
        return 0, 0


def nvidia_driver_too_old(v: str | None) -> bool:
    return v is not None and driver_tuple(v) < MIN_NVIDIA_DRIVER


def fingerprint(cpu: dict, adapters: list[str]) -> str:
    """Changes when the processor or a display adapter changes (not with drivers or settings)."""
    return "|".join([cpu.get("name") or "?", *adapters])
