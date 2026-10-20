"""ONNX Runtime session factory. Backend is switchable by a single flag.

    cpu  -> CPUExecutionProvider            (default, works everywhere)
    dml  -> DmlExecutionProvider            (Windows DirectML: AMD/Intel/NVIDIA; needs onnxruntime-directml)
    cuda -> CUDAExecutionProvider           (NVIDIA; needs onnxruntime-gpu)
    auto -> first that works of cuda -> dml -> cpu (the user's rule; same order as the app's system check)

If the requested provider is not available the session falls back to CPU and a
warning is printed (never silently pretends to be on GPU).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _cuda_dll_dirs():
    """The CUDA engine build (packaging/build_cuda_engine.ps1) puts the CUDA 12 / cuDNN 9 runtime DLLs next to
    onnxruntime_providers_cuda.dll in _internal/onnxruntime/capi; make sure Windows resolves them from there
    (also cuDNN's own on-demand sub-library loads) and never from some other CUDA install on PATH."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    capi = Path(getattr(sys, "_MEIPASS", "")) / "onnxruntime" / "capi"
    if (capi / "onnxruntime_providers_cuda.dll").is_file():
        os.add_dll_directory(str(capi))
        os.environ["PATH"] = str(capi) + os.pathsep + os.environ.get("PATH", "")


_cuda_dll_dirs()

import onnxruntime as ort  # noqa: E402

if (os.name == "nt" and not getattr(sys, "frozen", False) and hasattr(ort, "preload_dlls")
        and "CUDAExecutionProvider" in ort.get_available_providers()):
    # dev venv with onnxruntime-gpu + nvidia-*-cu12 wheels: load their DLLs from site-packages
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001 - no CUDA runtime: the CUDA probe fails and the chain moves on
        pass

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}
_EP = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}
_ORDER = ("cuda", "dml", "cpu")          # the user's rule: NVIDIA/CUDA, else DirectML, else CPU

BACKENDS = ("cpu", "dml", "cuda", "auto")


def _ep_list(b: str) -> list:
    """Providers for an InferenceSession. CUDA: cuDNN conv algorithm search per new input shape; the default
    EXHAUSTIVE benchmarks every algorithm (seconds per shape, and SCRFD gets a new shape per video size),
    HEURISTIC picks from cuDNN's heuristics. ZORIA_CUDA_CONV_SEARCH=EXHAUSTIVE|HEURISTIC|DEFAULT overrides."""
    if b == "cuda":
        algo = os.environ.get("ZORIA_CUDA_CONV_SEARCH", "HEURISTIC").upper()
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"]
    if b == "dml":
        a = dml_adapter()
        if a is not None:
            return [("DmlExecutionProvider", {"device_id": str(a["index"])}), "CPUExecutionProvider"]
    return list(_PROVIDERS[b])


def dml_adapter() -> dict | None:
    """The DXGI adapter DirectML runs on: ZORIA_DML_DEVICE (--dml-device) if set, else the high-performance /
    most-VRAM hardware GPU (dxgi.pick). None = ONNX Runtime's default adapter."""
    from .dxgi import pick
    return pick()

# which provider each loaded model really runs on, and every fallback that happened (reported to the UI)
USED: dict[str, str] = {}
FALLBACKS: list[dict] = []


def _disabled(b: str) -> bool:
    """ZORIA_DISABLE_CUDA=1 / ZORIA_DISABLE_DML=1 hide a provider (tests, or a user with a broken driver)."""
    return os.environ.get(f"ZORIA_DISABLE_{b.upper()}", "") not in ("", "0")


def available_backends() -> list[str]:
    avail = set(ort.get_available_providers())
    return [b for b in _ORDER if _EP[b] in avail and not _disabled(b)]


def backend_chain(backend: str) -> list[str]:
    """Backends to try in order. Every chain ends with cpu, so nothing ever fails for lack of a GPU."""
    backend = (backend or "cpu").lower()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}', choose one of {BACKENDS}")
    avail = available_backends()
    if backend == "auto":
        return avail if "cpu" in avail else avail + ["cpu"]
    chain = [b for b in _ORDER[_ORDER.index(backend):] if b in avail]
    if backend not in avail and backend != "cpu":
        print(f"[zoria-blur] warning: {_EP[backend]} not available in this onnxruntime build "
              f"(available: {sorted(ort.get_available_providers())}); falling back", file=sys.stderr)
    return chain if chain and chain[-1] == "cpu" else chain + ["cpu"]


def resolve_providers(backend: str) -> list[str]:
    return list(_PROVIDERS[backend_chain(backend)[0]])


_MODEL_BYTES: dict = {}


def _model_source(model_path: Path, b: str):
    """What InferenceSession loads. GPU backends get the model with free output shapes (onnx_shapes.py): the
    InsightFace exports declare 640x640 / batch-1 output shapes that DirectML takes literally (every other SCRFD
    input size failed in the last Reshape, ArcFace batches > 1 too). CPU loads the file unchanged."""
    if b == "cpu":
        return str(model_path)
    key = str(model_path)
    if key not in _MODEL_BYTES:
        raw = model_path.read_bytes()
        try:
            from .onnx_shapes import free_output_shapes
            _MODEL_BYTES[key] = free_output_shapes(raw)
        except Exception as e:  # noqa: BLE001 - unexpected protobuf: load the file as it is
            print(f"[zoria-blur] warning: could not free output shapes of {model_path.name}: {_err_text(e)}",
                  file=sys.stderr)
            _MODEL_BYTES[key] = raw
    return _MODEL_BYTES[key]


def _new_session(model_path: Path, b: str, threads: int, static_feed=None, input_meta=None) -> ort.InferenceSession:
    """static_feed: tuple of input shapes; pins every symbolic input dim to it (add_free_dimension_override_by_name)
    so the provider plans one static graph (DirectML: ~2x faster than one dynamic session that switches shapes)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads and threads > 0:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    if b == "dml":
        # DirectML requires these settings.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if static_feed and input_meta:
        for shape, dims in zip(static_feed, input_meta):
            for d, n in zip(dims, shape):
                if isinstance(d, str) and d.startswith("zb_in"):
                    so.add_free_dimension_override_by_name(d, int(n))
    sess = ort.InferenceSession(_model_source(model_path, b), sess_options=so, providers=_ep_list(b))
    if b != "cpu" and sess.get_providers()[0] != _EP[b]:
        raise RuntimeError(f"{_EP[b]} did not load (session runs on {sess.get_providers()[0]})")
    return sess


SHAPE_SESSIONS = 8   # CUDA / DirectML: at most this many per-input-shape sessions per model (see SafeSession)
DML_PIN_AFTER = 3    # DirectML: a shape gets its own pinned session from its 3rd call on (one-off shapes stay dynamic)
CUDA_SHAPE_SESSIONS = SHAPE_SESSIONS


class SafeSession:
    """InferenceSession that never takes the run down because of a GPU provider.

    CUDA: ORT's CUDA Conv kernels keep the cuDNN state (descriptors + heuristic algo choice) of the LAST input
    shape only and rebuild it whenever the shape changes. A session that alternates shapes (SCRFD full frame /
    small-face ROIs / tiles, ArcFace batch 1-2-3) pays that on every call: det_10g ~44 ms per call on an NVIDIA test GPU
    instead of 4-11 ms. So on CUDA each distinct input shape gets its own session (up to CUDA_SHAPE_SESSIONS per
    model, the rest share the first one); the detector keeps its shapes to a small fixed set (detector.py).
    DirectML has the same switch cost (det_10g on an AMD test GPU, 8 alternating shapes: 22.6 ms/call in one
    session, 10.7 ms with one session per shape), so it gets per-shape sessions too, each pinned to its shape
    (free-dimension overrides -> one static DirectML graph); shapes past the cap run on the dynamic session.
    A pinned DirectML session costs ~0.25 s to create (a 960x960 face-split crop seen once cost 1.2 s), so a shape is
    pinned only from its DML_PIN_AFTER-th call on; until then, and if a pinned session fails, it runs dynamic.

    Creation tries the backend chain in order (cuda -> dml -> cpu, filtered by what is installed).
    At run time, per input shape:
      * a shape the GPU session rejects (DirectML on some SCRFD / ArcFace exports only takes the shape the
        graph was exported for, e.g. 640x640 and batch 1) is remembered and runs on a CPU twin session;
        a batch the GPU rejects is first retried item by item on the GPU;
      * a shape that worked on the GPU before and now fails means the device is gone: the whole session
        moves to the next backend.
    Only a failure on CPU propagates (that is a real model/input error)."""

    def __init__(self, model_path: Path, backend: str, threads: int):
        self.model_path, self.threads = Path(model_path), threads
        self.chain = backend_chain(backend)
        self.sess, self.backend = None, None
        self._cpu = None
        self._by_shape: dict = {}
        self._seen: dict = {}
        self._ok: set = set()
        self._to_cpu: set = set()
        self._split: set = set()
        self.runs = {"gpu": 0, "cpu": 0}
        self._open(0)

    def _open(self, start: int):
        last = None
        for i in range(start, len(self.chain)):
            b = self.chain[i]
            try:
                self.sess, self.backend, self._i = _new_session(self.model_path, b, self.threads), b, i
                self._ok, self._to_cpu, self._split = set(), set(), set()
                self._by_shape, self._seen = {}, {}
                USED[self.model_path.name] = b
                return
            except Exception as e:  # noqa: BLE001 - any provider error means: try the next one
                last = e
                if b == "cpu":
                    raise
                self._fallback(b, e)
        raise last or RuntimeError("no execution provider")

    def _fallback(self, b, e, what="session"):
        msg = _err_text(e)
        FALLBACKS.append({"model": self.model_path.name, "from": b, "what": what, "error": msg})
        print(f"[zoria-blur] warning: {b} failed for {self.model_path.name} ({what}: {msg}); falling back",
              file=sys.stderr)

    def _cpu_sess(self):
        if self._cpu is None:
            self._cpu = _new_session(self.model_path, "cpu", self.threads)
        return self._cpu

    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

    def get_providers(self):
        return self.sess.get_providers()

    def _shape_sess(self, key):
        """The session for this input shape (CUDA / DirectML; CPU shares one session)."""
        if self.backend not in ("cuda", "dml"):
            return self.sess
        s = self._by_shape.get(key)
        if s is not None:
            return s
        if self.backend == "dml":
            self._seen[key] = self._seen.get(key, 0) + 1
            if self._seen[key] < DML_PIN_AFTER:
                return self.sess
        if self.backend == "cuda" and not self._by_shape:
            s = self.sess                                   # the first shape uses the session we already have
        elif len({id(v) for v in self._by_shape.values()}) < SHAPE_SESSIONS:
            try:
                if self.backend == "dml":
                    meta = [i.shape for i in self.sess.get_inputs()]
                    s = _new_session(self.model_path, "dml", self.threads, static_feed=key, input_meta=meta)
                else:
                    s = _new_session(self.model_path, "cuda", self.threads)
            except Exception as e:  # noqa: BLE001 - e.g. out of GPU memory: share the first session
                self._fallback(self.backend, e, what=f"extra session for shape {list(key[0]) if key else []}")
                s = self.sess
        else:
            s = self.sess
        self._by_shape[key] = s
        return s

    def _run_cpu(self, output_names, feed):
        self.runs["cpu"] += 1
        return self._cpu_sess().run(output_names, feed)

    def run(self, output_names, feed):
        if self.backend == "cpu":
            self.runs["cpu"] += 1
            return self.sess.run(output_names, feed)
        key = tuple(tuple(getattr(v, "shape", ())) for v in feed.values())
        if key in self._to_cpu:
            return self._run_cpu(output_names, feed)
        if key in self._split:
            return self._run_split(output_names, feed)
        sess = None
        try:
            sess = self._shape_sess(key)
            out = sess.run(output_names, feed)
            self._ok.add(key)
            self.runs["gpu"] += 1
            return out
        except Exception as e:  # noqa: BLE001
            if (self.backend == "dml" and sess is not None and sess is not self.sess
                    and not _looks_like_device_loss(e)):
                # the pinned DirectML session failed where the dynamic one may not: drop the pin for this shape
                self._fallback("dml", e, what=f"pinned session for shape {list(key[0]) if key else []}")
                self._by_shape[key] = self.sess
                return self.run(output_names, feed)
            if key in self._ok or _looks_like_device_loss(e):
                self._fallback(self.backend, e)          # worked before: the device is gone
                self._open(self._i + 1)
                return self.run(output_names, feed)
            batch = min((v.shape[0] for v in feed.values() if getattr(v, "ndim", 0) >= 1), default=1)
            if batch > 1:
                try:
                    out = self._run_split(output_names, feed)
                    self._split.add(key)
                    return out
                except Exception:  # noqa: BLE001
                    pass
            if not self._to_cpu:
                self._fallback(self.backend, e, what=f"shape {list(key[0]) if key else []}")
            self._to_cpu.add(key)
            return self._run_cpu(output_names, feed)

    def _run_split(self, output_names, feed):
        import numpy as np
        n = min(v.shape[0] for v in feed.values())
        sess = self._shape_sess(tuple((1,) + tuple(v.shape[1:]) for v in feed.values()))
        parts = [sess.run(output_names, {k: v[i:i + 1] for k, v in feed.items()}) for i in range(n)]
        self.runs["gpu"] += n
        return [np.concatenate([p[j] for p in parts], axis=0) for j in range(len(parts[0]))]


def _err_text(e) -> str:
    # ORT on a non-English Windows embeds the localized system message (cp125x) in the error, which
    # pybind11 fails to decode: keep the readable ASCII part
    if isinstance(e, UnicodeDecodeError) and isinstance(e.object, (bytes, bytearray)):
        txt = bytes(e.object).decode("utf-8", "replace")
    else:
        txt = str(e)
    first = txt.splitlines()[0] if txt else ""
    return f"{type(e).__name__}: {first[:200]}"


def _looks_like_device_loss(e) -> bool:
    t = _err_text(e).lower()
    return any(k in t for k in ("887a0005", "887a0006", "device removed", "device_removed", "device hung"))


def make_session(model_path: Path | str, backend: str = "cpu", threads: int = 0) -> SafeSession:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model not found: {model_path}. Run: python scripts/download_models.py")
    s = SafeSession(model_path, backend, threads)
    SESSIONS.append(s)
    return s


SESSIONS: list = []


def used_summary() -> dict:
    """What actually ran: {'backend': 'dml'|'cpu'|'cuda'|'mixed', 'models': {...}, 'runs': {...}, 'fallbacks': [...]}"""
    vals = set(USED.values())
    dml = dml_adapter() if "dml" in vals else None
    return {"backend": vals.pop() if len(vals) == 1 else ("mixed" if vals else None),
            "dml_adapter": ({"index": dml["index"], "name": dml.get("name"), "how": dml.get("how")} if dml else None),
            "models": dict(USED), "runs": {s.model_path.name: dict(s.runs) for s in SESSIONS},
            "fallbacks": list(FALLBACKS)}


def gpu_names() -> list[str]:
    """Display adapters by name (Windows: WMI; elsewhere: nvidia-smi only). Never raises."""
    import subprocess
    nowin = {"creationflags": 0x08000000} if os.name == "nt" else {}
    names: list[str] = []
    try:
        if os.name == "nt":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_VideoController).Name"],
                               capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL, **nowin)
            names = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if not names:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL, **nowin)
            names = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        pass
    return names


def _is_software_adapter(name: str) -> bool:
    n = name.lower()
    return "basic render" in n or "basic display" in n or "remote" in n or "virtual" in n or "warp" in n


def _probe_session(model: Path, b: str, threads: int):
    import numpy as np
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads and threads > 0:
        so.intra_op_num_threads = threads
    if b == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(model), sess_options=so, providers=_ep_list(b)[:1])
    if sess.get_providers()[0] != _EP[b]:
        raise RuntimeError(f"session runs on {sess.get_providers()[0]}")
    return sess, sess.get_inputs()[0].name, np.zeros((1, 3, 640, 640), np.float32)


def _timed_runs(sess, name, x, runs: int) -> float:
    """Warm-up (graph compile on a GPU) + `runs` timed forward passes; median ms. Raises on invalid output."""
    import time
    import numpy as np
    sess.run(None, {name: x})
    ts = []
    for _ in range(max(1, runs)):
        t = time.perf_counter()
        outs = sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
        if not outs or any(not np.isfinite(np.asarray(o)).all() for o in outs):
            raise RuntimeError("test inference gave invalid output")
    return round(sorted(ts)[len(ts) // 2], 1)


def probe_devices(models_dir: Path | None = None, threads: int = 0, only: str | None = None, test: bool = True,
                  progress=None, gpu_test_only: bool = False) -> dict:
    """The user's rule for 'Otomatik': CUDA if a test inference works on it, else DirectML on a real
    (non-software) adapter if a test inference works, else CPU. Each probe is one forward pass of the
    smallest face detector, timed, with only that provider in the session.

    test=True adds the *test run*: the real default face detector (det_10g, 640x640, warm-up + 3 passes) on the
    chosen backend; if it fails there the next backend that passed is tried (CPU last), so the choice is one that
    really ran the app's model. The same test on the CPU gives the comparison the app shows (`bench`).
    gpu_test_only=True runs the test run only on a GPU backend (the CUDA build, asked first by the app: its CPU test
    would repeat the main build's). only='dml' etc. probes just that backend (the app's quiet start-up check of a saved choice).
    progress(ev) gets {"event": "probe", "item": cpu|memory|gpus|cuda|dml|cpu_backend|test, "status": ...} as each
    step starts and ends, so the app's system check can show it live. `reason` codes: not_installed, disabled,
    no_nvidia, software_only, no_model, not_requested, error, test_failed."""
    from .sysinfo import cpu_info, fingerprint, hardware_adapters, memory_gb

    def ev(item, status, **kw):
        if progress:
            try:
                progress({"event": "probe", "item": item, "status": status, **kw})
            except Exception:  # noqa: BLE001
                pass

    md = Path(models_dir) if models_dir else default_models_dir()
    model = next((md / m for m in ("det_500m.onnx", "det_10g.onnx") if (md / m).is_file()), None)
    bench_model = next((md / m for m in ("det_10g.onnx", "det_500m.onnx") if (md / m).is_file()), None)

    ev("cpu", "checking")
    cpu = cpu_info()
    ev("cpu", "found" if cpu.get("name") else "missing", **cpu)
    ev("memory", "checking")
    mem = memory_gb()
    ev("memory", "found" if mem else "missing", gb=mem)
    ev("gpus", "checking")
    gpus = [] if only else gpu_names()          # WMI takes 1-2 s; the quiet check uses DXGI only
    hw = hardware_adapters([g for g in gpus if not _is_software_adapter(g)])
    real_gpus = [g for g in gpus if not _is_software_adapter(g)] or list(hw)
    if not gpus:
        gpus = list(hw)
    nvidia = [g for g in real_gpus if "nvidia" in g.lower()]
    try:
        from .dxgi import adapters as _dx
        dxgi_all = _dx()          # [] off Windows / on error: then DirectML is simply tried
    except Exception:  # noqa: BLE001
        dxgi_all = []
    ev("gpus", "found" if real_gpus else "missing", names=real_gpus, all=gpus)

    def _name_for(b):
        if b == "cuda":
            return (nvidia or [None])[0]
        if b == "dml":
            try:
                return (dml_adapter() or {}).get("name") or (real_gpus[0] if real_gpus else None)
            except Exception:  # noqa: BLE001
                return None
        return cpu.get("name")

    probes = []
    for b in _ORDER:
        p = {"backend": b, "installed": _EP[b] in ort.get_available_providers(), "disabled": _disabled(b),
             "ok": False, "ms": None, "error": None, "reason": None}
        probes.append(p)
        item = "cpu_backend" if b == "cpu" else b
        if only and b != only:
            p["reason"] = "not_requested"
            continue
        ev(item, "checking")
        if not p["installed"] or p["disabled"]:
            p["reason"] = "not_installed" if not p["installed"] else "disabled"
            ev(item, "skipped", reason=p["reason"], nvidia_present=bool(nvidia))
            continue
        if b == "cuda" and real_gpus and not nvidia:
            p["error"], p["reason"] = "no NVIDIA adapter", "no_nvidia"   # skip loading ~2 GB of CUDA DLLs for nothing
            ev(item, "skipped", reason="no_nvidia")
            continue
        if b == "dml" and ((gpus and not real_gpus) or (dxgi_all and not hw)):
            p["error"], p["reason"] = "only a software display adapter", "software_only"
            ev(item, "skipped", reason="software_only")
            continue
        if model is None:
            p["error"], p["reason"] = f"no model in {md}", "no_model"
            ev(item, "failed", reason="no_model")
            continue
        try:
            sess, name, x = _probe_session(model, b, threads)
            p["ms"] = _timed_runs(sess, name, x, 1)
            p["ok"] = True
            ev(item, "ok", ms=p["ms"], device=_name_for(b))
        except Exception as e:  # noqa: BLE001
            p["error"], p["reason"] = _err_text(e), "error"
            ev(item, "failed", reason="error", error=p["error"])

    bench = None
    if test and bench_model is not None:
        cand = [p for p in probes if p["ok"] and not (gpu_test_only and p["backend"] == "cpu")]
        if not any(p["backend"] == "cpu" for p in cand) and not only and not gpu_test_only:
            cand.append(next(p for p in probes if p["backend"] == "cpu"))
        cpu_ms = None
        for p in cand:
            b = p["backend"]
            ev("test", "checking", backend=b)
            try:
                sess, name, x = _probe_session(bench_model, b, threads)
                ms = _timed_runs(sess, name, x, 3)
                del sess
                p["ok"] = True
                bench = {"backend": b, "model": bench_model.stem, "ms": ms, "cpu_ms": ms if b == "cpu" else None}
                if b != "cpu" and not only:
                    try:                          # the same test on the CPU: how much the GPU helps here
                        sess, name, x = _probe_session(bench_model, "cpu", threads)
                        cpu_ms = _timed_runs(sess, name, x, 2)
                        bench["cpu_ms"] = cpu_ms
                    except Exception:  # noqa: BLE001
                        pass
                ev("test", "ok", **bench)
                break
            except Exception as e:  # noqa: BLE001
                p["ok"], p["error"], p["reason"] = False, f"test run failed: {_err_text(e)}", "test_failed"
                ev("test", "failed", backend=b, error=p["error"])
    chosen = next((p["backend"] for p in probes if p["ok"]), "cpu")
    from .dxgi import adapters
    dml = dml_adapter() if _EP["dml"] in ort.get_available_providers() else None
    device = None
    if chosen == "cuda":
        device = (nvidia or real_gpus or [None])[0]
    elif chosen == "dml":
        device = (dml or {}).get("name") or (real_gpus[0] if real_gpus else None)
    return {"chosen": chosen, "device_name": device, "gpus": gpus, "nvidia_present": bool(nvidia),
            "dml_adapter": ({"index": dml["index"], "name": dml.get("name"), "how": dml.get("how")} if dml else None),
            "dml_adapters": [{k: a[k] for k in ("index", "name", "dedicated_mb", "software")} for a in adapters()],
            "onnxruntime": ort.__version__, "probes": probes, "bench": bench, "quick": bool(only),
            "system": {"cpu": cpu, "memory_gb": mem, "adapters": hw, "fingerprint": fingerprint(cpu, hw)}}


def default_models_dir() -> Path:
    env = os.environ.get("ZORIA_MODELS")
    if env:
        return Path(env)
    # source tree / PyInstaller _internal; the CUDA engine borrows the main engine's models (paths.py)
    from .paths import find_resource, resource_dirs
    return find_resource("models") or resource_dirs()[0] / "models"
