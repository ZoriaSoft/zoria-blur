# NVIDIA (Turing) check (dev .venv-cuda, repo root): ORT CUDA rebuilds cuDNN conv state on every input-shape change.
# One session alternating two shapes ~44 ms/call; one session per shape 11 / 4 ms. python scripts/cuda_shape_switch_bench.py
import time
import onnxruntime as ort
ort.preload_dlls()
import numpy as np
def mk(opts=None):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession("models/det_10g.onnx", so, providers=[("CUDAExecutionProvider", dict({"cudnn_conv_algo_search": "HEURISTIC"}, **(opts or {}))), "CPUExecutionProvider"])
a = np.random.rand(1, 3, 640, 384).astype(np.float32); b = np.random.rand(1, 3, 192, 320).astype(np.float32)
def bench(sa, sb, n=20):
    na, nb = sa.get_inputs()[0].name, sb.get_inputs()[0].name
    for _ in range(2): sa.run(None, {na: a}); sb.run(None, {nb: b})
    ta = tb = 0
    for _ in range(n):
        t = time.perf_counter(); sa.run(None, {na: a}); ta += time.perf_counter() - t
        t = time.perf_counter(); sb.run(None, {nb: b}); tb += time.perf_counter() - t
    return f"A {ta/n*1000:.1f} ms  B {tb/n*1000:.1f} ms"
s = mk(); print("one session, alternating shapes  ", bench(s, s))
s1, s2 = mk(), mk(); print("two sessions, one shape each     ", bench(s1, s2))
s = mk({"cudnn_conv_use_max_workspace": "0"}); print("one session alt, max_workspace=0 ", bench(s, s))
