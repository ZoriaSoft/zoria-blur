"""Backend fallback rules (no GPU, no models needed): python tests/test_backend.py"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zoria_blur import backend as B  # noqa: E402


class FakeSess:
    """fail_run: True = every run fails; a set of shapes = only those fail; 'after1' = fails after one run."""
    def __init__(self, b, fail_run=False):
        self.b, self.fail_run, self.n = b, fail_run, 0

    def run(self, names, feed):
        self.n += 1
        shape = next(iter(feed.values())).shape if feed else ()
        bad = (self.fail_run is True or (isinstance(self.fail_run, set) and shape in self.fail_run)
               or (self.fail_run == "after1" and self.n > 1))
        if bad:
            raise RuntimeError(f"{self.b} rejects {shape}")
        n = shape[0] if shape else 1
        return [np.full((n, 1), {"cuda": 3, "dml": 2, "cpu": 1}[self.b])]

    def get_inputs(self):
        return []

    def get_outputs(self):
        return []


def with_fakes(avail, create_fail=(), run_fail=None):
    run_fail = run_fail or {}
    B.available_backends = lambda: [b for b in B._ORDER if b in avail and not B._disabled(b)]

    def new(model_path, b, threads, **kw):
        if b in create_fail:
            raise RuntimeError(f"{b} cannot create")
        return FakeSess(b, run_fail.get(b, False))
    B._new_session = new
    B.USED.clear(); B.FALLBACKS.clear()


def test_chains():
    with_fakes({"cuda", "dml", "cpu"})
    assert B.backend_chain("auto") == ["cuda", "dml", "cpu"]
    assert B.backend_chain("dml") == ["dml", "cpu"]
    assert B.backend_chain("cpu") == ["cpu"]
    with_fakes({"dml", "cpu"})
    assert B.backend_chain("auto") == ["dml", "cpu"]
    assert B.backend_chain("cuda") == ["dml", "cpu"]        # asked for CUDA without it: next in the rule
    os.environ["ZORIA_DISABLE_DML"] = "1"
    try:
        assert B.backend_chain("auto") == ["cpu"]
        assert B.backend_chain("dml") == ["cpu"]
    finally:
        del os.environ["ZORIA_DISABLE_DML"]


def test_create_and_run_fallback(model):
    x1 = {"x": np.zeros((1, 3, 640, 640), np.float32)}
    x2 = {"x": np.zeros((1, 3, 352, 640), np.float32)}
    xb = {"x": np.zeros((4, 3, 112, 112), np.float32)}
    with_fakes({"cuda", "dml", "cpu"}, create_fail={"cuda"})
    s = B.make_session(model, "auto")
    assert s.backend == "dml" and s.run(None, x1)[0][0, 0] == 2
    assert B.FALLBACKS[0]["from"] == "cuda"
    # a shape DirectML rejects runs on the CPU twin; the session stays on DirectML for the others
    with_fakes({"dml", "cpu"}, run_fail={"dml": {(1, 3, 352, 640), (4, 3, 112, 112)}})
    s = B.make_session(model, "auto")
    assert s.run(None, x1)[0][0, 0] == 2
    assert s.run(None, x2)[0][0, 0] == 1 and s.backend == "dml"
    assert s.run(None, x1)[0][0, 0] == 2, "a rejected shape must not move the whole session"
    out = s.run(None, xb)
    assert out[0].shape == (4, 1) and out[0][0, 0] == 2, "a rejected batch is retried item by item on the GPU"
    assert s.runs["cpu"] == 1
    # a shape that worked and now fails = device lost: the whole session moves on
    with_fakes({"dml", "cpu"}, run_fail={"dml": "after1"})
    s = B.make_session(model, "auto")
    assert s.run(None, x1)[0][0, 0] == 2
    assert s.run(None, x1)[0][0, 0] == 1 and s.backend == "cpu"
    assert B.used_summary()["backend"] == "cpu"
    with_fakes({"cpu"}, run_fail={"cpu": True})
    s = B.make_session(model, "auto")
    try:
        s.run(None, x1)
        raise AssertionError("a CPU failure must propagate")
    except RuntimeError:
        pass


def test_cuda_session_per_shape(model):
    """CUDA: every input shape runs on its own session (ORT CUDA rebuilds cuDNN state on each shape change),
    capped per model; DirectML too, each pinned to its shape; CPU keeps one session."""
    made = []
    with_fakes({"cuda", "cpu"})
    new = B._new_session
    B._new_session = lambda m, b, t, **kw: made.append(b) or new(m, b, t)
    s = B.make_session(model, "cuda")
    shapes = [(1, 3, 64 * i, 64) for i in range(1, B.CUDA_SHAPE_SESSIONS + 3)]
    for sh in shapes + shapes:
        assert s.run(None, {"x": np.zeros(sh, np.float32)})[0][0, 0] == 3
    assert made.count("cuda") == B.CUDA_SHAPE_SESSIONS, made          # first + extra, then capped
    first = s._by_shape[(shapes[0],)]
    assert first is s.sess and s._by_shape[(shapes[1],)] is not first and s._by_shape[(shapes[-1],)] is first
    made.clear()
    pinned = []
    with_fakes({"dml", "cpu"})
    B._new_session = lambda m, b, t, **kw: made.append(b) or pinned.append(kw.get("static_feed")) or new(m, b, t)
    s = B.make_session(model, "dml")
    for sh in shapes + shapes + shapes:
        assert s.run(None, {"x": np.zeros(sh, np.float32)})[0][0, 0] == 2
    # the dynamic session + one pinned session per shape (from its 3rd call) up to the cap; later shapes stay dynamic
    assert made.count("dml") == 1 + B.SHAPE_SESSIONS, made
    assert pinned[1:] == [(sh,) for sh in shapes[:B.SHAPE_SESSIONS]], pinned
    assert s._by_shape[(shapes[-1],)] is s.sess and s._by_shape[(shapes[0],)] is not s.sess
    # a pinned session that fails falls back to the dynamic session for that shape, not to the CPU
    made.clear()
    with_fakes({"dml", "cpu"})
    def new2(m, b, t, **kw):
        made.append(b)
        return FakeSess(b, True if kw.get("static_feed") else False)
    B._new_session = new2
    s = B.make_session(model, "dml")
    for _ in range(5):
        assert s.run(None, {"x": np.zeros(shapes[0], np.float32)})[0][0, 0] == 2
    assert s.backend == "dml" and s.runs["cpu"] == 0 and s._by_shape[(shapes[0],)] is s.sess
    assert made == ["dml", "dml"] and B.FALLBACKS[-1]["what"].startswith("pinned"), (made, B.FALLBACKS)
    made.clear()
    with_fakes({"cpu"})
    B._new_session = lambda m, b, t, **kw: made.append(b) or new(m, b, t)
    s = B.make_session(model, "cpu")
    for sh in shapes:
        s.run(None, {"x": np.zeros(sh, np.float32)})
    assert made == ["cpu"]


def test_det_buckets():
    from zoria_blur.detector import det_bucket, _fixed_shapes
    assert [det_bucket(n) for n in (32, 160, 192, 320, 352, 640, 672, 960, 1000)] == \
        [160, 160, 320, 320, 480, 640, 960, 960, 1280]
    assert _fixed_shapes("cuda") and _fixed_shapes("dml") and not _fixed_shapes("cpu")


def test_free_output_shapes():
    """The InsightFace exports declare 640x640 / batch-1 output shapes (DirectML took them literally); freeing them
    changes no number and gives the symbolic input dims unique names for per-shape overrides."""
    import onnxruntime as ort
    from zoria_blur.onnx_shapes import free_output_shapes, input_dim_name
    md = B.default_models_dir()
    det, emb = md / "det_10g.onnx", md / "w600k_mbf.onnx"
    if not det.is_file() or not emb.is_file():
        print("  (models missing, skipped)")
        return
    ort.set_default_logger_severity(3)
    for m, x in ((det, np.random.rand(1, 3, 128, 256).astype(np.float32)),
                 (emb, np.random.rand(3, 3, 112, 112).astype(np.float32))):
        a = ort.InferenceSession(str(m), providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(free_output_shapes(m.read_bytes()), providers=["CPUExecutionProvider"])
        assert all(isinstance(o.shape[0], str) for o in b.get_outputs()), b.get_outputs()[0].shape   # no 12800 / 1
        n = a.get_inputs()[0].name
        for p, q in zip(a.run(None, {n: x}), b.run(None, {n: x})):
            assert p.shape == q.shape and np.array_equal(p, q)
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name(input_dim_name(0, 2), 128)
    so.add_free_dimension_override_by_name(input_dim_name(0, 3), 256)
    s = ort.InferenceSession(free_output_shapes(det.read_bytes()), so, providers=["CPUExecutionProvider"])
    assert s.get_inputs()[0].shape == [1, 3, 128, 256], s.get_inputs()[0].shape


def test_embedder_gpu_batches():
    """GPU backends: ArcFace batches are cut to <= 8 and padded to 1/2/4/8 (few per-shape sessions), same features."""
    from zoria_blur.embedder import ArcFace
    seen = []

    class S:
        backend = "dml"

        def run(self, names, feed):
            x = next(iter(feed.values()))
            seen.append(x.shape[0])
            return [x.reshape(len(x), -1)[:, :4] + 1.0]
    e = ArcFace.__new__(ArcFace)
    e.session, e.output_name, e.input_name = S(), "o", "i"
    blob = np.random.rand(11, 3, 2, 2).astype(np.float32)
    out = e._run(blob)
    assert seen == [8, 4] and out.shape == (11, 4) and np.allclose(out, blob.reshape(11, -1)[:, :4] + 1.0)
    seen.clear()
    assert e._run(blob[:1]).shape == (1, 4) and seen == [2]          # DirectML: no batch-1 session
    seen.clear(); S.backend = "cuda"
    e._run(blob[:1]); assert seen == [1]
    seen.clear(); S.backend = "cpu"
    assert e._run(blob).shape == (11, 4) and seen == [11]


def test_dml_adapter_pick():
    """Dual-GPU: DirectML goes to the high-performance (else most-VRAM) hardware adapter; ZORIA_DML_DEVICE wins."""
    from zoria_blur.dxgi import pick
    lab = [{"index": 0, "name": "Intel(R) Graphics", "dedicated_mb": 128, "software": False, "perf_rank": 1},
           {"index": 1, "name": "NVIDIA GeForce Test GPU", "dedicated_mb": 3935, "software": False, "perf_rank": 0},
           {"index": 2, "name": "Microsoft Basic Render Driver", "dedicated_mb": 0, "software": True, "perf_rank": 2}]
    assert pick(lab)["index"] == 1 and pick(lab)["how"] == "high_performance"
    old_win = [dict(a, perf_rank=None) for a in lab]
    assert pick(old_win)["index"] == 1 and pick(old_win)["how"] == "most_vram"
    assert pick(lab[2:]) is None, "a software adapter is never picked"
    assert pick([]) is None
    os.environ["ZORIA_DML_DEVICE"] = "0"
    try:
        assert pick(lab)["index"] == 0 and pick(lab)["how"] == "manual"
        assert pick([])["index"] == 0
    finally:
        del os.environ["ZORIA_DML_DEVICE"]
    B.dml_adapter = lambda: {"index": 1, "name": "NVIDIA"}
    assert B._ep_list("dml")[0] == ("DmlExecutionProvider", {"device_id": "1"})
    B.dml_adapter = lambda: None
    assert B._ep_list("dml")[0] == "DmlExecutionProvider"


def test_err_text_non_utf8():
    try:
        b"Parametre hatal\xfd.".decode("utf-8")
    except UnicodeDecodeError as e:
        assert "Parametre" in B._err_text(e)


if __name__ == "__main__":
    orig = (B.available_backends, B._new_session, B.dml_adapter)
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "m.onnx"
        m.write_bytes(b"x")
        test_chains(); print("ok chains")
        test_create_and_run_fallback(m); print("ok fallback")
        test_cuda_session_per_shape(m); print("ok cuda session per shape")
        test_det_buckets(); print("ok detector shape buckets")
        test_free_output_shapes(); print("ok free output shapes")
        test_embedder_gpu_batches(); print("ok embedder gpu batches")
        test_dml_adapter_pick(); print("ok dml adapter pick")
        test_err_text_non_utf8(); print("ok error text")
    B.available_backends, B._new_session, B.dml_adapter = orig
    print("all backend tests passed")
