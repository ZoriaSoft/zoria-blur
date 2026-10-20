"""Free an ONNX model's declared output shapes, in memory, without the onnx package.

Why: the InsightFace exports declare their *outputs* with the shape of the export input, SCRFD
(det_10g / det_500m) as [12800, 1] ... [800, 10] (= 640x640) and ArcFace w600k_mbf as [1, 512] (batch 1),
and carry ~100-280 `value_info` entries with the same baked-in sizes. The input is dynamic ([1, 3, ?, ?] /
[None, 3, 112, 112]), so CPU and CUDA ignore the stale sizes, but DirectML takes them as the real output
shape: the last Reshape of every other input size fails with E_INVALIDARG (80070057, "Reshape_223"), so
DirectML ran SCRFD only at 640x640 and ArcFace only at batch 1 (measured on an AMD test GPU, ORT 1.24.4
DirectML, every graph optimization level).

`free_output_shapes(bytes)` rewrites only the top of the GraphProto: every fixed dim of a graph output
becomes a named symbolic dim, the `value_info` list is dropped (ORT re-infers it) and the symbolic input dims get
unique names (`input_dim_name`, so a session can pin them with add_free_dimension_override_by_name). Nodes and
initializers are copied byte for byte, so the math is unchanged.
"""
from __future__ import annotations


def _varint(buf, i):
    v = s = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << s
        if b < 0x80:
            return v, i
        s += 7


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _fields(buf):
    """Yield (field_no, wire_type, raw_bytes_of_whole_field, payload) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        start = i
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            _, i = _varint(buf, i)
            payload = None
        elif wt == 1:
            i += 8
            payload = None
        elif wt == 2:
            ln, i = _varint(buf, i)
            payload = buf[i:i + ln]
            i += ln
        elif wt == 5:
            i += 4
            payload = None
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield f, wt, buf[start:i], payload


def _ld(field: int, payload: bytes) -> bytes:
    return _enc_varint((field << 3) | 2) + _enc_varint(len(payload)) + payload


def _rewrite(buf, path, fn):
    """Rebuild a message, replacing the length-delimited field path[0] by its rewritten payload."""
    out = bytearray()
    for f, wt, raw, payload in _fields(buf):
        if wt == 2 and f == path[0]:
            new = fn(payload) if len(path) == 1 else _rewrite(payload, path[1:], fn)
            out += _ld(f, new)
        else:
            out += raw
    return bytes(out)


def _free_dims(shape: bytes, prefix: str, values: bool = True) -> bytes:
    """values=True: every fixed dim becomes a named symbolic dim (outputs). values=False: fixed dims stay,
    symbolic dims get a unique name (inputs: SCRFD names both H and W '?', so a per-dimension override
    could not tell them apart)."""
    out = bytearray()
    k = 0
    for f, wt, raw, payload in _fields(shape):
        if f == 1 and wt == 2:                       # Dimension
            dim = bytearray()
            for g, gwt, graw, _ in _fields(payload):
                if g == 1 and gwt == 0 and values:   # dim_value -> dim_param
                    dim += _ld(2, f"{prefix}_{k}".encode())
                elif g == 2 and gwt == 2:            # dim_param -> unique name
                    dim += _ld(2, f"{prefix}_{k}".encode())
                else:
                    dim += graw
            out += _ld(1, bytes(dim))
            k += 1
        else:
            out += raw
    return bytes(out)


def _free_output(vi: bytes, idx: int) -> bytes:
    prefix = f"zb_out{idx}"
    # ValueInfoProto.type(2) -> TypeProto.tensor_type(1) -> Tensor.shape(2)
    return _rewrite(vi, (2, 1, 2), lambda s: _free_dims(s, prefix))


def _name(msg, field: int) -> bytes:
    return next((bytes(q) for g, gwt, _, q in _fields(msg) if g == field and gwt == 2), b"")


def input_dim_name(i: int, k: int) -> str:
    """Name free_output_shapes gives to symbolic dim k of graph input i (for add_free_dimension_override_by_name)."""
    return f"zb_in{i}_{k}"


def _free_graph(graph: bytes) -> bytes:
    out = bytearray()
    idx = n_in = 0
    init_names = {_name(p, 8) for f, wt, _, p in _fields(graph) if f == 5 and wt == 2}   # TensorProto.name = 8
    for f, wt, raw, payload in _fields(graph):
        if f == 11 and wt == 2:                      # graph input (opset-11 exports may list initializers too)
            vi = bytes(payload)
            if _name(vi, 1) not in init_names:
                pre = f"zb_in{n_in}"
                vi = _rewrite(vi, (2, 1, 2), lambda s, pre=pre: _free_dims(s, pre, values=False))
                n_in += 1
            out += _ld(11, vi)
            continue
        if f == 13 and wt == 2:                      # value_info: drop, ORT infers it again
            continue
        if f == 12 and wt == 2:                      # graph output
            out += _ld(12, _free_output(bytes(payload), idx))
            idx += 1
        else:
            out += raw
    return bytes(out)


def free_output_shapes(model: bytes) -> bytes:
    """ModelProto bytes -> the same model with symbolic output dims and no value_info."""
    mv = memoryview(model)
    out = bytearray()
    for f, wt, raw, payload in _fields(mv):
        if f == 7 and wt == 2:                       # ModelProto.graph
            out += _ld(7, _free_graph(payload))
        else:
            out += raw
    return bytes(out)
