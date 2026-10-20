"""Quick offline checks (no video, no models): python tests/test_engine.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zoria_blur.clustering import cluster_tracks  # noqa: E402
from zoria_blur.renderer import apply_regions, dense_sequence  # noqa: E402
from zoria_blur.tracker import FaceTracker  # noqa: E402


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_dense_sequence():
    frames = [0, 2, 4, 20, 22]           # gap 4->20 (16 frames) should be interpolated at 30 fps
    boxes = [[100 + f, 100, 150 + f, 160] for f in frames]
    d = dense_sequence(frames, boxes, 30.0, 100, extend_s=0.1)
    assert all(f in d for f in range(0, 23)), "gap not filled"
    assert 23 in d and 25 in d, "end not extrapolated"
    w = [d[f][2] for f in range(0, 23)]
    assert min(w) >= 50 * 0.97 - 1e-6, "smoothing shrank box"


def test_scene_cut_not_bridged():
    frames = [0, 2, 4, 6, 8, 10]
    boxes = [[100, 100, 150, 160]] * 3 + [[900, 500, 950, 560]] * 3   # jumps across the screen
    d = dense_sequence(frames, boxes, 30.0, 100, extend_s=0.0)
    assert 5 not in d, "scene cut was interpolated across the frame"


def test_clustering_cannot_link():
    a, b = unit(np.random.RandomState(0).randn(512)), unit(np.random.RandomState(1).randn(512))
    descs = [a, unit(a + 0.1 * b), b, a]
    frames = [set(range(0, 10)), set(range(20, 30)), set(range(0, 30)), set(range(5, 8))]
    lab = cluster_tracks(descs, frames, 0.4)
    assert lab[2] not in (lab[0], lab[1], lab[3]), lab
    assert lab[3] != lab[0], "overlapping tracks were merged"


def test_tracker_follows():
    tr = FaceTracker()
    e = unit(np.ones(512))
    for f in range(0, 20, 2):
        tr.step(f, np.array([[100 + 3 * f, 100, 160 + 3 * f, 170, 0.9]], np.float32), [e])
    ts = tr.finalize()
    assert len(ts) == 1 and len(ts[0].obs) == 10


def test_blur_changes_only_face():
    img = np.random.RandomState(0).randint(0, 255, (360, 640, 3), np.uint8)
    out = apply_regions(img.copy(), [(320, 180, 60, 80)])
    assert np.abs(out[180, 320].astype(int) - img[180, 320]).sum() > 0
    assert (out[10, 10] == img[10, 10]).all()


def _hist(c):
    h = np.zeros(256, np.float32); h[c] = 1.0
    return h


def test_body_tracker_crossing():
    from zoria_blur.bodies import BodyTracker
    bt = BodyTracker()
    for f in range(0, 40, 2):
        a = [100 + 6 * f, 100, 160 + 6 * f, 300, 0.9]      # moves right
        b = [340 - 6 * f, 100, 400 - 6 * f, 300, 0.9]      # moves left, crosses a
        bt.step(f, np.array([a, b], np.float32), [_hist(3), _hist(200)])
    ts = bt.finalize()
    assert len(ts) == 2, len(ts)
    for t in ts:
        xs = [o[1][0] for o in t.obs]
        assert all(np.diff(xs) > 0) or all(np.diff(xs) < 0), "identity switched at the crossing"


def test_face_body_association_and_head():
    from zoria_blur.bodies import HeadModel, associate_faces, body_box_at
    bodies = [{"id": 1, "frames": [0, 10, 20], "boxes": [[100, 100, 200, 400]] * 3},
              {"id": 2, "frames": [0, 10, 20], "boxes": [[300, 100, 400, 400]] * 3}]
    faces = [{"id": 7, "frames": [0, 10], "boxes": [[135, 110, 165, 150]] * 2},
             {"id": 8, "frames": [0, 10, 20], "boxes": [[335, 110, 365, 150]] * 3}]
    associate_faces(faces, bodies)
    assert set(bodies[0]["face_tracks"]) == {7} and set(bodies[1]["face_tracks"]) == {8}
    fb = {(t["id"], f): b for t in faces for f, b in zip(t["frames"], t["boxes"])}
    hm = HeadModel(bodies[0], lambda tid, f: fb.get((tid, f)))
    h = hm.head(body_box_at(bodies[0], 20), 20, 30.0)       # face missing at 20: head from body
    assert abs(h[0] - 150) < 5 and 110 < h[1] < 150 and h[2] >= 30, h


def test_link_bodies_cannot_link():
    from zoria_blur.bodies import link_bodies
    b1 = {"id": 1, "frames": [0, 10], "boxes": [[100, 100, 200, 400]] * 2}
    b2 = {"id": 2, "frames": [20, 30], "boxes": [[110, 100, 210, 400]] * 2}   # after b1: link
    b3 = {"id": 3, "frames": [5, 30], "boxes": [[110, 100, 210, 400]] * 2}    # overlaps b1: never
    link_bodies([b1, b2, b3], {1: _hist(5), 2: _hist(5), 3: _hist(5)}, 30.0)
    assert b2["reid_prev"] == 1 and b3["reid_prev"] is None


def test_head_covers_span_when_face_long_gone():
    """Face seen off-centre (arm out), then she turns away for 2 s: the head must cover the generic
    top-centre spot too, not only the old offset (real clip b_turning_head)."""
    from zoria_blur.bodies import HeadModel, body_box_at, head_prior
    body = {"id": 1, "frames": [0, 10, 70], "boxes": [[100, 100, 500, 700]] * 3,
            "face_frames": [0, 10], "face_tracks": [7, 7]}
    fb = {(7, 0): [140, 120, 220, 220], (7, 10): [140, 120, 220, 220]}   # face far left of centre
    hm = HeadModel(body, lambda tid, f: fb.get((tid, f)))
    near = hm.head(body_box_at(body, 10), 10, 30.0)
    assert abs(near[0] - 180) < 10, near                     # face just seen: learnt spot
    far = hm.head(body_box_at(body, 70), 70, 30.0)           # 2 s later
    pr = head_prior(body_box_at(body, 70))
    assert far[0] - far[2] / 2 <= 180 - 40 and far[0] + far[2] / 2 >= pr[0] + pr[2] / 2 - 1, (far, pr)


def test_drop_nested_face_part():
    from zoria_blur.analyze import _drop_nested
    d = np.array([[530, 289, 748, 606, 0.85], [528, 299, 753, 421, 0.71],     # face + its eye band
                  [800, 300, 900, 420, 0.80]], np.float32)                    # another face
    k = np.zeros((3, 5, 2), np.float32)
    d2, k2 = _drop_nested(d, k)
    assert len(d2) == 2 and 0.71 not in [round(float(x), 2) for x in d2[:, 4]], d2
    # two overlapping real faces (one partly behind the other) are both kept
    d = np.array([[100, 100, 200, 220, 0.9], [160, 90, 250, 200, 0.7]], np.float32)
    assert len(_drop_nested(d, np.zeros((2, 5, 2), np.float32))[0]) == 2
    # a weak big false positive does not swallow a confident small face
    d = np.array([[0, 0, 400, 400, 0.3], [100, 100, 150, 160, 0.9]], np.float32)
    assert len(_drop_nested(d, np.zeros((2, 5, 2), np.float32))[0]) == 2


def test_blur_except_carves_kept_face():
    """blur-except: a neighbour's padded blur must not cover the confidently kept face next to it."""
    img = np.random.RandomState(1).randint(0, 255, (300, 400, 3)).astype(np.uint8)
    out = apply_regions(img.copy(), [(150, 150, 60, 80)], padding=0.6, keep=[[190, 110, 250, 190]])
    assert np.array_equal(out[130:170, 215:245], img[130:170, 215:245]), "kept face blurred"
    assert not np.array_equal(out[140:160, 140:160], img[140:160, 140:160]), "blurred face not blurred"


def test_kept_body_test():
    """blur-except: a weak / low-similarity face on a body that only ever carried the kept person is
    the kept person (turned away); on a body that also carried someone else it is not."""
    from types import SimpleNamespace
    from zoria_blur.renderer import _kept_body_test
    bodies = [{"id": 1, "frames": [0, 30, 60], "boxes": [[100, 100, 300, 700]] * 3,
               "face_frames": [0, 10, 20], "face_tracks": [1, 1, 1]},
              {"id": 2, "frames": [0, 30, 60], "boxes": [[600, 100, 800, 700]] * 3,
               "face_frames": [0, 10, 20], "face_tracks": [1, 2, 1]}]
    test = _kept_body_test(SimpleNamespace(data={"bodies": bodies}), {1}, 30.0)
    assert test(30, [170, 110, 230, 180])            # head zone of the kept-only body, 10 frames later
    assert not test(30, [670, 110, 730, 180])        # body 2 also carried track 2
    assert not test(30, [170, 500, 230, 570])        # not in the head zone
    assert not test(60 + 40, [170, 110, 230, 180])   # body gone


def test_tile_grid_and_merge():
    """Tiles cover the frame with overlap; a face seen by two passes is merged into one box."""
    from zoria_blur.detector import tile_grid, merge_dets
    from zoria_blur.smallfaces import face_tile_grid
    t = tile_grid(1280, 720, 3, 2, 0.2)
    assert len(t) == 6 and t[0][:2] == (0, 0) and t[-1][2:] == (1280, 720)
    assert all(b[0] < a[2] for a, b in zip(t[:2], t[1:3])), "tiles do not overlap"
    assert face_tile_grid(1280, 720) == (3, 2) and np.prod(face_tile_grid(3840, 2160)) <= 12
    d1 = (np.array([[100, 100, 112, 114, 0.6]], np.float32), np.zeros((1, 5, 2), np.float32))
    d2 = (np.array([[101, 100, 113, 115, 0.8], [400, 300, 410, 312, 0.5]], np.float32), np.zeros((2, 5, 2), np.float32))
    d, _ = merge_dets([d1, d2])
    assert len(d) == 2 and abs(d[0, 4] - 0.8) < 1e-6


def test_body_parts_and_head_size():
    """A tile's head-and-torso box of a person the full pass sees whole is dropped (it became a
    faceless 'unknown' body whose head blur covered the kept face); a child in front of an adult stays.
    Head size: 12 % of a standing body's height, 40 % of a partial body's width."""
    from zoria_blur.smallfaces import drop_body_parts, head_size, merge_rois
    d = np.array([[674, 174, 837, 714, .9], [669, 168, 830, 395, .8], [700, 400, 760, 600, .8]], np.float32)
    out = drop_body_parts(d)
    assert len(out) == 2 and not any((out[:, 3] == 395)), out
    hs = head_size([[1173, 350, 1229, 501], [685, 157, 841, 392]])
    assert abs(hs[0] - 0.12 * 151) < 1e-6 and abs(hs[1] - 0.4 * 156) < 1e-6
    r = merge_rois([[0, 0, 10, 10, 5], [20, 0, 30, 10, 4], [500, 500, 520, 520, 6]])
    assert len(r) == 2 and any(x[:4] == [0, 0, 30, 10] and x[4] == 4 for x in r)


def test_guessed_head_carves_only_confirmed_kept_face():
    """blur-except: an unseen (guessed) head of someone else ("k") is carved out over a kept face whose
    identity is confirmed (strong keep box), even inside its own box; over an unconfirmed kept face it
    blurs (fail-safe); a never-carve region (False) blurs regardless."""
    img = np.random.RandomState(2).randint(0, 255, (300, 400, 3)).astype(np.uint8)
    kept = [190, 110, 250, 190]
    out = apply_regions(img.copy(), [(215, 160, 90, 110, "k")], keep=[kept + [True]])
    diff = np.abs(out.astype(int) - img.astype(int))
    assert diff[135:165, 212:228].max() <= 3, "confirmed kept face blurred"
    assert not np.array_equal(out[200:215, 200:230], img[200:215, 200:230]), "guessed head not blurred"
    for flag, strong, msg in (("k", False, "unconfirmed face carved"), (False, True, "fail-safe path carved")):
        out = apply_regions(img.copy(), [(215, 160, 90, 110, flag)], keep=[kept + [strong]])
        assert np.abs(out.astype(int) - img.astype(int))[135:165, 212:228].mean() > 20, msg


def test_group_body_heads():
    """blur-except: a body box holding several people (faces of >= 3 persons) does not guess heads where
    an individual body inside it has its head - that body predicts the head itself; a group guess no
    individual body explains stays (fail-safe)."""
    from types import SimpleNamespace
    from zoria_blur.renderer import _plan_body_heads
    ff = list(range(0, 12, 2))
    tracks, face_frames, face_tracks = [], [], []
    for i, cx in enumerate((160, 300, 440)):
        tracks.append({"id": i + 1, "person_id": i + 1, "frames": ff, "boxes": [[cx - 20, 110, cx + 20, 160]] * len(ff)})
    for f in ff:
        for i in range(3):
            face_frames.append(f); face_tracks.append(i + 1)
    group = {"id": 1, "frames": [0, 60], "boxes": [[100, 100, 500, 700]] * 2,
             "face_frames": face_frames, "face_tracks": face_tracks}
    members = [{"id": 2 + i, "frames": [0, 60], "boxes": [[cx - 40, 102, cx + 40, 700]] * 2,
                "face_frames": [], "face_tracks": []} for i, cx in enumerate((160, 300))]
    proj = SimpleNamespace(tracks=tracks, data={"video": {"fps_float": 30.0}, "analysis": {"params": {"every": 2}},
                                                "bodies": [group] + members})
    rep, pf = {}, {}
    _plan_body_heads(proj, "blur-except", {99}, rep, pf, set())
    assert rep["group_heads_dropped"] > 0
    g14 = [r for r in pf.get(14, []) if r[1] < 200]
    # persons 1 and 2 stand in individual bodies: their group guesses are dropped, the members' own
    # (unknown -> blurred) heads remain; person 3 has no individual body: its group guess stays
    assert any(abs(r[0] - 440) < 30 for r in g14), g14
    assert sum(abs(r[0] - 160) < 30 for r in g14) == 1 and sum(abs(r[0] - 300) < 30 for r in g14) == 1, g14


def test_strong_owner_only_near_its_span():
    """blur-selected: a long body track that switched owner (kept man first, a selected person's face 4 s
    later) does not blur the kept man's head from frame 0 because the selected person is "strong" on it;
    near the selected person's own span the fail-safe still blurs."""
    from types import SimpleNamespace
    from zoria_blur.renderer import _plan_body_heads
    fa, fb = list(range(0, 60, 2)), list(range(240, 300, 2))
    tracks = [{"id": 1, "person_id": 1, "frames": fa, "boxes": [[280, 110, 320, 160]] * len(fa)},
              {"id": 2, "person_id": 2, "frames": fb, "boxes": [[280, 110, 320, 160]] * len(fb)}]
    body = {"id": 1, "frames": [0, 300], "boxes": [[250, 100, 350, 500]] * 2,
            "face_frames": fa + fb, "face_tracks": [1] * len(fa) + [2] * len(fb)}
    proj = SimpleNamespace(tracks=tracks, data={"video": {"fps_float": 30.0}, "analysis": {"params": {"every": 2}},
                                                "bodies": [body]})
    rep, pf = {}, {}
    _plan_body_heads(proj, "blur-selected", {2}, rep, pf, set())
    assert not pf.get(20), pf.get(20)          # kept person 1 seen here, selected person 2 is 7 s away
    assert pf.get(200) and pf.get(250)         # within 2 s of / inside person 2's span on this body


def test_qa_source_face_score():
    """QA re-detects a candidate in the source frame: only a detection at the same place counts."""
    from zoria_blur.qa import source_face_score

    class Det:
        def __init__(self, dets):
            self.dets = np.array(dets, np.float32).reshape(-1, 5)

        def detect_region(self, img, box, **kw):
            return self.dets, None
    img = np.zeros((100, 100, 3), np.uint8)
    box = [40, 40, 60, 64]
    assert source_face_score(Det([]), img, box) == 0.0
    assert abs(source_face_score(Det([[41, 41, 61, 63, 0.8], [0, 0, 10, 10, 0.95]]), img, box) - 0.8) < 1e-6
    assert source_face_score(Det([[70, 70, 90, 90, 0.9]]), img, box) == 0.0      # somewhere else
    assert abs(source_face_score(Det([[30, 30, 80, 90, 0.6]]), img, box) - 0.6) < 1e-6   # centre inside


def test_risks_precise():
    """Occlusion only where a lost body is behind another body and its person is not seen elsewhere; no 1 s coast
    range from one-off boxes; no face gap while a body carrying the person is detected."""
    from types import SimpleNamespace
    from zoria_blur.risks import compute_risks
    fr = list(range(0, 60, 2))
    walker = {"id": 1, "frames": fr, "boxes": [[100, 100, 200, 400]] * len(fr), "face_tracks": [1],
              "coast_frames": list(range(60, 90)), "coast_boxes": [[100, 100, 200, 400]] * 30}
    front = {"id": 2, "frames": list(range(60, 90, 2)), "boxes": [[90, 90, 210, 420]] * 15, "face_tracks": [],
             "coast_frames": [], "coast_boxes": []}
    flicker = {"id": 3, "frames": [10], "boxes": [[500, 100, 600, 400]], "face_tracks": [],
               "coast_frames": list(range(11, 41)), "coast_boxes": [[500, 100, 600, 400]] * 30}
    track = {"id": 1, "person_id": 1, "frames": [0, 2, 4, 50, 52], "boxes": [[130, 100, 170, 150]] * 5,
             "confidence": 1.0}

    def proj(bodies):
        return SimpleNamespace(data={"video": {"fps_float": 30.0, "frame_count": 120},
                                     "analysis": {"params": {"every": 2}}, "bodies": bodies},
                               persons=[{"id": 1}], tracks=[track])
    rs = compute_risks(proj([walker, front, flicker]))
    occ = [r for r in rs if r["type"] == "occlusion"]
    assert len(occ) == 1 and occ[0]["body"] == 1 and occ[0]["start_frame"] == 60, occ
    assert not [r for r in rs if r["type"] == "track_break"], rs         # body seen through the face gap
    rs = compute_risks(proj([walker, flicker]))                           # nobody in front: walked away
    assert not [r for r in rs if r["type"] == "occlusion"], rs
    rs = compute_risks(proj([dict(walker, frames=[0, 2, 4], boxes=[[100, 100, 200, 400]] * 3, coast_frames=[],
                                  coast_boxes=[])]))
    assert [r for r in rs if r["type"] == "track_break"], rs             # face gap, no body: still a risk


def _syn_face(face, dark=False, seed=1):
    """A 200x200 frame: 1/f background, an elliptic face with skin texture, eyes, brows, nose and mouth."""
    import cv2
    rs = np.random.RandomState(seed)
    H = W = 200
    f = np.fft.fftfreq(W)[None, :] ** 2 + np.fft.fftfreq(H)[:, None] ** 2
    f[0, 0] = 1
    bg = np.real(np.fft.ifft2(np.fft.fft2(rs.randn(H, W)) / np.sqrt(f)))
    img = np.dstack([(bg - bg.mean()) / bg.std() * 25 + 110] * 3).astype(np.float32)
    cx = cy = 100
    w, h = face, int(face * 1.25)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)
    tex = np.real(np.fft.ifft2(np.fft.fft2(rs.randn(H, W)) / np.sqrt(f) ** 0.5))
    img[m > 0] = np.array([150, 170, 200], np.float32) + ((tex - tex.mean()) / tex.std() * 6)[m > 0, None]
    e = max(1, int(face * 0.08))
    for dx in (-1, 1):
        x, y = cx + dx * int(face * 0.2), cy - int(face * 0.12)
        cv2.ellipse(img, (x, y), (2 * e, e), 0, 0, 360, (40, 40, 50), -1)
        cv2.circle(img, (x, y), max(1, e // 2), (10, 10, 10), -1)
        cv2.line(img, (x - 2 * e, y - 2 * e), (x + 2 * e, y - 2 * e - dx), (60, 50, 50), max(1, e // 2))
    cv2.line(img, (cx, cy - int(face * .05)), (cx - e, cy + int(face * .15)), (110, 120, 150), max(1, e // 2))
    cv2.ellipse(img, (cx, cy + int(face * .32)), (int(face * .15), e), 0, 0, 180, (70, 60, 120), max(1, e // 2))
    if dark:
        img = img * 0.25 + 5
    return np.clip(img, 0, 255).astype(np.uint8), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_qa_sharpness():
    """QA sharpness (mid-band DoG projection): a re-encoded / slightly moving face that was NOT blurred - small, dark
    or both - must read sharp (the Laplacian projection of 0.1.3 read 0.16-0.32 here and counted it as blurred, so
    a kept person looked blurred and, worse, a leak looked blurred); mild blurs that leave a face recognisable
    (3x3, sigma 1.5) must read sharp too (fail-safe: a leak); the renderer's blur, a strong blur and a heavy blur
    with sensor noise must read blurred."""
    import cv2
    from zoria_blur.qa import MIN_FACE_PX, SHARP_THRESH, sharp_ratio

    def jpeg(x, q):
        return cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    rs = np.random.RandomState(2)
    for face in (14, 24, 48, 90):
        for dark in (False, True):
            img, b = _syn_face(face, dark)
            moved = cv2.filter2D(img, -1, np.ones((1, 3), np.float32) / 3)
            reg = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])]
            sharp = {"untouched": img, "jpeg40": jpeg(img, 40), "motion+jpeg70": jpeg(moved, 70),
                     "gauss3x3": cv2.GaussianBlur(img, (3, 3), 0), "sigma1.5": cv2.GaussianBlur(img, (0, 0), 1.5)}
            blurred = {"render+jpeg80": jpeg(apply_regions(img.copy(), reg, "gaussian", 1.0, 0.3), 80),
                       "sigma3": cv2.GaussianBlur(img, (0, 0), 3.0),
                       "sigma8+noise": np.clip(cv2.GaussianBlur(img, (0, 0), 8) + rs.normal(0, 4, img.shape),
                                               0, 255).astype(np.uint8)}
            for k, out in sharp.items():
                r, _ = sharp_ratio(img, out, b)
                assert r is not None and r >= SHARP_THRESH, (face, dark, k, r)
            for k, out in blurred.items():
                r, _ = sharp_ratio(img, out, b)
                assert r is not None and r < SHARP_THRESH, (face, dark, k, r)
            if face >= 24:
                r, _ = sharp_ratio(img, jpeg(img, 60), b)
                assert r >= 0.85, (face, dark, r)                # untouched re-encode stays near 1
                r, _ = sharp_ratio(img, sharp["gauss3x3"], b)
                assert r >= 0.6, (face, dark, r)
                r, _ = sharp_ratio(img, blurred["render+jpeg80"], b)
                assert r < 0.1, (face, dark, r)
    img, b = _syn_face(10)
    assert b[2] - b[0] < MIN_FACE_PX and sharp_ratio(img, img, b)[0] is None    # too small to judge
    flat = np.full((200, 200, 3), 90, np.uint8)
    assert sharp_ratio(flat, flat, [80, 80, 120, 130])[0] is None                # no detail to lose


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
