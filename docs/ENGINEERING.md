# Zoria Blur: engineering notes

These are the working notes kept while building the engine and the Windows app: every design
decision, measurement and fix, in the order they were made. For a short introduction see the
[README](../README.md).

## Install (Windows or Linux, Python 3.12)

```bat
python -m venv .venv
.venv\Scripts\activate            (Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts\download_models.py
```

ffmpeg >= 5.1 is required (`-fps_mode`). Lookup order: `--ffmpeg PATH` → `ZORIA_FFMPEG` env →
`bin\ffmpeg.exe` next to the app → `imageio-ffmpeg` package → PATH. ffprobe is used if it sits
next to ffmpeg or on PATH; otherwise OpenCV probes the file.

## Use

```bat
:: 1) analyze: detect + track + embed + cluster  ->  myvideo_zoria\project.json + thumbs\person_XXX\*.jpg
python -m zoria_blur analyze myvideo.mp4

:: look at thumbs\person_001 ... and pick ids
python -m zoria_blur persons myvideo_zoria

:: 2) render
python -m zoria_blur render myvideo.mp4 --mode blur-except   --persons 1      -o out.mp4
python -m zoria_blur render myvideo.mp4 --mode blur-selected --persons 2,3 --style pixelate

:: fix clustering mistakes (UI will call these)
python -m zoria_blur merge     myvideo_zoria 5 7          (person 7 becomes part of person 5)
python -m zoria_blur split     myvideo_zoria --tracks 12  (track 12 -> new person)
python -m zoria_blur assign    myvideo_zoria --tracks 12 --person 3
python -m zoria_blur rename    myvideo_zoria 3 "Ali"
python -m zoria_blur recluster myvideo_zoria --threshold 0.45

:: QA a rendered file (also runs automatically at the end of render; --no-qa to skip)
python -m zoria_blur qa out.mp4 -p myvideo_zoria

:: manual fix: draw one box where a face was missed; optical flow carries it to the next detection
python -m zoria_blur addbox myvideo.mp4 -p myvideo_zoria --person 2 --frame 340 --box 210,220,250,280

:: one-frame preview for the UI
python -m zoria_blur preview myvideo.mp4 --frame 300 --mode blur-except --persons 1 -o prev.png --side-by-side
```

Every command prints one JSON object on stdout (`{"ok": true, ...}` or `{"ok": false, "error": ...}`).
`--json-progress` adds JSON-lines progress events (`{"event":"progress","phase":"analyze","frame":..,"total":..}`)
before the result — this is the UI contract. Timeline markers for the UI:
`{"event":"risk", "type":..., "start_s":..., "end_s":..., "persons":[...]}` (after analyze) and
`{"event":"qa_issue", "kind":"leak"|"risk", "start_s":..., "end_s":..., ...}` (after render / qa).

Global flags: `--backend cpu|dml|cuda|auto`, `--threads N`, `--models DIR`, `--ffmpeg PATH`.

`devices [--only cuda|dml|cpu] [--no-test]` is the system check: processor (name, physical cores, threads), memory, display
adapters, then CUDA → DirectML → CPU each with a test inference, then a *test run* of the real face detector (det_10g,
640x640, warm-up + 3 passes) on the chosen backend (falls to the next one that passed if it fails there; the same test
on the CPU gives `bench.cpu_ms`). With `--json-progress` every step is a `{"event":"probe","item":cpu|memory|gpus|cuda|dml|cpu_backend|test,
"status":checking|found|missing|ok|skipped|failed, "reason":...}` line. The result carries `system.fingerprint`
(processor name + hardware DXGI adapter names). `--only B --no-test` is the app's quiet start-up check (no WMI, ~1.7 s).

### Main options

| analyze | default | meaning |
|---|---|---|
| `--det-model` | det_10g | `det_10g` (accurate), `det_500m` (~3.5x faster, slightly weaker on small/side faces), `det_2.5g` (`download_models.py --with-2.5g`) |
| `--every` | 2 | run the detector every N frames; boxes in between are interpolated at render |
| `--det-mode` | balanced | small faces: `fast` (one 640 pass, v0.2), `balanced` (switches small-face mode on by itself when small people are seen), `thorough` (tiles on every detection frame). See *Small faces* below |
| `--det-size` | 640 | full-frame detector input (frame is downscaled to fit; non-square input, no padding). Small faces are the job of `--det-mode`, not of a bigger det-size (it loses large faces) |
| `--cluster-thresh` | 0.40 | cosine similarity to merge tracks into one person (higher = more persons) |
| `--person-model` | yolox_tiny | body detector: `yolox_tiny`, `yolox_nano` (~3x faster, a bit weaker), `yolox_s`, `none` (face-only, v0.1 behaviour) |
| `--person-every` | 1 | run the body detector on every Nth face-detection frame (Kalman predicts in between) |
| `--no-split-faces` | off | disable merged-face re-detection |
| `--weak-thresh` | 0.3 | weak faces (this score up to `--det-thresh`) that start no track are stored; blur-except blurs them |

| render | default | meaning |
|---|---|---|
| `--style` | gaussian | or `pixelate` |
| `--strength` | 1.0 | >1 stronger |
| `--padding` | 0.3 | ellipse is 30 % larger than the face box (+ taller, shifted up for forehead/hair) |
| `--extend` | 0.3 s | blur starts earlier / ends later than the track (fail-safe) |
| `--max-gap` | 1.0 s | detection gaps up to this are interpolated |
| `--smooth` | 0.12 s | zero-phase temporal smoothing of boxes (no jitter, no lag) |
| `--keep-conf` | 0.30 | blur-except: a kept person's track must match the person (leave-one-out) at least this much, else it is blurred |
| `--keep-det-sim` | 0.18 | blur-except: a single detection that does not look like the kept person is blurred |
| `--crf` / `--preset` | 18 / veryfast | libx264 quality / speed |
| `--no-qa` / `--qa-every` | off / 3 | skip the QA pass / check every Nth frame |

## How it works

```
zoria_blur/
  backend.py     ONNX Runtime session factory (cpu / DirectML / CUDA by one flag, falls back to CPU loudly)
  detector.py    SCRFD (InsightFace det_*) with 5 landmarks; downscale to det_size, padded only to the
                 stride (non-square input); overlapping tiles + NMS merge
  embedder.py    ArcFace w600k_mbf (MobileFaceNet, 512-d), 5-point similarity alignment
  tracker.py     ByteTrack/BoT-SORT style: Kalman (xyah) + IoU, high/low score 2-stage, appearance
                 fusion, lost-track re-ID by embedding (2 s buffer)
  clustering.py  average-linkage agglomerative over track descriptors + cannot-link
                 (tracks visible at the same time are never one person)
  analyze.py     phase 1 pipeline; also splits tracks at abrupt appearance changes (ID switches)
  project.py     project.json / embeddings.npz, merge / split / assign / rename / recluster, thumbnails
  renderer.py    dense per-frame boxes (interpolate, smooth, extrapolate ends), feathered ellipse blur
  video_io.py    ffmpeg pipes; CFR decode via fps filter so frame N is identical in both phases
                 (VFR phone videos, rotation handled); encode libx264 + original audio stream-copied
  person_detector.py  YOLOX (Apache-2.0) person boxes, COCO class 0
  smallfaces.py  --det-mode: small-mode trigger, tiled bodies, head-ROI pass, tile refresh (v0.3)
  bodies.py      body tracker (Kalman + IoU + torso HSV histogram), face<->body association,
                 body re-ID after occlusion, per-person head model, orphan face-track adoption
  risks.py       risky moments as time ranges (track breaks, occlusions, merges, ambiguity, low conf)
  qa.py          post-render leak check (detector pass + mask pass), kept-face collateral stats
  manual.py      addbox: user box + optical-flow continuation, stored as a manual track
  cli.py         argparse CLI, JSON on stdout
```

### Person-level tracking (v0.2)

```
frame ──► SCRFD faces ──► merged-face check ──► ArcFace ──► face tracker ──► face tracks ─┐
   └────► YOLOX persons ─► body tracker (Kalman+IoU+torso colour) ─► body tracks ───────┤
                                                                                         ▼
          clustering (faces -> persons) ◄── face<->body association (Hungarian, head zone)
                       │                         │
                       ▼                         ▼
              persons (ids for the UI)   bodies know which persons they carried, and where the
                                         head sits relative to the body box
```

* **Head from body.** For a person who must be blurred, the head is blurred on every frame their
  body is visible: from the face box when it is detected, otherwise from the body box using that
  person's own face-vs-body offset learnt on this body (generic prior when never seen). The head
  region is at least the generic head size, so hair and the back of the head are covered too.
* **Occlusion.** A body that is lost (behind someone) keeps a 1 s Kalman prediction ("coast"); a
  body track that ends and a new one that starts ≤ 2 s later are bridged when they carry the same
  person (face embeddings) or match by torso colour (re-ID, never across tracks that coexist).
* **Merged faces.** A face box that is abnormally wide, jumps in size vs. its track, or sits where
  two people's heads overlap is re-detected in an upscaled crop with a looser NMS (IoU 0.6 instead
  of 0.4, i.e. *less* suppression, so two overlapping faces survive); a box that splits into two is
  replaced by both.
* **Orphan face tracks** (too small / cut by the frame edge to embed) take the person of the body
  that carries them when that body is otherwise clearly one person.
* **Fail-safe — uncertain ⇒ blur.** Candidates for a body's head at frame *f* are the persons whose
  faces it carried nearest before *and* after *f*, plus every person it carried a lot (≥ 20 %) — the latter only within 2 s of the span where that
  person's face rode this body (a long body track can switch owner: in one real test clip the kept man's body
  track carried a selected man's face 4 s later and blurred the kept man from frame 0).
  *blur-selected*: blurred if any candidate is selected; a selected head is drawn **uncarved** when
  its face is not detected, so it may briefly blur a kept face it overlaps (accepted trade-off);
  when the selected face is detected, kept faces next to it are carved out. *blur-except*: blurred
  unless every candidate is a confidently kept track; bodies with no known face are blurred.
* **Head away from the face.** The learnt face-vs-body offset is pose dependent (arm out vs. turned
  away), so once the last face sample is > 0.15 s old the head box spans from the learnt spot to the
  generic top-centre of the body (full span after 0.75 s).
* **Weak faces** (score 0.3–0.55, mostly tiny / far / dark faces) never start a track; analyze keeps
  the ones no track took (`weak_faces[]` in project.json) and *blur-except* blurs them, chained over
  time, except where they sit on a kept face.
* **Kept faces are carved in both modes.** blur-except carves confidently kept faces out of a
  neighbour's padded blur (never out of the neighbour's own face box; uncertain regions stay uncarved).
  A person whose face is detected on another body at that frame gets no head predicted from this
  body (two people in one body box).
* **Nested face parts** (a box ≥ 70 % inside a larger, about-as-confident face box, e.g. the eye band
  of a close-up face) are dropped — two real faces never nest.
* Face-level gap fill (`dense_sequence`) and optical-flow bridging (`bridge.py`) are still used as
  a second layer.

### Small faces (`--det-mode`, v0.3)

SCRFD runs at 640 px, so in a 720p frame every face is shrunk 2x; faces under ~20 px (street crowds,
people far away) mostly fall below what the model can see. Measured, not guessed:

**WIDER FACE validation** (`tests/eval_wider.py`, 352 images = every 10th per event, 6 417 valid faces;
recall at IoU 0.5 by face size = max(w, h) in original px; easy/medium/hard = the official face lists;
FP/img = detections on no annotated face, an upper bound because WIDER leaves many crowd faces
unannotated; s/img on 2 CPU cores, idle). Dataset used for internal evaluation only, never shipped.

Score >= 0.5 (starts a track):

| setting | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | all | easy | medium | hard | FP/img | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 (fast, v0.2) | 0.3 | 14.9 | 52.5 | 67.6 | 83.5 | 93.5 | 42.3 | 92.2 | 86.9 | 59.6 | 0.24 | 0.17 |
| 960 | 5.8 | 33.3 | 65.3 | 74.7 | 85.5 | 92.6 | 50.5 | 91.2 | 87.6 | 68.7 | 0.33 | 0.38 |
| 1280 | 14.2 | 46.0 | 68.8 | 75.2 | 84.6 | 89.6 | 55.3 | 89.0 | 85.9 | 71.8 | 0.41 | 0.66 |
| 640 + 2x2 tiles | 10.0 | 41.9 | 69.9 | 76.6 | 86.4 | 95.4 | 54.7 | 93.9 | 89.7 | 72.7 | 0.49 | 0.86 |
| **640 + 3x2 tiles (thorough)** | 17.6 | 47.3 | 69.1 | 76.0 | 86.6 | 95.8 | 57.5 | 94.1 | 89.8 | 73.6 | 0.56 | 1.43 |
| 1280 + 3x2 tiles | 20.7 | 51.2 | 71.4 | 77.1 | 86.0 | 90.7 | 59.0 | 90.2 | 87.1 | 74.3 | 0.60 | 1.96 |
| **balanced small mode** (640 + head ROIs from bodies) | 10.5 | 48.3 | 69.7 | 77.0 | 86.8 | 94.0 | 55.8 | 92.9 | 89.1 | 74.0 | 0.45 | 0.45 |
| det_500m 640 | 0.1 | 5.2 | 32.3 | 50.3 | 68.3 | 87.4 | 31.9 | 83.2 | 75.3 | 45.0 | 0.18 | 0.02 |

Score >= 0.3 (what blur-except blurs as weak faces):

| setting | <10 | 10-16 | 16-24 | 24-32 | 32-64 | >64 | all | easy | medium | hard | FP/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 (fast) | 3.5 | 37.4 | 72.2 | 80.0 | 89.8 | 96.3 | 53.5 | 95.0 | 91.9 | 73.9 | 1.63 |
| 960 | 20.8 | 60.4 | 79.9 | 86.1 | 90.9 | 95.4 | 64.4 | 94.6 | 92.2 | 81.7 | 2.21 |
| 1280 | 35.1 | 68.4 | 81.8 | 85.0 | 91.1 | 91.9 | 69.6 | 91.9 | 91.0 | 83.2 | 2.51 |
| 640 + 2x2 tiles | 30.0 | 68.2 | 82.0 | 87.6 | 91.9 | 97.0 | 69.1 | 95.9 | 93.6 | 84.5 | 3.18 |
| **640 + 3x2 tiles (thorough)** | 38.2 | 70.6 | 82.7 | 88.2 | 91.5 | 97.2 | 71.9 | 95.9 | 93.5 | 85.2 | 3.49 |
| 1280 + 3x2 tiles | 43.1 | 72.6 | 83.9 | 87.2 | 91.5 | 92.5 | 73.2 | 92.4 | 91.6 | 85.1 | 3.53 |
| **balanced small mode** | 23.3 | 64.9 | 81.7 | 87.2 | 91.3 | 96.5 | 66.5 | 95.4 | 93.0 | 83.5 | 2.64 |
| det_500m 640 | 0.2 | 19.6 | 56.0 | 69.5 | 79.7 | 93.7 | 43.5 | 90.9 | 84.5 | 61.3 | 1.92 |

(GT faces per bucket: <10: 1475, 10-16: 922, 16-24: 1102, 24-32: 525, 32-64: 823, >64: 570.)
Take-aways: a bigger full-frame det-size helps small faces but **loses large ones** (>64 px: 93.5 → 89.6 %
at 1280) because the model was trained on 640 crops; tiles keep the 640 full pass for large faces and
add the small ones. The balanced ROI pass gets ~the 10-16 px recall of 3x2 tiles at 1/3 of the cost;
its weak spot is < 10 px faces with no detected body.

**Real clips, semi-ground truth** (`tests/semigt.py`, 3 frames each of c_crowd_selfie, f_small_far,
g_night_street; the strongest setting's boxes verified by eye on zoomed crops; boxes too blurred to
call face / not-face are excluded; `work/testset/semigt/gt.json`):

| setting | c_crowd_selfie | f_small_far | g_night_street | total @0.5 | total @0.3 | FP @0.3 (9 frames) | s/frame (720p) |
|---|---|---|---|---|---|---|---|
| 640 (fast) | 15/15 | 6/15 → 12/15 | 5/30 → 11/30 | 26/60 (43 %) | 38/60 (63 %) | 6 | 0.14 |
| 960 | 15/15 | 6 → 12/15 | 12 → 16/30 | 33/60 (55 %) | 43/60 (72 %) | 10 | 0.31 |
| 1280 | 15/15 | 8 → 12/15 | 16 → 24/30 | 39/60 (65 %) | 51/60 (85 %) | 11 | 0.55 |
| 640 + 2x2 tiles | 15/15 | 9 → 12/15 | 11 → 19/30 | 35/60 (58 %) | 46/60 (77 %) | 14 | 0.75 |
| 640 + 3x2 tiles (thorough) | 15/15 | 9 → 13/15 | 18 → 27/30 | 42/60 (70 %) | 55/60 (92 %) | 14 | 1.36 |
| 1280 + 3x2 tiles | 15/15 | 9 → 13/15 | 20 → 27/30 | 44/60 (73 %) | 55/60 (92 %) | 14 | 1.82 |
| balanced small mode (one frame) | 15/15 | 8 → 12/15 | 14 → 20/30 | 37/60 (62 %) | 47/60 (78 %) | 12 | 0.47 |

(a → b = found at score ≥ 0.5 → at ≥ 0.3.) The "balanced" row is a single frame; in a video balanced also
re-runs the full tile pass every 5th detection frame and re-detects around last frame's small faces.

**Modes**

* `fast` — one 640 pass (v0.2 behaviour, minus the square padding).
* `balanced` (default) — starts as fast. A *scout* every 10 detection frames runs the body detector on 2x2
  tiles; when it sees ≥ 2 people whose head would be < 20 px at the 640 scale (or the full pass finds such a
  face) the clip switches to **small mode** for good: bodies on 2x2 tiles every 2nd detection frame (YOLOX at
  416 px misses people under ~100 px tall in 720p: 15 → 48 bodies on a street frame), a **head-ROI pass**
  (head zones of small bodies + around the previous frame's small faces, merged into a few rectangles and
  re-detected upscaled so the smallest expected head is ~24 px at the detector input, ≤ 0.8 MP per frame), and
  a full 3x2 tile pass every 5th detection frame for faces without a visible body. Clips with only big faces
  (selfies, close-ups) never leave fast mode and cost the same as before.
* `thorough` — 640 full pass + 3x2 tiles (grid chosen so the frame is seen at ≥ 1.3x native, ≤ 12 tiles) and
  tiled bodies on every detection frame.

**Body fail-safe for faces no detector finds.** In blur-except every body track that carries no face is an
"unknown" person and its head (generic head box from the body box) is blurred; with tiled bodies this now
covers far people whose 5-10 px face no face detector sees. QA of a small-mode project runs its detector at 960.

**Guessed heads never hide the kept person (blur-except, v0.3.1).** Tiled bodies also bring *group boxes*
(one body box over 3-4 people walking together) and many short faceless bodies. A head predicted from a
body while that person's face is not seen is a guess; in balanced mode on the street clip such guesses of
the people behind the kept passer-by blurred him in 42 % of his frames (v0.2 engine, fast detection: 18 %). Two rules, both
in `renderer._plan_body_heads`:
* a guessed head (unseen body head or body-gap bridge) is carved out over a kept face whose identity is
  **confirmed** (ArcFace similarity ≥ 0.30 to the kept person, or an unembeddable detection within 2
  detection frames of a confirmed one on the same track). Over an unconfirmed kept face it still blurs both
  (fail-safe). Face-track gaps of a blurred person (a face going behind someone) stay uncarved, as before;
* a **group box** (carried faces of ≥ 3 persons, or holds ≥ 2 other body boxes at that frame) does not guess
  a head where an individual body inside it has its own head: that body predicts the head itself
  (faceless → unknown → blurred; the kept person's own body → visible). Guesses no individual body explains
  stay (fail-safe for people seen only inside the group).
Measured: kept-person collateral on the street clip 42 % → 0 %, faceless-body head coverage unchanged
(96.8 → 96.6 % of body-frames), no new real leaks (see `work/testset/RESULTS.md`).

### QA (`qa`, automatic after `render`)

1. **Detector pass** (every `--qa-every` frames): SCRFD at threshold 0.3 on the rendered frame; each
   face is compared with the original pixels (Laplacian-variance ratio, `--sharp` 0.45). Sharp faces on
   a kept person's box are excluded unless the embedding clearly says it is a blurred person.
   Leak kinds: `in_mask`, `identity`, `not_kept` (blur-except). Sharp faces matching nobody are
   reported as `unknown_face` risks (blur-selected), not leaks.
2. **Mask pass**: every real detection of a person who must be blurred is checked directly
   (`mask_miss` if it kept its detail).

3. **Source check (0.1.3)**: a detector-pass candidate must be a face of the *source* video. It is re-detected in the
   original frame in an upscaled crop (face ~80 px at the detector input): ≥ 0.5 → leak as above; 0.2-0.5 →
   `unconfirmed_face` risk (timeline, not a leak); < 0.2 → `render_artifacts` (no face there in the source). The clip4
   "leak" at 0.30-0.42 s on every backend was a face-like emblem on a glass door and clasped hands next to a blurred
   head; an unblurred clip4 still gives 600 leaks.
4. **Sharpness (0.1.4)**: kept detail = regression coefficient of the rendered image on the original in a mid
   band, `<Do,Dr>/<Do,Do>` with D = difference of Gaussians (sigma 0.8-2.0), in the core 70 % of the face box;
   sharp = ≥ 0.35 (`qa.SHARP_THRESH`). Measured (`tests/eval_sharpness.py`, 7 test clips, faces ≥ 12 px):
   untouched crf-18 re-encode 0.77-1.0, kept faces in real renders 0.54-1.0, real blurred faces 0.00-0.26
   (749 faces), 3x3 blur 0.62-1.1 (100 % flagged), sigma-1.5 blur 98 % flagged, sigma-3 9.5 %, sigma-8 + noise ~0.
   Mild blurs that leave a face recognisable therefore count as leaks (fail-safe).
   Why not the earlier ones: a Laplacian *variance ratio* counts codec noise and blur edges as detail (a fully
   blurred dark head read 0.47-0.57, 8x pixelation 1.9-2.3); the *Laplacian projection* of 0.1.3 lost real detail to
   the re-encode (untouched crf-18 small / dark / moving faces 0.23-0.52, the kept passer-by in f_small_far read
   blurred in 25 of 26 frames) - it would also have called a real leak "blurred". 0.1.3 was not shipped.
   Faces < 12 px wide are not judged (`faces_too_small` in the result): their core is < 9 px, where the band is the
   whole face (a renderer-blurred 6 px face reads up to 0.81, a 3x3 blur down to 0). The CLI's `--style pixelate`
   keeps ≥ 3 blocks across a face and often reads as sharp on small faces (flagged) - QA is calibrated for the
   default Gaussian blur.

Result: `leak_count`, `leak_ranges` (time ranges), `kept_face_blurred` / `kept_blurred_ranges`
(collateral of the fail-safe), `risky` (with `relevant` = involves a blurred person), `render_artifacts`,
`unconfirmed_faces`, `passed`.

**Risky moments are precise (0.1.3).** `occlusion` only where a lost body's predicted box is ≥ 30 % behind another
detected body and the people it carried are not seen elsewhere (their face, or another body carrying them = an id
change); bodies with < 3 detections add no 1 s coast range; `track_break` only where no body carrying the person is
detected (with a body the head is blurred from it, not guessed). On a 16 s real test clip: one 3.78-15.79 s band (12.3 of 16.1 s) →
5 moments, 6.2 s.
QA cannot see a back-of-head that was not blurred (no face to detect) — check contact sheets for that.

### Real-video test set (regression suite)

`PYTHONPATH=. python tests/run_testset.py [--reuse] [--sheets] [clip ...]` runs analyze → render
(blur-selected and blur-except on the most prominent person) → QA on 8 Pexels stock clips in
`work/testset/` (occlusion, head turning away, 6-face crowd, shaky handheld, leaving the frame, small
far faces, night street, faces close together) and prints one table; sources, licenses and cuts are in
`work/testset/SOURCES.md` (clips are not committed). `--sheets` writes contact sheets for the visual
check QA cannot do (back of head). ~20 min for the whole set on 2 cores.

### Scenario tests

`PYTHONPATH=. python tests/scenarios.py [behind leave_return fast small_far merge]` builds synthetic clips
from real people cut out of a local test clip (`--src`, not included), runs analyze → render → QA and also checks
the ground truth (A's visible face never keeps its detail). ~20–35 s per scenario on 2 cores.

**Fail-safe (blur-except):** only tracks assigned to a kept person *and* matching that person's other
tracks (leave-one-out similarity ≥ `--keep-conf`, ≥ 2 embeddings) stay visible. Every other track —
other persons, low-confidence persons, unembeddable tiny faces — is blurred, and individual frames of a
kept track whose embedding does not match are blurred too. Blur is extended 0.3 s past each track end.

**Project JSON (schema_version 1)** — `video` (size, fps, frame_count, audio codec), `analysis` (params,
timing), `persons[]` (`id, name, track_ids, num_detections, confidence, low_confidence, thumbnails[]`),
`tracks[]` (`id, person_id, frames[], boxes[[x1,y1,x2,y2]], scores[], det_sims[], confidence, crops[]`),
`edits[]` (log of merge/split/...). Boxes are in original-resolution pixels at detection frames.
v0.2 adds: `bodies[]` (`id, frames[], boxes[], scores[], coast_frames[], coast_boxes[], face_frames[],
face_tracks[], reid_prev, reid_sim`; torso histograms in embeddings.npz as `b{id}_h`),
`analysis.risks[]`, `analysis.merge_events[]`, `analysis.ambiguous_assoc_frames[]`,
`analysis.adopted_tracks[]`, `manual[]` + tracks with `"manual": true` (addbox), and `last_render`
(mode/persons/options the `qa` command reuses).

## Performance (2 CPU cores, 464x832 60 fps, 965 frames / 16 s, `--every 2`)

| step | seconds |
|---|---|
| face detector (det_10g, 483 frames) | 120 |
| person detector (yolox_tiny, 483 frames, ~66 ms each) | 38 |
| merged-face re-detection | 18 |
| embeddings | 11 |
| **analyze total** | **189** (5.1 video fps) |
| render (blur + encode) | 8 |
| QA (every 3rd frame) | 87 |

**Per detection mode** (v0.3.1, 2 CPU cores idle, 1280x720 25/30 fps, `--every 2`, det_10g + YOLOX-tiny;
`work/modes/timings.md`). Clips with only big faces never leave fast mode in balanced, so they cost the same
as fast (c/d/e of the test set: 33-75 s).

| clip | mode | analyze s | face det s | body det s | video fps | persons / bodies |
|---|---|---|---|---|---|---|
| f_small_far (8 s, 200 frames) | fast | 31.7 | 15.1 | 8.3 | 6.3 | 14 / 30 |
| | **balanced** (default) | 135.7 | 79.8 | 21.8 | 1.47 | 22 / 62 |
| | thorough | 196.0 | 136.2 | 35.8 | 1.02 | 27 / 62 |
| g_night_street (8 s, 240 frames) | fast | 32.2 | 18.3 | 9.7 | 7.46 | 7 / 32 |
| | **balanced** (default) | 148.7 | 89.1 | 26.8 | 1.61 | 31 / 66 |
| | thorough | 247.9 | 167.4 | 44.4 | 0.97 | 33 / 69 |

Render 3-21 s and QA 13-34 s per 8 s clip in every mode. Recall per mode: *Small faces* above.

Faster on weak CPUs: `--det-mode fast` for footage without far-away people, `--person-model yolox_nano` (~21 ms/frame, 3x faster), `--person-every 2`,
`--det-model det_500m`, `--qa-every 6` or `--no-qa`. All models follow `--backend cpu|dml|cuda|auto`.

## Known limitations

* Face clustering can split one person into two ids when one track is frontal and another is
  profile / back of the head (the user then selects both or runs `merge`). Body tracks do not merge
  persons automatically (an ID switch at a crossing would merge the wrong people).
* A fully hidden face is only *predicted* (body coast ≤ 1 s, body-gap bridge ≤ 2 s); in blur-selected
  this blurs the face in front of it for that time (fail-safe by design).
* QA finds sharp *faces*; an unblurred back of the head has no face to detect — check contact sheets.
* Body re-ID is a torso colour histogram: people in identical uniforms are not told apart by it
  (faces and the cannot-link rule still are).
* People seen only from far away / in a crowd without bodies detected fall back to face-only tracking.
  Faces under ~10 px with no detected body are mostly missed even in balanced (WIDER < 10 px: 10.5 % at
  score ≥ 0.5, 23 % at ≥ 0.3; thorough 17.6 / 38 %).
* Small-face modes cost time: balanced analyzes a street clip ~4x slower than fast, thorough ~6-8x.
* blur-except, guessed heads: a head predicted from a body while that face is unseen is carved out only over a
  kept face whose identity is confirmed (similarity ≥ 0.30). An unconfirmed kept face (profile, tiny, blurry)
  under such a guess is still blurred (fail-safe), and part of a blurred person's hair next to a confirmed
  kept face can stay visible where it overlaps the kept face's ellipse.
* Group body boxes (3-4 people in one YOLOX box) give drifting head guesses; where no individual body inside
  explains them they are kept and may blur empty background (cosmetic).

## Verify

`python tests/verify_leaks.py rendered.mp4 project_dir --persons 1` re-detects faces in a rendered
blur-except video and reports any visible face that is recognisable as a non-kept person.
`python tests/test_engine.py` runs quick offline unit checks (no video needed).
`python tests/eval_wider.py run|report --wider work/wider` measures detector recall on WIDER FACE val
(download instructions in the script); `python tests/semigt.py propose|eval` the hand-verified clip frames.

## Licenses — read before selling

* **Model weights (SCRFD det_10g / det_500m / det_2.5g, ArcFace w600k_mbf) are InsightFace pretrained
  models: non-commercial research use only.** The InsightFace *code* is MIT, the weights are not.
  For a paid/commercial Zoria Blur you need either a commercial license from InsightFace
  (contact via insightface.ai) or replacement models with permissive licenses (e.g. YuNet (Apache-2.0,
  OpenCV Zoo) for detection, and a face-embedding model trained on a commercially usable dataset).
  The engine only depends on ONNX input/output shapes, so swapping models is contained to
  `detector.py` / `embedder.py`.
* **Person detector: YOLOX (Megvii) — code and weights Apache-2.0, commercial use OK.** Official ONNX
  exports from the YOLOX GitHub release `0.1.1rc0` (`yolox_tiny`, `yolox_nano`, optional `yolox_s`),
  sha256-pinned in `scripts/download_models.py`. No Ultralytics/AGPL or non-commercial body weights are used.
  Body re-ID uses a torso HSV colour histogram (no extra model).
* Weights are downloaded by `scripts/download_models.py` (Hugging Face mirror for det_10g, official
  InsightFace GitHub release v0.7 `buffalo_s.zip` for det_500m + w600k_mbf) and are git-ignored.
* Python deps: numpy (BSD), OpenCV (Apache-2.0), ONNX Runtime (MIT).
* ffmpeg: use an LGPL build if you ship it with a closed-source app; libx264 makes an ffmpeg build GPL.
  Simplest for the Windows app: let the user point to / download ffmpeg separately, or ship a GPL build
  and comply (offer source).
* Test clip used in development: Mixkit stock videos #21368 and #42732 (Mixkit license), not committed.

## Windows app (ui/, Flutter) and packaging

`dist\ZoriaBlur\ZoriaBlur.exe` is a one-window Flutter app (Turkish UI, all texts in `ui/lib/strings.dart`).

**First-run setup (0.1.7).** The first launch after install (no `setup_done` in settings.json) shows a full-window setup
instead of the editor (`ui/lib/screens/setup_wizard.dart`): a stepper *1 Hoş geldiniz → 2 Sistem → 3 Hazır*, Geri / İleri,
*Şimdilik geç* in the header (the check then finishes in the background and is used). 1: app mark, one-line purpose, the
privacy line, what the check does; *Başlayalım*. 2: *Sistem kontrolü*, a live checklist fed by the engine's `devices
--json-progress` events (`ui/lib/system_check.dart`): İşlemci (model, cores / threads), Bellek, Ekran kartı (names), NVIDIA
CUDA desteği, DirectML desteği (AMD, Intel ve diğer kartlar), Deneme çalıştırması; each row Bekliyor / Kontrol ediliyor /
Bulundu (Çalışıyor, Başarılı) / Yok / Çalışmadı with a plain reason ("NVIDIA ekran kartı yok, bu adım atlandı", "Ekran
kartında bir karede yüz arama 0,016 sn sürdü (işlemcide 0,099 sn)"). Rows are revealed one at a time (≥ 380 ms apart) so
a fast PC still reads as a check; İleri waits for the last row. Engines in order: `engine-cuda` (if shipped) first, then the
main engine (with `--no-test` when CUDA already passed, so DirectML is still reported). Each engine call has a timeout
(CUDA 150 s, main 90 s); a timeout or crash marks open rows "Yanıt alınamadı" and the choice falls to the CPU with
"Ekran kartı kontrol edilemedi. Güvenli seçenek olarak işlemci kullanılacak". 3: *Bu bilgisayarda en hızlı seçenek:
Radeon Pro Graphics · DirectML*, why (CUDA / DirectML / CPU sentence), the measured test-run ratio ("yaklaşık 6 kat
hızlı… tüm işlemde fark daha küçüktür"; shown only when ≥ 1.5x), "Gelişmiş → Donanım" to change it, and the default
output folder (next to the video, or a fixed folder; `output_dir` in settings, used as the save dialog's start folder);
*Bitir* → the empty window. *Yardım → Sistem kontrolünü tekrar çalıştır* opens step 2 with a fresh check.
**Saved device choice.** The check's result is kept in settings.json (`device`: app version, engine exe + label, chosen
backend, adapter, test-run timings, fingerprint). Later starts use it at once (no "Bilgisayar kontrol ediliyor" wait) and
check it quietly in the background (`devices --only <backend> --no-test` on that engine): a different fingerprint or a
failing device runs the full check again without the setup (never while a job runs) and says "Donanım değişmiş; en hızlı
seçenek yeniden belirlendi". A new app version, a missing engine or a check that got no answer at all re-checks at start.
**One "Video aç…" (0.1.8).** The sidebar's top button was *Yeni video* and only went back to the empty start page, which
users read as "open a video". It is now **Video aç…** (folder icon; the collapsed strip has the same icon, tooltip
"Video aç…  (Ctrl+O)") and opens the video-only file dialog straight away; *İptal* leaves everything as it was. *Dosya*
has one *Video aç…* entry (Ctrl+O; Ctrl+N is kept as a second key for the same action). The start page's drop zone and
**Video seç** are unchanged. Disabled while a face search or export runs. Switching videos never loses work: each project's
chosen people and mode are saved in its *Son projeler* entry (`selected`, `mode` in settings.json) on every change, when
another video is opened and on exit, and come back when the project is reopened from the list (ids that no longer exist
are dropped).

**Risky-moment count (0.1.7 fix).** 0.1.6's done panel counted the QA's `risky` list (analysis risks recomputed at render
+ QA-only `unknown_face` / `unconfirmed_face` ranges) while the timeline showed the project's stored `analysis.risks`; the
QA-only ranges never reached the timeline (and the stored list went stale after merge / split): "5 riskli an" vs 4 marks.
Now project payloads carry freshly computed risks (`cli._current_risks`), the QA-only relevant ranges of an export join the
timeline (`AppState.qaExtra`), and the panel counts `riskMoments`, the timeline's own list.
The top bar's device chip shrinks with an ellipsis (0.1.6 overflowed by 77 px at 1024 px with a long adapter name).

**Layout (0.1.6, desktop video-tool look).** Top: a slim menu bar (*Dosya*: Video aç… Ctrl+O (0.1.8; 0.1.6 also had Yeni video Ctrl+N), Son
projeler, Bulanıklaştır ve kaydet… Ctrl+S, Çıkış; *Görünüm*: Kenar çubuğu Ctrl+B, Tema → Windows ayarı / Açık / Koyu;
*Yardım*: Klavye kısayolları, Hakkında) in one 40 px strip with the file name + length, the 1 Video → 2 Kişiler → 3 Kaydet
steps and an *Ekran kartı* chip (adapter name, full text in its tooltip). Left: a collapsible sidebar with recent projects
(thumbnail = a frame with everyone blurred, name, when, length, people; a recent project whose face search exists opens
straight into the people step via `persons PROJECT`). Centre: the preview (paused = the engine's exact frame, playing = live
blurred playback), a transport bar (first / previous frame / play / next frame / last, time, frame number) and a timeline:
time ruler with playhead, the *Kontrol edilecek anlar* band and one presence row per person (from `tracks PROJECT`, teal =
will be blurred, grey = stays visible); click or drag anywhere on it to scrub, click a band to jump. Right: the inspector
(*Ne yapılsın?*, *Bulanıklık*, the people grid with select / rename / *Birleştir* / *Ayır*, *Gelişmiş* collapsed) with
**Bulanıklaştır ve kaydet** anchored at the bottom. A 24 px status bar shows what is happening, the video's size / fps /
length and "Video bu bilgisayardan dışarı çıkmaz". Export progress and the result (QA in plain words, *Videoyu izle*,
*Klasörde göster*, *Kapat*) are a modal panel over the editor, not a page swap. Before a video: a drop zone with **Video
seç** and the last four projects; first run stays Video seç → **Başlat** → select people → **Bulanıklaştır ve kaydet**.
The picker and drag-drop take video files only (mp4 mov mkv avi webm m4v wmv mpg mpeg ts mts m2ts 3gp flv ogv; anything
else gets "Bu dosya video değil…"). *Başlat* runs the face search (progress, phase, ETA, cancel; *Gelişmiş*:
Hızlı/Dengeli/Detaylı and Otomatik/İşlemci/Ekran kartı; while the PC check runs the button waits with a note).
Keys: Space play / pause, ← → one frame, Home / End, Ctrl+O, Ctrl+N, Ctrl+S, Ctrl+B (shown in menus and tooltips).
Look: light and dark theme (follows Windows, *Görünüm → Tema* overrides; the window caption follows the app theme via
the `zoria_blur/window` channel "setDark"), neutral greys + one teal accent, Material outlined icons, Segoe UI, 4/8 px
spacing, 1 px borders and 4 px radii instead of cards and shadows; every colour comes from `Pal` in `ui/lib/theme.dart`.
Settings (theme, sidebar, recent projects) are in `%LOCALAPPDATA%\ZoriaBlur\settings.json` (`ZORIA_SETTINGS_DIR`
overrides). The runner keeps the window size / position / maximized state in `HKCU\Software\ZoriaSoft\ZoriaBlur`
(`WindowPlacement`, restored only if it is on a connected monitor; first run is centred and fits the work area), and the
window cannot get smaller than 1024×640; the layout is tested at 1360×860, 1280×720 and 1024×640 in both themes
(`ui/test/flow_test.dart`, no overflow).

**Live preview (0.1.5).** Play in the preview runs `frames VIDEO --start F --width W --height H` (raw RGBA on stdout,
same `fps=` filter and input seek as `FrameReader`, so stream frame k = renderer frame F+k; the engine starts ffmpeg
with CREATE_NO_WINDOW; scaled to fit 960×720) and paints blur ellipses from `plan PROJECT -o plan.json --mode …
--persons …` (the renderer's own `plan_regions`, each region grown to the full feathered ellipse + 2 px:
`renderer.preview_plan`). Fail-safe: kept-face carving is not applied (the preview may blur a bit more around kept
faces, never less), each frame also blurs the previous and next frame's regions, while the plan for a new selection
is computed (~1-2 s, cached per selection) the "everyone blurred" plan is shown with a note, with no plan at all the
whole frame is blurred. `tests/eval_preview_parity.py PROJECT` checks that no pixel the export blurs (alpha > 0.02)
lies outside the preview ellipses of the same frame (0 misses on work/proj and the a/b/c/e/h test-set projects, both
modes). Differences from the export: the blur look (Flutter Gaussian, sigma ~ 0.15 × strength × region radius, vs
the renderer's downscale blur), no audio, frames are dropped when decoding falls behind; paused, the app shows the
engine's exact `preview` frame. Scrubbing or drawing a box pauses playback.

**Single instance (0.1.5).** `windows/runner/main.cpp`: named mutex `Local\ZoriaSoft.ZoriaBlur.SingleInstance`; a
second launch finds the first window (window property `ZoriaBlurMain`), restores it if minimized, brings it to the
front, forwards its first file argument with WM_COPYDATA (→ method channel `zoria_blur/instance` "open") and exits.
A video given on the first launch's command line is opened after the device check.

The app spawns the engine for every step with `--json-progress` (`ui/lib/engine.dart`) and kills the whole
process tree on cancel. Engine lookup: `ZORIA_ENGINE` (exe) → `engine-cuda\zoria-blur-engine.exe` (optional
CUDA build) → `engine\zoria-blur-engine.exe` next to `ZoriaBlur.exe` → dev: `ZORIA_ENGINE_DEV=<repo>` runs
`.venv-dml` / `.venv` python `-m zoria_blur`. Projects are kept in `%LOCALAPPDATA%\ZoriaBlur\projects`,
the last engine error in `%LOCALAPPDATA%\ZoriaBlur\logs\last_error.txt`.

UI commands added to the engine: `devices`, `probe VIDEO`, `tracks PROJECT [--person N]`, `plan`, `frames` (live preview); every project
command returns `project` (absolute dir), `video` (duration, fps, frame count), `persons`, `risks`;
`preview` takes `--original-out` and, with no persons in blur-selected, writes the untouched frame; analyze /
render / qa results carry `device` (what really ran, per-model GPU/CPU run counts, fallbacks).

**Backend rule (Otomatik).** At start the app runs `devices`: a test inference (det_500m, 640x640) on
CUDA → DirectML → CPU, first that works wins (DirectML is skipped on a software-only adapter). The top bar shows
it in plain words ("Ekran kartı: Radeon (TM) Pro Graphics (DirectML)" / "İşlemci ile çalışıyor").
At run time `backend.SafeSession` never lets a provider take the job down: a session that cannot be
created moves to the next backend; a *shape* the GPU graph rejects runs on a CPU twin session (a batch is first
retried item by item); a shape that worked and then fails (device lost) moves the whole session on.
**DirectML shapes (0.1.3).** The InsightFace exports declare their *outputs* with the export shape (SCRFD
`[12800,1] ... [800,10]` = 640x640, ArcFace `[1,512]`) plus ~100 `value_info` entries with the same sizes; the input is
dynamic. CPU / CUDA ignore the stale sizes, DirectML takes them literally: every SCRFD input other than 640x640 failed in
the last Reshape (`Reshape_223`, E_INVALIDARG 80070057, at every graph optimization level) and ArcFace batches > 1 too,
so about half the det_10g calls (ROI / face-split crops) ran on the CPU twin. `zoria_blur/onnx_shapes.py` rewrites the
model bytes in memory for GPU sessions (no onnx package): free output dims, no value_info, unique input dim names;
nodes and weights untouched (CPU outputs bit-identical). DirectML also pays a shape-switch cost like CUDA (det_10g, 8
alternating shapes on the AMD test GPU: 22.6 ms/call in one session, 10.7 ms with one session per shape), so DirectML keeps
one session per input shape too, each pinned with free-dimension overrides (a static DirectML graph), created from a
shape's 3rd call on (a pinned session costs ~0.25 s; one-off shapes stay on the dynamic session), max 8 per model; the
detector uses the CUDA shape buckets on DirectML and no longer pads the full frame to 640x640; ArcFace batches are padded
to 2/4/8 on DirectML (a pinned batch-1 w600k session takes 2-3 s to create on the AMD test GPU) and 1/2/4/8 on CUDA.
`ZORIA_DISABLE_DML=1` / `ZORIA_DISABLE_CUDA=1` hide a provider (tests, broken drivers).
**DirectML adapter.** ONNX Runtime's DirectML uses DXGI adapter 0 unless told otherwise, which on a dual-GPU PC can be
the integrated GPU. `zoria_blur/dxgi.py` (ctypes, no packages) lists the DXGI adapters and DirectML gets `device_id` =
the first hardware adapter in Windows' high-performance order (IDXGIFactory6), else the one with the most dedicated
VRAM; software adapters are never picked. `ZORIA_DML_DEVICE=<index>` or `--dml-device <index>` overrides. `devices`
reports `dml_adapter` (index, name, how chosen) and `dml_adapters`; `device_name` for DirectML is that adapter's name.
**CUDA shapes.** ORT's CUDA Conv keeps the cuDNN state of the last input shape only and rebuilds it on every shape
change (~40 ms for det_10g on an NVIDIA test GPU). The analyze loop alternates shapes (full frame, small-face ROIs, tiles,
ArcFace batch 1-3), so every call paid it: det_10g ~44 ms/call instead of 4-11 ms. On CUDA `SafeSession` keeps one
session per input shape (max 8 per model, the rest share the first) and SCRFD pads ROI / face-split crops to square
buckets 160/320/480/640 (then steps of 320); the full frame and the 640 tiles keep their exact padded shape
(`ZORIA_DET_FIXED_SHAPES=1/0` forces the buckets on/off on any backend). `scripts/cuda_shape_switch_bench.py` shows the
effect in isolation.

**CUDA engine (NVIDIA).** onnxruntime-directml and onnxruntime-gpu cannot live in one Python env, so the main
build ships onnxruntime-directml (DirectML + CPU) and NVIDIA support is a second engine folder, `engine-cuda\`
(onnxruntime-gpu 1.24.4 = CUDA + CPU). The app runs `devices` on it first and uses it only if its CUDA test
inference passes; otherwise it uses the main engine (DirectML, then CPU). A CUDA failure in the middle of a job falls
back to CPU inside the CUDA engine (`SafeSession`), not to DirectML.
Build on a Windows PC (an NVIDIA card is only needed to test): `powershell -ExecutionPolicy Bypass -File
packaging\build_cuda_engine.ps1 [-Python <python3.12.exe>]` → `build\engine_cuda_dist\zoria-blur-engine\`; the next
`build_windows.ps1` run (also `-InstallerOnly`) copies it into `dist\ZoriaBlur\engine-cuda` (`-NoCuda` leaves it out).
No CUDA toolkit install: `.venv-cuda` gets the runtime from NVIDIA's pip wheels (cuda-runtime 12.9.79, cublas 12.9.2.10,
cufft 11.4.1.4, cudnn 9.10.2.21) and the script copies only the DLLs the engine loads next to
`onnxruntime_providers_cuda.dll` (`_internal\onnxruntime\capi`; `backend.py` adds that folder to the DLL search;
in a dev venv it calls `onnxruntime.preload_dlls()`). `packaging\cuda_dll_probe.py VIDEO` lists what a real
analyze + render loads. cuDNN is pinned: 9.27 fails every Conv on Turing GPUs with "Failed to initialize
CUDNN Frontend"; 9.10.2 works. The CUDA engine carries no models / ffmpeg: it uses the main engine's
(`<app>\engine\_internal`, `zoria_blur\paths.py`). cuDNN conv algorithm search is HEURISTIC
(`ZORIA_CUDA_CONV_SEARCH=EXHAUSTIVE|HEURISTIC|DEFAULT`). Size: `engine-cuda` about 2.2 GB on disk (cublasLt 638 MB,
cuDNN precompiled engines 490 MB, ORT CUDA provider 298 MB, cufft 274 MB, cudnn ops 121 MB, cublas 98 MB); left out:
cudnn_adv / cudnn_cnn, curand, nvrtc, nvJitLink, TensorRT provider. With it the setup is about 1.2 GB (vs 164 MB) and
ISCC (lzma2/max, solid) takes about 15 min. On a PC without an NVIDIA adapter `devices` skips the CUDA
probe ("no NVIDIA adapter") and the app goes on to the main engine (checked on an AMD test PC → DirectML).
`scripts\bench_backends.ps1 -Clip <mp4>` runs that comparison from a built `dist\ZoriaBlur` without installing.
NVIDIA test PC (Turing GPU + integrated Intel GPU, 2026-10-14, 0.1.1 dist, a 4 s / 240 frame test clip, analyze balanced + render
blur-except + QA): `devices` on `engine-cuda` → cuda (NVIDIA test GPU), all models on GPU, no fallbacks. Analyze engine time:
CUDA 23.9 s, DirectML 16.2 s (on the integrated Intel GPU, det_10g half on CPU twin), CPU 49.0 s; render wall incl. QA: CUDA 11.1 s,
DirectML 25.2 s, CPU 26.3 s; QA leaks 2 / 3 / 2 (same face at 0.35-0.42 s on every backend, not backend-specific).
So CUDA analyze is *slower* than DirectML-on-iGPU: det_10g on CUDA takes ~44 ms per call (14.6 s of 23.9). Not caused
by the missing nvrtc (adding nvrtc DLLs: detector the same, embedder 1.3 → 8.9 s, so nvrtc stays out; the
"Could not locate nvrtc64_120_0.dll" stderr lines are harmless). `ZORIA_CUDA_CONV_SEARCH=DEFAULT` is catastrophic
(584 s, "Conv running in Fallback mode"), HEURISTIC stays the default. Open: why det_10g is slow on CUDA (suspects:
many distinct input shapes from small-face tiles / non-square det input, each a new cuDNN plan; try fixed padded shapes
like the DirectML path). The main engine's DirectML picks adapter 0 (integrated Intel GPU) on this PC, not the NVIDIA GPU.
Correction (0.1.2 work, same day): DXGI adapter 0 on the NVIDIA test PC *is* the NVIDIA GPU (integrated Intel GPU is 1); the "integrated Intel"
label came from WMI's first adapter name, so 0.1.1 DirectML already ran on the NVIDIA GPU. Forcing the integrated Intel GPU
(`--dml-device 1`) is far slower (~40 of 241 frames in 2 min).
0.1.2 lab results (same clip, bench_backends.ps1 on the 0.1.2 dist): CUDA analyze 23.9 → 12.1 s (detector 14.6 →
5.0 s), render wall 11.1 → 10.1 s, QA leaks 2 (unchanged), 3 persons; DirectML now reports and pins the NVIDIA GPU
(`how: high_performance`): 26.6 s, but DirectML on this PC is noisy run to run (0.1.1 engine re-run back to back:
23.2 / 24.2 s, 0.1.2: 22.7 / 26.6 s) and half the det_10g calls run on the CPU twin (ROI shapes DirectML rejects);
render wall 28.3 s, leaks 3 (as 0.1.1); CPU 51.5 s, leaks 2. AMD test PC: `devices` → dml, Radeon (TM) Pro Graphics
(adapter 0, high_performance); DirectML analyze with and without device_id identical (20.8-21.1 s).
0.1.4 (DirectML shape fix of 0.1.3 + QA source check + mid-band QA metric), clip4, bench_backends.ps1 on the built
dist, analyze / render wall: NVIDIA test PC CUDA 14.1-16.2 / 11.1-12.1 s (3 runs; 0.1.2: 12.1 / 10.1), DirectML
14.1 / 11.1 s (0.1.2: 26.6 / 28.3; every det_10g / ArcFace call on the GPU now), CPU 57.8 / 28.3 s (0.1.2: 51.5 / 25.3);
AMD test PC DirectML 20.2 / 14.2 s (0.1.3: 21.3 / 14.2), CPU 52.1 / 24.3 s (0.1.3: 58.9 / 29.4). QA leaks 0 on
every backend (0.1.2: 2-3, a face-like emblem and clasped hands, see QA step 3). Render wall includes QA, which now
re-detects each candidate in the source. Open: CUDA analyze 2-4 s slower than 0.1.2 on the NVIDIA test PC (run to run 14.1 vs
16.2 s), DirectML is now faster than CUDA on the NVIDIA GPU for this clip.

**Build** (Windows, repo root): `.venv-dml` with `numpy opencv-python-headless onnxruntime-directml
pyinstaller`, `bin\` with ffmpeg/ffprobe (BtbN n8.1 GPL shared build: exe + DLLs), models downloaded, then
`powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1` → `dist\ZoriaBlur\` (PyInstaller
`--onedir --windowed` engine in `engine\` with models + ffmpeg in `_internal`, Flutter release next to it).
Package size (2026-10-14): `dist\ZoriaBlur` 469 MB, engine 441 MB. Biggest parts: ffmpeg shared build 192 MB
(avcodec alone 113 MB; exe + DLLs copied once as plain files, not through PyInstaller's DLL analysis, which doubled them),
OpenCV 82 MB, ONNX Runtime + DirectML 63 MB, models 54 MB, numpy 26 MB. The engine no longer uses scipy
(`zoria_blur/numeric.py`: Hungarian + gaussian smoothing, identical results; -65 MB) and OpenCV's own ffmpeg plugin is
dropped (-29 MB). Getting well under 400 MB needs a smaller ffmpeg (e.g. an own build with only h264/aac/mp4/mov/mkv
demuxers + libx264), not done yet.
The engine exe is a GUI-subsystem binary (no console flashes); it talks over stdout pipes only, so run it from a
shell with redirection (`Start-Process -RedirectStandardOutput`), not bare in a console.
No engine exception may reach PyInstaller's windowed bootloader (it shows a modal "Unhandled exception in script" box on
the user's desktop): `packaging/engine_entry.py` catches everything and appends it to
`%LOCALAPPDATA%\ZoriaBlur\logs\engine_crash.txt`, and a write to a closed stdout (the app closed or cancelled the job)
ends the engine quietly with exit code 3 (`cli._emit`; `tests/test_cli_ui.py` checks both). The app kills every engine
process it started when the window closes or the self-test exits (`Engine.killAll`). Seen in 0.1.6 testing: the
self-test exited while a preview job was still running and the orphaned engine showed that box.

**Installer.** The build ends with `dist\ZoriaBlur-Setup-<version>.exe` (Inno Setup 6, `packaging\installer.iss`;
`-InstallerOnly` rebuilds just the setup from an existing `dist\ZoriaBlur`, `-SkipInstaller` skips it). Version comes
from `ui\pubspec.yaml` (0.1.0+1 → file name 0.1.0, file version 0.1.0.1). Turkish by default, English selectable;
per-user into `%LOCALAPPDATA%\Programs\Zoria Blur` with no UAC prompt (the dialog or `/ALLUSERS` installs to Program
Files); Start Menu + desktop shortcut (checked); LZMA2 solid: 469 MB → 164 MB. Uninstall keeps
`%LOCALAPPDATA%\ZoriaBlur` (logs, projects) unless the user says yes at the end of an interactive uninstall.
Silent: `ZoriaBlur-Setup-0.1.0.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER [/DIR=...]`.
On a test PC with Symantec, SONAR (`SONAR.SuspScript`) blocks `unins000.exe` when a script starts it (PowerShell or a
cmd task): it hangs / exits 1 without a log. Install from a script is not affected.

**Self-test (no mouse needed).** `ZORIA_AUTOPILOT_VIDEO=<mp4>` (+ `ZORIA_AUTOPILOT_OUT`, `_SELECT=1,3,4`,
`_MODE=selected|except`, `_EXPORT`) makes the app open the video, analyze, select, preview a risky moment,
export, write `autopilot.json` and a PNG of its own window at each step, then exit. The window opens off screen and
without focus. On a build PC it must run in the logged-in user's desktop session (a service / remote shell has no
desktop, Flutter does not draw there): `schtasks /Create /TN ZoriaBlurAutopilot /TR "wscript.exe //B <launcher.vbs>"
/SC ONCE /ST 23:59 /IT /RU <DOMAIN\user from quser>`, `schtasks /Run`, then `schtasks /Delete`; the .vbs sets the
`ZORIA_AUTOPILOT_*` variables and starts `ZoriaBlur.exe` with window style 4 (no activation).
0.2.0: the export result is its own screen instead of a modal panel: the saved file plays in the app with sound (`video_player` + `video_player_win`, Windows Media Foundation), *Bulanık hali / Orijinal* switches between the result and the source at the same moment, Space and the arrow keys (5 s) drive it, the red leak moments sit on its seek bar and their buttons jump the player there; the right panel holds the file, *Klasörde göster*, *Başka oynatıcıda aç*, the final check and *Yeni video* (the main button; with a leak the main button is *Düzenlemeye dön ve düzelt*). The sidebar and drag-drop stay usable. Person thumbnails: crops are kept for every observed face, not only for faces with an embedding (score >= det-thresh and width >= min-emb-face); non-embedded crops rank below embedded ones (quality - 2), the top 24 are taken after the identity split (0.1.9 took them before, so a split part could end up with none), small crops are enlarged with cubic filtering, and a hand-marked face (`addbox`) gives its track a crop. On two crowd clips 0.1.9 left 7 of 73 and 25 of 74 people without a picture. System check: the NVIDIA driver number is read from the display-adapter class in the registry (`32.0.16.1656` -> 616.56); below 527.41 (CUDA 12.0 GA, Windows) the CUDA probe is skipped with reason `old_driver`, and the setup shows one piece of advice with a button (driver page by card maker, or the full NVIDIA installer when this install has no CUDA engine). Encoding: `render --encoder auto` (default) uses NVENC / Quick Sync / AMF when a 16-frame test encode passes, else x264; a hardware encode that fails on the real video is redone with x264 (`encoder`, `encoder_fallback` in the result). Measured on a laptop with an entry-level NVIDIA card, 1080p30 decode -> pipe -> encode: x264 60.6 fps, NVENC 74.1, Quick Sync 65.9; encoder alone: x264 107, NVENC 239, Quick Sync 163 (4K: 40 / 76 / 49). Packaging builds two installers: a light one without `engine-cuda` and `-NVIDIA` with it.
0.1.9: every outside open ("Video aç…", Ctrl+O/Ctrl+N, drag-drop, command line, a second launch) goes through `AppState.openPicked`: during the PC check the video waits and opens when the check ends (0.1.8 showed "engine missing" for a video dropped during the first check); a video already in *Son projeler* with a finished face search reopens that project with its chosen people instead of asking for a new face search, but only when it is the same file (project key with the file size, plus pixel size, frame rate and frame count (1 % slack: ffprobe estimates, the project holds the decoded count) from `persons`; a different file at that path, e.g. a camera reusing GOPR0001.MP4, opens fresh with a short note, so old tracks never cover a new video). No engine found = the setup closes and the main window shows the error (it used to wait on a check that could not start). The done panel counts leaks as the timeline's red moments (`leakMoments` = leak ranges merged only with each other, never with yellow risks; one jump button each, drawn on top). The autopilot fails when the export QA reports any leak moment (`exported.leak_moments`).
0.1.8 autopilot: after `09_start_recent` it opens the *Dosya* menu (`09b_file_menu`), reopens the project from *Son
projeler* and checks the chosen people came back (`recent_restored`, shot `10_recent_restored`; a mismatch fails the run).
0.1.7 autopilot: a fresh `ZORIA_SETTINGS_DIR` shows the setup, which it drives first (shots `w1_welcome`, `w2_checking`,
`w2_system`, `w3_ready`, rows + choice in autopilot.json `wizard_checked`), then the normal flow; with a kept settings folder
it logs `start_path: saved_choice` and the quiet check (`quiet_check: same|changed`); `ZORIA_AUTOPILOT_RERUN=1` also runs
Yardım → Sistem kontrolünü tekrar çalıştır at the end (`w4_rerun`). `exported` logs `timeline_moments` next to `qa_summary`.
0.1.6 autopilot: `ZORIA_AUTOPILOT_THEME=light|dark`, `ZORIA_AUTOPILOT_SIZE=1440x900` (logical client size) and
`ZORIA_SETTINGS_DIR` (use a scratch folder so the build PC's own recent list is not touched); after `08_done` it closes
the result, goes back to the empty window and shoots `09_start_recent` (the welcome page with the new recent project).
The self-test never saves or restores the window position.
0.1.5 autopilot also plays the live preview from frame 0, toggles one more person while playing (shots `05b_playing`,
`05c_playing_changed`), and with `ZORIA_AUTOPILOT_PICKER_DIR` first opens the real "Video seç" dialog; a UI Automation
script in the same session (the dialog is a child of the app window in the UIA tree) lists its items and closes it
with WM_COMMAND IDCANCEL. Result: a folder with jpg/png/webp/txt/mp4/mkv showed only the mp4 and the mkv.
The preview timeline merges overlapping / near (< 0.5 s) risky ranges into one moment.
