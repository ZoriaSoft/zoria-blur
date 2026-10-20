// Live preview while playing: the engine streams the original video as raw RGBA frames (`frames`, same
// CFR frame numbers as the renderer) and the app paints blur over the regions of the engine's `plan`
// (the renderer's own plan_regions, grown to the full feathered ellipse). Paused, the app shows the
// engine's exact preview frame instead.
//
// Fail-safe rules: regions of the frame before and after are added to each frame; while the plan for the
// current selection is being computed the "everyone blurred" plan is used; with no plan at all the whole
// frame is blurred.
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'engine.dart';

class LivePlan {
  LivePlan(this.frames);
  final Map<int, List<List<double>>> frames; // frame -> [cx, cy, rx, ry] in video pixels

  factory LivePlan.fromJson(Map<String, dynamic> j) {
    final out = <int, List<List<double>>>{};
    (j['frames'] as Map<String, dynamic>? ?? const {}).forEach((k, v) {
      out[int.parse(k)] = [
        for (final r in v as List) [for (final x in r as List) (x as num).toDouble()]
      ];
    });
    return LivePlan(out);
  }

  /// Regions to blur at [frame]: its own plus the neighbours' (covers a one-frame timing slip).
  List<List<double>> at(int frame) => [
        ...?frames[frame - 1],
        ...?frames[frame],
        ...?frames[frame + 1],
      ];
}

class LivePlayer extends ChangeNotifier {
  bool playing = false;
  bool loading = false; // started, first frame not shown yet
  int frame = 0;
  ui.Image? image;
  int imageW = 0, imageH = 0;
  VoidCallback? onEnded;

  Process? _proc;
  StreamSubscription<List<int>>? _sub;
  Timer? _timer;
  final _queue = <Uint8List>[];
  Uint8List? _fill;
  int _fillPos = 0;
  int _frameBytes = 0;
  int _start = 0, _shown = 0, _frameCount = 0;
  double _fps = 30;
  bool _streamDone = false, _decoding = false;
  final _clock = Stopwatch();
  int _session = 0;

  /// Display size: the video scaled to fit 960 x 720 (even sizes).
  static (int, int) displaySize(int w, int h) {
    final s = math.min(1.0, math.min(960 / w, 720 / h));
    int even(double v) => math.max(2, (v / 2).floor() * 2);
    return (even(w * s), even(h * s));
  }

  Future<void> play(Engine engine, String video,
      {required int startFrame, required int frameCount, required double fps, required int width, required int height}) async {
    await stop();
    final session = ++_session;
    final (w, h) = displaySize(width, height);
    imageW = w;
    imageH = h;
    _frameBytes = w * h * 4;
    _start = startFrame;
    _shown = 0;
    _frameCount = frameCount;
    _fps = fps > 0 ? fps : 30;
    _streamDone = false;
    frame = startFrame;
    playing = true;
    loading = true;
    notifyListeners();
    Process proc;
    try {
      proc = await engine.startRaw(['frames', video, '--start', '$startFrame', '--width', '$w', '--height', '$h']);
    } catch (_) {
      if (session == _session) {
        playing = false;
        loading = false;
        notifyListeners();
      }
      return;
    }
    if (session != _session) {
      _killTree(proc);
      return;
    }
    _proc = proc;
    unawaited(proc.stdin.close().catchError((_) {}));
    unawaited(proc.stderr.drain<void>().catchError((_) {}));
    _sub = proc.stdout.listen(_onBytes, onDone: () => _streamDone = true, onError: (_) => _streamDone = true);
    _clock
      ..reset()
      ..stop();
    _timer = Timer.periodic(const Duration(milliseconds: 4), (_) => _tick(session));
  }

  void _onBytes(List<int> chunk) {
    var off = 0;
    while (off < chunk.length) {
      _fill ??= Uint8List(_frameBytes);
      final n = math.min(_frameBytes - _fillPos, chunk.length - off);
      _fill!.setRange(_fillPos, _fillPos + n, chunk, off);
      _fillPos += n;
      off += n;
      if (_fillPos == _frameBytes) {
        _queue.add(_fill!);
        _fill = null;
        _fillPos = 0;
      }
    }
    if (_queue.length >= 8 && !(_sub?.isPaused ?? true)) _sub?.pause();
  }

  void _tick(int session) {
    if (session != _session || _decoding) return;
    if (_queue.isEmpty) {
      if (_streamDone) _finish(session);
      return;
    }
    if (!_clock.isRunning) _clock.start(); // the clock starts with the first frame
    final due = (_clock.elapsedMicroseconds * _fps / 1e6).floor();
    if (_shown > due) return;
    // behind schedule: drop frames (the blur is drawn per frame number, so dropping is safe)
    while (_queue.length > 1 && _shown < due) {
      _queue.removeAt(0);
      _shown++;
    }
    final bytes = _queue.removeAt(0);
    final idx = _start + _shown;
    _shown++;
    if (_queue.length <= 3 && (_sub?.isPaused ?? false)) _sub?.resume();
    _decoding = true;
    ui.decodeImageFromPixels(bytes, imageW, imageH, ui.PixelFormat.rgba8888, (img) {
      _decoding = false;
      if (session != _session) {
        img.dispose();
        return;
      }
      final old = image;
      image = img;
      frame = math.min(idx, math.max(0, _frameCount - 1));
      loading = false;
      notifyListeners();
      old?.dispose();
    });
  }

  void _finish(int session) {
    if (session != _session) return;
    _timer?.cancel();
    playing = false;
    loading = false;
    notifyListeners();
    onEnded?.call();
  }

  Future<void> stop() async {
    _session++;
    _timer?.cancel();
    _timer = null;
    await _sub?.cancel();
    _sub = null;
    final p = _proc;
    _proc = null;
    if (p != null) _killTree(p);
    _queue.clear();
    _fill = null;
    _fillPos = 0;
    _decoding = false;
    if (playing || loading) {
      playing = false;
      loading = false;
      notifyListeners();
    }
  }

  void _killTree(Process p) {
    if (Platform.isWindows) {
      Process.run('taskkill', ['/T', '/F', '/PID', '${p.pid}']).catchError((_) => ProcessResult(0, 0, '', ''));
    } else {
      p.kill(ProcessSignal.sigkill);
    }
  }

  @override
  void dispose() {
    stop();
    image?.dispose();
    super.dispose();
  }
}

/// Paints the current frame (contain-fit) and blurs every region; [regions] == null = blur everything.
class LiveFramePainter extends CustomPainter {
  LiveFramePainter({required this.image, required this.regions, required this.videoW, required this.videoH,
      required this.strength});
  final ui.Image image;
  final List<List<double>>? regions;
  final int videoW, videoH;
  final double strength;

  @override
  void paint(Canvas canvas, Size size) {
    if (videoW <= 0 || videoH <= 0) return;
    final s = math.min(size.width / videoW, size.height / videoH);
    final dst = Rect.fromLTWH((size.width - videoW * s) / 2, (size.height - videoH * s) / 2, videoW * s, videoH * s);
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final base = Paint()..filterQuality = FilterQuality.medium;
    final rs = regions;
    if (rs == null) {
      canvas.drawImageRect(image, src, dst, base..imageFilter = ui.ImageFilter.blur(sigmaX: 24, sigmaY: 24, tileMode: TileMode.clamp));
      return;
    }
    canvas.drawImageRect(image, src, dst, base);
    canvas.save();
    canvas.clipRect(dst);
    final kx = image.width / videoW, ky = image.height / videoH; // video px -> image px
    for (final r in rs) {
      final oval = Rect.fromCenter(
          center: Offset(dst.left + r[0] * s, dst.top + r[1] * s), width: 2 * r[2] * s, height: 2 * r[3] * s);
      if (!oval.overlaps(dst)) continue;
      // face width on screen ~ rx / 1.1 (rx includes padding + feather); renderer: ~11/strength px across a face
      final sigma = math.max(6.0, r[2] * s * 0.15 * strength);
      final pad = 3 * sigma;
      final area = oval.inflate(pad).intersect(dst);
      final srcArea = Rect.fromLTRB((area.left - dst.left) / s * kx, (area.top - dst.top) / s * ky,
          (area.right - dst.left) / s * kx, (area.bottom - dst.top) / s * ky);
      canvas.save();
      canvas.clipPath(Path()..addOval(oval));
      canvas.drawImageRect(image, srcArea, area,
          Paint()..imageFilter = ui.ImageFilter.blur(sigmaX: sigma, sigmaY: sigma, tileMode: TileMode.clamp));
      canvas.restore();
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(LiveFramePainter old) =>
      old.image != image || old.regions != regions || old.strength != strength;
}
