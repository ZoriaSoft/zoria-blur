import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../strings.dart';
import '../theme.dart';
import 'preview_pane.dart';

/// One person's row on the timeline: when they are on screen.
class TimelineLane {
  const TimelineLane({
    required this.id,
    required this.label,
    required this.segments,
    this.thumbnail,
    this.willBlur = false,
    this.selected = false,
  });
  final int id;
  final String label;
  final List<List<double>> segments; // [[start_s, end_s], ...]
  final String? thumbnail;
  final bool willBlur;
  final bool selected;
}

/// Editor timeline under the preview: time ruler with the playhead, the "Kontrol edilecek anlar" band and
/// one presence row per person. Click or drag anywhere on the rows to scrub; click a band to jump to it.
class Timeline extends StatelessWidget {
  const Timeline({
    super.key,
    required this.fps,
    required this.frameCount,
    required this.frame,
    required this.onFrameChanged,
    required this.onFrameChangeEnd,
    this.marks = const [],
    this.lanes = const [],
    this.maxVisibleLanes = 4,
  });

  final double fps;
  final int frameCount;
  final int frame;
  final ValueChanged<int> onFrameChanged;
  final ValueChanged<int> onFrameChangeEnd;
  final List<RiskMark> marks;
  final List<TimelineLane> lanes;
  final int maxVisibleLanes;

  static const labelWidth = 176.0;
  static const rulerHeight = 24.0;
  static const riskHeight = 24.0;
  static const laneHeight = 22.0;
  static const pad = 8.0; // inner horizontal padding of the time area (marks at the very ends stay visible)

  int get _maxFrame => math.max(0, frameCount - 1);
  double get _fps => fps.isFinite && fps > 0 ? fps : 0;
  double get _duration => _fps > 0 ? _maxFrame / _fps : 0;
  double get _position => _maxFrame == 0 ? 0 : frame.clamp(0, _maxFrame) / _maxFrame;

  double _fractionAtSeconds(double s) {
    if (_fps == 0 || _maxFrame == 0 || s.isNaN || s <= 0) return 0;
    return (s * _fps / _maxFrame).clamp(0.0, 1.0);
  }

  int _frameAtSeconds(double s) {
    if (_fps == 0 || _maxFrame == 0 || s.isNaN || s <= 0) return 0;
    final f = s * _fps;
    if (!f.isFinite || f >= _maxFrame) return _maxFrame;
    return f.round().clamp(0, _maxFrame);
  }

  int _frameAtX(double x, double width) {
    final inner = width - 2 * pad;
    if (inner <= 0 || _maxFrame == 0) return 0;
    return (((x - pad) / inner).clamp(0.0, 1.0) * _maxFrame).round();
  }

  double _x(double fraction, double width) => pad + fraction * math.max(0.0, width - 2 * pad);

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final visibleLanes = math.min(lanes.length, maxVisibleLanes);
    final lanesHeight = visibleLanes * laneHeight;
    return Container(
      decoration: BoxDecoration(color: c.surface, border: Border(top: BorderSide(color: c.border))),
      child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        _row(
          context,
          height: rulerHeight,
          label: Text(S.timeline, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: c.muted)),
          key: const ValueKey('preview-timeline'),
          body: (w) => Stack(children: [
            Positioned.fill(child: CustomPaint(painter: _RulerPainter(duration: _duration, pad: pad, pal: c))),
            _playhead(context, w, rulerHeight, key: const ValueKey('preview-current-marker'), handle: true),
          ]),
        ),
        _row(
          context,
          height: riskHeight,
          label: Row(children: [
            Flexible(
              child: Text(marks.isEmpty ? S.noRiskyMoments : S.riskyMoments,
                  maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: c.text)),
            ),
            if (marks.isNotEmpty) ...[
              const SizedBox(width: Sp.xs + 2),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                decoration: BoxDecoration(color: c.warnSoft, borderRadius: BorderRadius.circular(3)),
                child: Text('${marks.length}',
                    key: const ValueKey('timeline-risk-count'),
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: c.warn)),
              ),
            ],
          ]),
          body: (w) => Stack(children: [
            Positioned(
                left: pad, right: pad, top: riskHeight / 2 - 0.5, height: 1, child: ColoredBox(color: c.border)),
            for (var i = 0; i < marks.length; i++) _mark(context, marks[i], i, w),
            _playhead(context, w, riskHeight),
          ]),
        ),
        if (lanes.isNotEmpty)
          SizedBox(
            height: lanesHeight,
            child: ListView.builder(
              padding: EdgeInsets.zero,
              itemExtent: laneHeight,
              itemCount: lanes.length,
              itemBuilder: (context, i) => _lane(context, lanes[i]),
            ),
          ),
      ]),
    );
  }

  Widget _row(BuildContext context,
      {required double height, required Widget label, required Widget Function(double width) body, Key? key}) {
    final c = context.pal;
    return SizedBox(
      height: height,
      child: Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Container(
          width: labelWidth,
          padding: const EdgeInsets.symmetric(horizontal: Sp.m),
          alignment: Alignment.centerLeft,
          decoration: BoxDecoration(
            color: c.surfaceAlt,
            border: Border(right: BorderSide(color: c.border), bottom: BorderSide(color: c.border)),
          ),
          child: label,
        ),
        Expanded(
          child: DecoratedBox(
            decoration: BoxDecoration(border: Border(bottom: BorderSide(color: c.border))),
            child: LayoutBuilder(
              key: key,
              builder: (context, cons) => _scrubber(cons.maxWidth, body(cons.maxWidth)),
            ),
          ),
        ),
      ]),
    );
  }

  Widget _scrubber(double width, Widget child) => MouseRegion(
        cursor: SystemMouseCursors.click,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTapUp: _maxFrame == 0 ? null : (d) => onFrameChangeEnd(_frameAtX(d.localPosition.dx, width)),
          onHorizontalDragStart: _maxFrame == 0 ? null : (d) => onFrameChanged(_frameAtX(d.localPosition.dx, width)),
          onHorizontalDragUpdate:
              _maxFrame == 0 ? null : (d) => onFrameChanged(_frameAtX(d.localPosition.dx, width)),
          onHorizontalDragEnd: _maxFrame == 0
              ? null
              : (d) => onFrameChangeEnd(_frameAtX(d.localPosition.dx, width)),
          child: child,
        ),
      );

  Widget _playhead(BuildContext context, double width, double height, {Key? key, bool handle = false}) {
    final c = context.pal;
    final x = _x(_position, width);
    return Positioned(
      left: (x - (handle ? 5 : 0.75)).clamp(0.0, math.max(0.0, width - 1.5)).toDouble(),
      top: 0,
      bottom: 0,
      width: handle ? 10 : 1.5,
      child: IgnorePointer(
        child: handle
            ? CustomPaint(key: key, painter: _HandlePainter(c.accent))
            : ColoredBox(key: key, color: c.accent),
      ),
    );
  }

  Widget _mark(BuildContext context, RiskMark mark, int index, double width) {
    final c = context.pal;
    final a = _x(_fractionAtSeconds(mark.startS), width), b = _x(_fractionAtSeconds(mark.endS), width);
    final w = math.max(4.0, (b - a).abs()).clamp(0.0, width).toDouble();
    final left = math.min(a, b).clamp(0.0, math.max(0.0, width - w)).toDouble();
    final tip = '${mark.label}\n${formatTime(mark.startS)} – ${formatTime(mark.endS)}';
    return Positioned(
      left: left,
      top: 5,
      width: w,
      height: riskHeight - 10,
      child: Tooltip(
        message: tip,
        child: Semantics(
          label: tip,
          button: true,
          child: GestureDetector(
            key: ValueKey('preview-risk-mark-$index'),
            behavior: HitTestBehavior.opaque,
            onTap: () => onFrameChangeEnd(_frameAtSeconds(mark.startS)),
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: mark.severe ? c.riskSevere : c.risk.withValues(alpha: 0.85),
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _lane(BuildContext context, TimelineLane lane) {
    final c = context.pal;
    return _row(
      context,
      height: laneHeight,
      key: ValueKey('timeline-lane-${lane.id}'),
      label: Row(children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: SizedBox(
            width: 16,
            height: 16,
            child: lane.thumbnail == null
                ? ColoredBox(color: c.hover)
                : Image.file(File(lane.thumbnail!),
                    fit: BoxFit.cover,
                    cacheWidth: 32,
                    errorBuilder: (_, _, _) => ColoredBox(color: c.hover)),
          ),
        ),
        const SizedBox(width: Sp.s),
        Expanded(
          child: Text(lane.label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                  fontSize: 12, color: c.text, fontWeight: lane.selected ? FontWeight.w600 : FontWeight.w400)),
        ),
        Tooltip(
          message: lane.willBlur ? S.willBlur : S.visible,
          child: Icon(lane.willBlur ? Icons.blur_on : Icons.visibility_outlined,
              size: 14, color: lane.willBlur ? c.accentText : c.faint),
        ),
      ]),
      body: (w) => Stack(children: [
        for (final s in lane.segments)
          if (s.length == 2) _segment(context, s[0], s[1], w, lane.willBlur),
        _playhead(context, w, laneHeight),
      ]),
    );
  }

  Widget _segment(BuildContext context, double a, double b, double width, bool blur) {
    final c = context.pal;
    final x0 = _x(_fractionAtSeconds(a), width), x1 = _x(_fractionAtSeconds(b), width);
    final w = math.max(3.0, x1 - x0);
    return Positioned(
      left: x0.clamp(0.0, math.max(0.0, width - w)).toDouble(),
      top: 6,
      height: laneHeight - 12,
      width: w,
      child: DecoratedBox(
        decoration: BoxDecoration(color: blur ? c.presenceBlur : c.presence, borderRadius: BorderRadius.circular(2)),
      ),
    );
  }
}

class _RulerPainter extends CustomPainter {
  _RulerPainter({required this.duration, required this.pad, required this.pal});
  final double duration, pad;
  final Pal pal;

  static const _steps = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0];

  @override
  void paint(Canvas canvas, Size size) {
    final inner = size.width - 2 * pad;
    if (inner <= 0 || duration <= 0) return;
    final pxPerS = inner / duration;
    final step = _steps.firstWhere((s) => s * pxPerS >= 72, orElse: () => _steps.last);
    final minor = step / 5;
    final tick = Paint()
      ..color = pal.borderStrong
      ..strokeWidth = 1;
    for (var t = 0.0; t <= duration + 1e-6; t += minor) {
      final x = pad + t * pxPerS;
      final major = ((t / step) - (t / step).round()).abs() < 1e-6;
      canvas.drawLine(Offset(x, size.height - (major ? 8 : 4)), Offset(x, size.height), tick);
      if (major) {
        final tp = TextPainter(
          text: TextSpan(
              text: _fmt(t, step),
              style: TextStyle(
                  fontSize: 10.5,
                  color: pal.muted,
                  fontFamily: 'Segoe UI',
                  fontFeatures: const [FontFeature.tabularFigures()])),
          textDirection: TextDirection.ltr,
        )..layout();
        final lx = (x + 3).clamp(0.0, math.max(0.0, size.width - tp.width - 2)).toDouble();
        if (lx + tp.width <= size.width) tp.paint(canvas, Offset(lx, 3));
        tp.dispose();
      }
    }
  }

  String _fmt(double t, double step) {
    final m = t ~/ 60, s = t - m * 60;
    final ss = step < 1 ? s.toStringAsFixed(1).padLeft(4, '0') : s.round().toString().padLeft(2, '0');
    return '$m:$ss';
  }

  @override
  bool shouldRepaint(_RulerPainter old) => old.duration != duration || old.pad != pad || old.pal != pal;
}

class _HandlePainter extends CustomPainter {
  _HandlePainter(this.color);
  final Color color;
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()..color = color;
    final cx = size.width / 2;
    final path = Path()
      ..moveTo(cx - 5, 0)
      ..lineTo(cx + 5, 0)
      ..lineTo(cx + 5, 6)
      ..lineTo(cx, 11)
      ..lineTo(cx - 5, 6)
      ..close();
    canvas.drawPath(path, p);
    canvas.drawRect(Rect.fromLTWH(cx - 0.75, 0, 1.5, size.height), p);
  }

  @override
  bool shouldRepaint(_HandlePainter old) => old.color != color;
}
