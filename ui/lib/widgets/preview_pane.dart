import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../strings.dart';
import '../theme.dart';

class RiskMark {
  final double startS, endS;
  final String label;
  final bool severe;

  const RiskMark({
    required this.startS,
    required this.endS,
    required this.label,
    this.severe = false,
  });
}

/// Maps a selection in a contain-fitted preview to original video pixels.
///
/// Coordinates outside the displayed image are clamped to its edges. A reversed
/// selection is normalized, and an invalid viewport or video yields [Rect.zero].
Rect mapToVideo(Rect screenRect, Size boxSize, int videoW, int videoH) {
  final imageRect = _displayedImageRect(boxSize, videoW, videoH);
  if (imageRect.isEmpty ||
      !screenRect.left.isFinite ||
      !screenRect.top.isFinite ||
      !screenRect.right.isFinite ||
      !screenRect.bottom.isFinite) {
    return Rect.zero;
  }

  final selection = _clampSelection(screenRect, imageRect);
  double videoX(double x) => ((x - imageRect.left) / imageRect.width * videoW)
      .clamp(0.0, videoW.toDouble());
  double videoY(double y) => ((y - imageRect.top) / imageRect.height * videoH)
      .clamp(0.0, videoH.toDouble());

  return Rect.fromLTRB(
    videoX(selection.left),
    videoY(selection.top),
    videoX(selection.right),
    videoY(selection.bottom),
  );
}

/// Formats nonnegative seconds as m:ss.d, rounding with carries to minutes.
String formatTime(double seconds) {
  final tenths = _safeSeconds(seconds) * 10;
  final totalTenths = tenths.isFinite ? tenths.round() : 0;
  final minutes = totalTenths ~/ 600;
  final wholeSeconds = (totalTenths ~/ 10 % 60).toString().padLeft(2, '0');
  return '$minutes:$wholeSeconds.${totalTenths % 10}';
}

double _safeSeconds(double seconds) =>
    seconds.isFinite && seconds > 0 ? seconds : 0;

String _formatWholeTime(double seconds) => formatTime(seconds).split('.').first;

Rect _displayedImageRect(Size size, int videoW, int videoH) {
  if (!size.width.isFinite ||
      !size.height.isFinite ||
      size.width <= 0 ||
      size.height <= 0 ||
      videoW <= 0 ||
      videoH <= 0) {
    return Rect.zero;
  }
  final scale = math.min(size.width / videoW, size.height / videoH);
  final width = videoW * scale;
  final height = videoH * scale;
  return Rect.fromLTWH(
    (size.width - width) / 2,
    (size.height - height) / 2,
    width,
    height,
  );
}

Rect _clampSelection(Rect selection, Rect bounds) => Rect.fromLTRB(
  math.min(selection.left, selection.right).clamp(bounds.left, bounds.right),
  math.min(selection.top, selection.bottom).clamp(bounds.top, bounds.bottom),
  math.max(selection.left, selection.right).clamp(bounds.left, bounds.right),
  math.max(selection.top, selection.bottom).clamp(bounds.top, bounds.bottom),
);

class PreviewPane extends StatefulWidget {
  const PreviewPane({
    super.key,
    required this.image,
    required this.videoWidth,
    required this.videoHeight,
    required this.fps,
    required this.frameCount,
    required this.frame,
    required this.onFrameChanged,
    required this.onFrameChangeEnd,
    this.busy = false,
    this.drawMode = false,
    this.onBoxDrawn,
    this.live,
    this.playing = false,
    this.onPlayPause,
  });

  final Uint8List? image;
  final int videoWidth, videoHeight;
  final double fps;
  final int frameCount;
  final int frame;

  /// Continuous changes (kept for API symmetry with the timeline; the transport only commits).
  final ValueChanged<int> onFrameChanged;
  final ValueChanged<int> onFrameChangeEnd;
  final bool busy;
  final bool drawMode;
  final void Function(Rect boxInVideoPixels)? onBoxDrawn;

  /// Shown instead of [image] while the video plays (live blurred frames).
  final Widget? live;
  final bool playing;

  /// Play / pause button in the transport bar (hidden when null).
  final VoidCallback? onPlayPause;

  @override
  State<PreviewPane> createState() => _PreviewPaneState();
}

class _PreviewPaneState extends State<PreviewPane> {
  Offset? _dragOrigin;
  Offset? _dragPosition;

  int get _maxFrame => math.max(0, widget.frameCount - 1);
  int get _currentFrame => widget.frame.clamp(0, _maxFrame);
  double get _fps => widget.fps.isFinite && widget.fps > 0 ? widget.fps : 0;
  double _secondsAt(int frame) => _fps > 0 ? _safeSeconds(frame / _fps) : 0;

  @override
  void didUpdateWidget(covariant PreviewPane oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!widget.drawMode ||
        oldWidget.videoWidth != widget.videoWidth ||
        oldWidget.videoHeight != widget.videoHeight) {
      _dragOrigin = null;
      _dragPosition = null;
    }
  }

  void _startDraw(DragDownDetails details, Size size) {
    final imageRect = _displayedImageRect(size, widget.videoWidth, widget.videoHeight);
    if (imageRect.isEmpty || !imageRect.contains(details.localPosition)) return;
    setState(() {
      _dragOrigin = details.localPosition;
      _dragPosition = details.localPosition;
    });
  }

  void _updateDraw(DragUpdateDetails details) {
    if (_dragOrigin == null) return;
    setState(() => _dragPosition = details.localPosition);
  }

  void _cancelDraw() {
    if (_dragOrigin == null && _dragPosition == null) return;
    setState(() {
      _dragOrigin = null;
      _dragPosition = null;
    });
  }

  Rect? _selection(Size size) {
    final origin = _dragOrigin;
    final position = _dragPosition;
    if (origin == null || position == null) return null;
    return _clampSelection(
      Rect.fromPoints(origin, position),
      _displayedImageRect(size, widget.videoWidth, widget.videoHeight),
    );
  }

  void _finishDraw(Size size) {
    final selection = _selection(size);
    _cancelDraw();
    if (selection == null || selection.width < 8 || selection.height < 8) return;
    widget.onBoxDrawn?.call(mapToVideo(selection, size, widget.videoWidth, widget.videoHeight));
  }

  void _go(int f) => widget.onFrameChangeEnd(f.clamp(0, _maxFrame));

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final c = context.pal;
    final timeText = '${formatTime(_secondsAt(_currentFrame))} / ${_formatWholeTime(_secondsAt(_maxFrame))}';
    final mono = TextStyle(fontSize: 12.5, color: c.text, fontFeatures: const [FontFeature.tabularFigures()]);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(child: _buildImageArea(context, colors, theme)),
        Container(
          height: 40,
          padding: const EdgeInsets.symmetric(horizontal: Sp.s),
          decoration: BoxDecoration(
            color: c.surface,
            border: Border(top: BorderSide(color: c.border)),
          ),
          child: Directionality(
            textDirection: TextDirection.ltr,
            child: Row(children: [
              _TransportButton(
                key: const ValueKey('preview-first'),
                icon: Icons.first_page,
                tooltip: '${S.firstFrame}  (Home)',
                onPressed: () => _go(0),
              ),
              _TransportButton(
                key: const ValueKey('preview-step-back'),
                icon: Icons.chevron_left,
                tooltip: '${S.prevFrame}  (\u2190)',
                onPressed: () => _go(_currentFrame - 1),
              ),
              if (widget.onPlayPause != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: Sp.xs),
                  child: Tooltip(
                    message: '${widget.playing ? S.pause : S.play}  (Space)',
                    child: IconButton(
                      key: const ValueKey('preview-play'),
                      onPressed: widget.onPlayPause,
                      style: IconButton.styleFrom(
                        minimumSize: const Size(38, 30),
                        padding: EdgeInsets.zero,
                        backgroundColor: c.accentSoft,
                        foregroundColor: c.accentText,
                      ),
                      icon: Icon(widget.playing ? Icons.pause : Icons.play_arrow, size: 20),
                    ),
                  ),
                ),
              _TransportButton(
                key: const ValueKey('preview-step-forward'),
                icon: Icons.chevron_right,
                tooltip: '${S.nextFrame}  (\u2192)',
                onPressed: () => _go(_currentFrame + 1),
              ),
              _TransportButton(
                key: const ValueKey('preview-last'),
                icon: Icons.last_page,
                tooltip: '${S.lastFrame}  (End)',
                onPressed: () => _go(_maxFrame),
              ),
              const SizedBox(width: Sp.m),
              Flexible(
                flex: 3,
                child: Text(timeText,
                    key: const ValueKey('preview-time-label'), style: mono, maxLines: 1, overflow: TextOverflow.clip),
              ),
              const SizedBox(width: Sp.m),
              Expanded(
                flex: 2,
                child: Text(
                    '${S.frameLabel} ${_currentFrame + (widget.frameCount > 0 ? 1 : 0)} / ${math.max(0, widget.frameCount)}',
                    textAlign: TextAlign.right,
                    style: mono.copyWith(color: c.muted),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis),
              ),
            ]),
          ),
        ),
      ],
    );
  }

  Widget _buildImageArea(BuildContext context, ColorScheme colors, ThemeData theme) => ClipRect(
    child: ColoredBox(
      color: context.pal.canvas,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final size = constraints.biggest;
          final selection = _selection(size);
          return MouseRegion(
            cursor: widget.drawMode
                ? SystemMouseCursors.precise
                : MouseCursor.defer,
            child: GestureDetector(
              key: const ValueKey('preview-image-area'),
              behavior: HitTestBehavior.opaque,
              onPanDown: widget.drawMode
                  ? (details) => _startDraw(details, size)
                  : null,
              onPanUpdate: widget.drawMode ? _updateDraw : null,
              onPanEnd: widget.drawMode ? (_) => _finishDraw(size) : null,
              onPanCancel: widget.drawMode ? _cancelDraw : null,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (widget.live != null)
                    widget.live!
                  else if (widget.image == null)
                    _placeholder(colors)
                  else
                    Image.memory(
                      widget.image!,
                      fit: BoxFit.contain,
                      gaplessPlayback: true,
                      errorBuilder: (context, error, stackTrace) =>
                          _placeholder(colors),
                    ),
                  if (selection != null)
                    Positioned.fromRect(
                      rect: selection,
                      child: IgnorePointer(
                        child: DecoratedBox(
                          key: const ValueKey('preview-selection'),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.14),
                            border: Border.all(color: Colors.white, width: 2),
                          ),
                        ),
                      ),
                    ),
                  if (widget.drawMode)
                    Positioned(
                      top: 12,
                      left: 12,
                      right: 12,
                      child: IgnorePointer(
                        child: Align(
                          alignment: Alignment.topCenter,
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.66),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 7,
                              ),
                              child: Text(
                                S.drawBoxHint,
                                style: const TextStyle(color: Colors.white, fontSize: 12.5, fontWeight: FontWeight.w600),
                                textAlign: TextAlign.center,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  if (widget.busy)
                    Positioned(
                      top: 12,
                      right: 12,
                      child: IgnorePointer(
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.55),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: const Padding(
                            padding: EdgeInsets.all(6),
                            child: SizedBox.square(
                              dimension: 16,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      ),
    ),
  );

  Widget _placeholder(ColorScheme colors) => Center(
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: Text(
        S.previewEmpty,
        textAlign: TextAlign.center,
        style: const TextStyle(color: Color(0xFF9AA1AB), fontSize: 13),
      ),
    ),
  );
}

class _TransportButton extends StatelessWidget {
  const _TransportButton({super.key, required this.icon, required this.tooltip, required this.onPressed});
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;
  @override
  Widget build(BuildContext context) => IconButton(
        tooltip: tooltip,
        onPressed: onPressed,
        icon: Icon(icon, size: 20),
        color: context.pal.text,
        style: IconButton.styleFrom(minimumSize: const Size(30, 30), padding: const EdgeInsets.all(4)),
      );
}
