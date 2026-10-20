import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zoria_blur/strings.dart';
import 'package:zoria_blur/theme.dart';
import 'package:zoria_blur/widgets/preview_pane.dart';

const _imageKey = ValueKey('preview-image-area');


Future<Uint8List> _png({int width = 1, int height = 1}) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  canvas.drawColor(const Color(0xFF708090), BlendMode.src);
  final picture = recorder.endRecording();
  final image = await picture.toImage(width, height);
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  picture.dispose();
  return data!.buffer.asUint8List();
}

Future<void> _pumpPreview(
  WidgetTester tester, {
  Uint8List? image,
  int videoWidth = 1920,
  int videoHeight = 1080,
  double fps = 30,
  int frameCount = 301,
  int frame = 0,
  bool busy = false,
  bool drawMode = false,
  ValueChanged<int>? onFrameChanged,
  ValueChanged<int>? onFrameChangeEnd,
  ValueChanged<Rect>? onBoxDrawn,
  VoidCallback? onPlayPause,
}) async {
  await tester.pumpWidget(
    MaterialApp(
      theme: buildTheme(),
      home: Scaffold(
        body: PreviewPane(
          image: image,
          videoWidth: videoWidth,
          videoHeight: videoHeight,
          fps: fps,
          frameCount: frameCount,
          frame: frame,
          onFrameChanged: onFrameChanged ?? (_) {},
          onFrameChangeEnd: onFrameChangeEnd ?? (_) {},
          busy: busy,
          drawMode: drawMode,
          onBoxDrawn: onBoxDrawn,
          onPlayPause: onPlayPause,
        ),
      ),
    ),
  );
}

void _expectRectClose(Rect actual, Rect expected) {
  expect(actual.left, closeTo(expected.left, 0.0001));
  expect(actual.top, closeTo(expected.top, 0.0001));
  expect(actual.right, closeTo(expected.right, 0.0001));
  expect(actual.bottom, closeTo(expected.bottom, 0.0001));
}

void main() {
  group('mapToVideo', () {
    test('accounts for horizontal letterboxing around a tall video', () {
      final result = mapToVideo(
        const Rect.fromLTRB(160, 20, 240, 180),
        const Size(400, 200),
        100,
        200,
      );
      expect(result, const Rect.fromLTRB(10, 20, 90, 180));
    });

    test('accounts for vertical letterboxing around a wide video', () {
      final result = mapToVideo(
        const Rect.fromLTRB(20, 160, 180, 240),
        const Size(200, 400),
        200,
        100,
      );
      expect(result, const Rect.fromLTRB(20, 10, 180, 90));
    });

    test('scales coordinates to original video pixels', () {
      final result = mapToVideo(
        const Rect.fromLTRB(120, 110, 360, 190),
        const Size(480, 300),
        1920,
        640,
      );
      expect(result, const Rect.fromLTRB(480, 160, 1440, 480));
    });

    test('clamps partly and completely out-of-image rectangles', () {
      expect(
        mapToVideo(
          const Rect.fromLTRB(-20, -20, 500, 300),
          const Size(400, 200),
          100,
          200,
        ),
        const Rect.fromLTRB(0, 0, 100, 200),
      );
      expect(
        mapToVideo(
          const Rect.fromLTRB(20, 20, 100, 80),
          const Size(400, 200),
          100,
          200,
        ),
        const Rect.fromLTRB(0, 20, 0, 80),
      );
    });

    test('normalizes a rectangle drawn in the reverse direction', () {
      expect(
        mapToVideo(
          const Rect.fromLTRB(240, 180, 160, 20),
          const Size(400, 200),
          100,
          200,
        ),
        const Rect.fromLTRB(10, 20, 90, 180),
      );
    });

    test('returns an empty rectangle for invalid geometry', () {
      const screen = Rect.fromLTRB(10, 10, 20, 20);
      for (final size in [
        Size.zero,
        const Size(-100, 100),
        const Size(double.infinity, 100),
        const Size(100, double.nan),
      ]) {
        expect(mapToVideo(screen, size, 100, 100), Rect.zero);
      }
      expect(mapToVideo(screen, const Size(100, 100), 0, 100), Rect.zero);
      expect(mapToVideo(screen, const Size(100, 100), 100, -1), Rect.zero);
    });
  });

  group('formatTime', () {
    test('formats padded seconds and a single decimal place', () {
      expect(formatTime(0), '0:00.0');
      expect(formatTime(4.3), '0:04.3');
      expect(formatTime(4.26), '0:04.3');
      expect(formatTime(65), '1:05.0');
      expect(formatTime(3600), '60:00.0');
    });

    test('carries rounded tenths into the next minute', () {
      expect(formatTime(59.96), '1:00.0');
      expect(formatTime(119.96), '2:00.0');
    });

    test('handles negative and nonfinite seconds', () {
      expect(formatTime(-1), '0:00.0');
      expect(formatTime(double.nan), '0:00.0');
      expect(formatTime(double.infinity), '0:00.0');
    });
  });

  testWidgets('empty preview: placeholder, time label, transport keys', (tester) async {
    await _pumpPreview(tester, onPlayPause: () {});
    expect(find.text(S.previewEmpty), findsOneWidget);
    expect(find.text('0:00.0 / 0:10'), findsOneWidget);
    for (final k in ['preview-play', 'preview-first', 'preview-step-back', 'preview-step-forward', 'preview-last']) {
      expect(find.byKey(ValueKey(k)), findsOneWidget);
    }
    expect(tester.takeException(), isNull);
  });

  testWidgets('first / last buttons jump to the ends', (tester) async {
    final ended = <int>[];
    await _pumpPreview(tester, frame: 100, onFrameChangeEnd: ended.add);
    await tester.tap(find.byKey(const ValueKey('preview-last')));
    await tester.tap(find.byKey(const ValueKey('preview-first')));
    await tester.pump();
    expect(ended, [300, 0]);
  });

  testWidgets('loads a valid PNG with gapless playback and busy indicator', (
    tester,
  ) async {
    final png = (await tester.runAsync(_png))!;
    await _pumpPreview(tester, image: png, busy: true);
    await tester.pump();

    expect(find.text(S.previewEmpty), findsNothing);
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    final image = tester.widget<Image>(find.byType(Image));
    expect(image.gaplessPlayback, isTrue);
    expect(image.fit, BoxFit.contain);
    await tester.pump(const Duration(milliseconds: 100));
    expect(tester.takeException(), isNull);

    await _pumpPreview(tester, image: Uint8List.fromList(png));
    await tester.pump();
    expect(find.byType(Image), findsOneWidget);
    expect(find.byType(CircularProgressIndicator), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('rounds current and total time consistently at a minute', (
    tester,
  ) async {
    await _pumpPreview(
      tester,
      fps: 25,
      frameCount: 1500,
      frame: 1499,
    );

    expect(find.text('1:00.0 / 1:00'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('step buttons use the controlled frame and clamp at both ends', (
    tester,
  ) async {
    var frame = 0;
    final ended = <int>[];
    final changed = <int>[];
    late StateSetter update;

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StatefulBuilder(
            builder: (context, setState) {
              update = setState;
              return PreviewPane(
                image: null,
                videoWidth: 1920,
                videoHeight: 1080,
                fps: 30,
                frameCount: 3,
                frame: frame,
                onFrameChanged: changed.add,
                onFrameChangeEnd: (value) {
                  ended.add(value);
                  setState(() => frame = value);
                },
              );
            },
          ),
        ),
      ),
    );

    await tester.tap(find.byIcon(Icons.chevron_left));
    await tester.pump();
    expect(frame, 0);
    expect(ended.every((value) => value == 0), isTrue);
    ended.clear();

    await tester.tap(find.byIcon(Icons.chevron_right));
    await tester.pump();
    expect(frame, 1);
    await tester.tap(find.byIcon(Icons.chevron_right));
    await tester.pump();
    expect(frame, 2);
    expect(ended, [1, 2]);

    await tester.tap(find.byIcon(Icons.chevron_right));
    await tester.pump();
    expect(frame, 2);
    expect(ended.last, 2);
    expect(ended.every((value) => value >= 0 && value <= 2), isTrue);
    expect(changed, isEmpty);

    update(() => frame = 1);
    await tester.pump();
    await tester.tap(find.byIcon(Icons.chevron_left));
    await tester.pump();
    expect(frame, 0);
    expect(ended.last, 0);
    expect(tester.takeException(), isNull);
  });

  testWidgets('single or missing frames and invalid fps remain safe', (
    tester,
  ) async {
    for (final frameCount in [-1, 0, 1]) {
      for (final fps in [0.0, -30.0]) {
        final ended = <int>[];
        await _pumpPreview(
          tester,
          fps: fps,
          frameCount: frameCount,
          frame: 50,
          onFrameChangeEnd: ended.add,
        );
        await tester.tap(find.byIcon(Icons.chevron_left));
        await tester.tap(find.byIcon(Icons.chevron_right));
        await tester.pump();
        expect(ended.every((value) => value == 0), isTrue);
        expect(find.text('0:00.0 / 0:00'), findsOneWidget);
        expect(find.text(frameCount <= 0 ? '${S.frameLabel} 0 / 0' : '${S.frameLabel} 1 / 1'), findsOneWidget);
        expect(tester.takeException(), isNull);
      }
    }
  });

  testWidgets('drawing maps the full drag from letterbox to video pixels', (
    tester,
  ) async {
    final png = (await tester.runAsync(() => _png(width: 100, height: 200)))!;
    final boxes = <Rect>[];
    await _pumpPreview(
      tester,
      image: png,
      videoWidth: 100,
      videoHeight: 200,
      drawMode: true,
      onBoxDrawn: boxes.add,
    );
    await tester.pump();
    expect(find.text(S.drawBoxHint), findsOneWidget);

    final area = tester.getRect(find.byKey(_imageKey));
    final fitted = applyBoxFit(BoxFit.contain, const Size(100, 200), area.size);
    final displayed = Alignment.center.inscribe(fitted.destination, area);
    final start = Offset(
      displayed.left + displayed.width * 0.2,
      displayed.top + displayed.height * 0.2,
    );
    final end = Offset(
      displayed.left + displayed.width * 0.8,
      displayed.top + displayed.height * 0.8,
    );
    final gesture = await tester.startGesture(start);
    await gesture.moveBy(const Offset(20, 20));
    await tester.pump();
    await gesture.moveTo(end);
    await tester.pump();
    expect(boxes, isEmpty);
    await gesture.up();
    await tester.pump();

    expect(boxes, hasLength(1));
    _expectRectClose(boxes.single, const Rect.fromLTRB(20, 40, 80, 160));
    expect(boxes.single.left, greaterThanOrEqualTo(0));
    expect(boxes.single.top, greaterThanOrEqualTo(0));
    expect(boxes.single.right, lessThanOrEqualTo(100));
    expect(boxes.single.bottom, lessThanOrEqualTo(200));
    expect(tester.takeException(), isNull);
  });

  testWidgets('ignores small rectangles and drags outside draw mode', (
    tester,
  ) async {
    final png = (await tester.runAsync(_png))!;
    final boxes = <Rect>[];
    await _pumpPreview(
      tester,
      image: png,
      videoWidth: 100,
      videoHeight: 100,
      drawMode: true,
      onBoxDrawn: boxes.add,
    );
    final center = tester.getCenter(find.byKey(_imageKey));
    await tester.dragFrom(center, const Offset(40, 4));
    await tester.pump();
    expect(boxes, isEmpty);

    await _pumpPreview(tester, image: png, onBoxDrawn: boxes.add);
    await tester.drag(find.byKey(_imageKey), const Offset(100, 100));
    await tester.pump();
    expect(boxes, isEmpty);
    expect(find.text(S.drawBoxHint), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
