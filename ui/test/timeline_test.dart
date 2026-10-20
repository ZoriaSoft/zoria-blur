import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zoria_blur/strings.dart';
import 'package:zoria_blur/theme.dart';
import 'package:zoria_blur/widgets/preview_pane.dart';
import 'package:zoria_blur/widgets/timeline.dart';

const _marks = [
  RiskMark(startS: 1, endS: 2, label: 'Yüzler üst üste'),
  RiskMark(startS: 4, endS: 4.05, label: 'Gizlenmemiş yüz', severe: true),
  RiskMark(startS: 9, endS: 10, label: 'Hızlı hareket'),
];

Future<void> _pump(
  WidgetTester tester, {
  double fps = 30,
  int frameCount = 301,
  int frame = 0,
  List<RiskMark> marks = _marks,
  List<TimelineLane> lanes = const [],
  ValueChanged<int>? onChanged,
  ValueChanged<int>? onEnd,
  Brightness brightness = Brightness.light,
}) async {
  tester.view.physicalSize = const Size(1000, 400);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: buildTheme(brightness),
    home: Scaffold(
      body: Align(
        alignment: Alignment.bottomCenter,
        child: Timeline(
          fps: fps,
          frameCount: frameCount,
          frame: frame,
          onFrameChanged: onChanged ?? (_) {},
          onFrameChangeEnd: onEnd ?? (_) {},
          marks: marks,
          lanes: lanes,
        ),
      ),
    ),
  ));
}

void main() {
  testWidgets('risk bands: count, tooltips, click jumps to the start frame', (tester) async {
    final ended = <int>[];
    await _pump(tester, onEnd: ended.add);
    expect(find.text(S.riskyMoments), findsOneWidget);
    expect(find.byKey(const ValueKey('timeline-risk-count')), findsOneWidget);
    expect(find.text('3'), findsOneWidget);
    for (var i = 0; i < _marks.length; i++) {
      final m = find.byKey(ValueKey('preview-risk-mark-$i'));
      expect(m, findsOneWidget);
      expect(tester.getSize(m).width, greaterThanOrEqualTo(4));
      await tester.tap(m);
      await tester.pump();
    }
    expect(ended, [30, 120, 270]);
    for (final mark in _marks) {
      final tip = find.byWidgetPredicate((w) => w is Tooltip && (w.message?.contains(mark.label) ?? false));
      expect(tip, findsOneWidget);
      final msg = tester.widget<Tooltip>(tip).message!;
      expect(msg, contains(formatTime(mark.startS)));
      expect(msg, contains(formatTime(mark.endS)));
    }
    expect(tester.takeException(), isNull);
  });

  testWidgets('no risks: plain label', (tester) async {
    await _pump(tester, marks: const []);
    expect(find.text(S.noRiskyMoments), findsOneWidget);
    expect(find.byKey(const ValueKey('timeline-risk-count')), findsNothing);
  });

  testWidgets('mark start frames are rounded and clamped', (tester) async {
    final ended = <int>[];
    await _pump(
      tester,
      fps: 10,
      frameCount: 21,
      marks: const [
        RiskMark(startS: -1, endS: 0.1, label: 'Başlangıç'),
        RiskMark(startS: 1.26, endS: 1.5, label: 'Kesirli zaman'),
        RiskMark(startS: 1.99, endS: 8, label: 'Bitiş'),
      ],
      onEnd: ended.add,
    );
    for (var i = 0; i < 3; i++) {
      await tester.tap(find.byKey(ValueKey('preview-risk-mark-$i')), warnIfMissed: false);
      await tester.pump();
    }
    expect(ended, [0, 13, 20]);
  });

  testWidgets('dragging on the ruler scrubs continuously, release commits once', (tester) async {
    final changed = <int>[];
    final ended = <int>[];
    await _pump(tester, frameCount: 101, fps: 25, marks: const [], onChanged: changed.add, onEnd: ended.add);
    final r = tester.getRect(find.byKey(const ValueKey('preview-timeline')));
    final g = await tester.startGesture(Offset(r.left + 30, r.center.dy));
    await g.moveTo(Offset(r.center.dx, r.center.dy));
    await tester.pump();
    await g.moveTo(Offset(r.right - 40, r.center.dy));
    await tester.pump();
    expect(changed.length, greaterThanOrEqualTo(2));
    expect(changed.last, greaterThan(changed.first));
    expect(changed.every((v) => v >= 0 && v <= 100), isTrue);
    expect(ended, isEmpty);
    await g.up();
    await tester.pump();
    expect(ended, hasLength(1));
    expect(ended.single, closeTo(changed.last, 1));
  });

  testWidgets('click on the ruler jumps; ends clamp', (tester) async {
    final ended = <int>[];
    await _pump(tester, frameCount: 101, fps: 25, marks: const [], onEnd: ended.add);
    final r = tester.getRect(find.byKey(const ValueKey('preview-timeline')));
    await tester.tapAt(Offset(r.left + 1, r.center.dy));
    await tester.tapAt(Offset(r.right - 1, r.center.dy));
    await tester.tapAt(Offset(r.center.dx, r.center.dy));
    await tester.pump();
    expect(ended[0], 0);
    expect(ended[1], 100);
    expect(ended[2], closeTo(50, 1));
  });

  testWidgets('person lanes in both themes, scroll when many', (tester) async {
    final lanes = [
      for (var i = 1; i <= 7; i++)
        TimelineLane(
          id: i,
          label: 'Kişi $i uzun bir isim ile taşma kontrolü',
          segments: const [
            [0.5, 3.0],
            [6.0, 9.5],
          ],
          willBlur: i.isOdd,
          selected: i == 1,
        ),
    ];
    for (final b in Brightness.values) {
      await _pump(tester, lanes: lanes, brightness: b);
      expect(find.byKey(const ValueKey('timeline-lane-1')), findsOneWidget);
      expect(find.byKey(const ValueKey('timeline-lane-7')), findsNothing); // 4 visible, rest scroll
      expect(find.byIcon(Icons.blur_on), findsWidgets);
      expect(tester.takeException(), isNull);
    }
  });

  testWidgets('zero frames / bad fps are safe', (tester) async {
    for (final n in [-1, 0, 1]) {
      await _pump(tester, frameCount: n, fps: 0, onEnd: (_) {});
      await tester.tapAt(tester.getCenter(find.byKey(const ValueKey('preview-timeline'))));
      await tester.pump();
      expect(tester.takeException(), isNull);
    }
  });
}
