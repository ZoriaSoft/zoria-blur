import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zoria_blur/app_state.dart';
import 'package:zoria_blur/engine.dart';
import 'package:zoria_blur/models.dart';
import 'package:zoria_blur/screens/home.dart';
import 'package:zoria_blur/settings.dart';
import 'package:zoria_blur/strings.dart';
import 'package:zoria_blur/system_check.dart';
import 'package:zoria_blur/theme.dart';

final _main = EngineCommand(r'C:\app\engine\zoria-blur-engine.exe', label: 'packaged');
final _cuda = EngineCommand(r'C:\app\engine-cuda\zoria-blur-engine.exe', label: 'packaged-cuda');

Map<String, dynamic> _ev(String item, String status, [Map<String, dynamic> extra = const {}]) =>
    {'event': 'probe', 'item': item, 'status': status, ...extra};

List<Map<String, dynamic>> _sysEvents(List<String> gpus) => [
      _ev('cpu', 'checking'),
      _ev('cpu', 'found', {'name': 'Intel(R) Core(TM) Test CPU @ 3.50GHz', 'cores': 4, 'threads': 8}),
      _ev('memory', 'checking'),
      _ev('memory', 'found', {'gb': 31.9}),
      _ev('gpus', 'checking'),
      _ev('gpus', gpus.isEmpty ? 'missing' : 'found', {'names': gpus}),
    ];

Map<String, dynamic> _result(String chosen, {String? device, List<String> ok = const [], Map<String, dynamic>? bench}) => {
      'ok': true,
      'chosen': chosen,
      'device_name': device,
      'probes': [
        for (final b in ['cuda', 'dml', 'cpu']) {'backend': b, 'ok': ok.contains(b) || b == 'cpu'}
      ],
      'bench': bench,
      'system': {
        'fingerprint': 'cpu|gpu',
        'cpu': {'name': 'Intel(R) Core(TM) Test CPU @ 3.50GHz'}
      },
    };

/// AMD card, main engine only -> DirectML.
Future<Map<String, dynamic>> _radeon(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
  for (final e in _sysEvents(['Radeon (TM) Pro Graphics'])) {
    on(e);
  }
  on(_ev('cuda', 'checking'));
  on(_ev('cuda', 'skipped', {'reason': 'not_installed', 'nvidia_present': false}));
  on(_ev('dml', 'checking'));
  on(_ev('dml', 'ok', {'ms': 9.1, 'device': 'Radeon (TM) Pro Graphics'}));
  on(_ev('cpu_backend', 'checking'));
  on(_ev('cpu_backend', 'ok', {'ms': 40}));
  on(_ev('test', 'checking', {'backend': 'dml'}));
  final bench = {'backend': 'dml', 'model': 'det_10g', 'ms': 21.0, 'cpu_ms': 252.0};
  on(_ev('test', 'ok', bench));
  return _result('dml', device: 'Radeon (TM) Pro Graphics', ok: ['dml'], bench: bench);
}

void main() {
  group('system check (rows + choice)', () {
    test('AMD card, no CUDA build: DirectML, with a test run and a measured comparison', () async {
      final c = SystemCheck(runner: _radeon);
      final r = await c.run([_main]);
      expect(r.device.chosen, 'dml');
      expect(r.cmd, _main);
      expect(r.why, S.whyDml);
      expect(c.row('cpu').detail, contains('4 çekirdek, 8 iş parçacığı'));
      expect(c.row('cpu').detail, isNot(contains('(R)')));
      expect(c.row('memory').detail, '32 GB');
      expect(c.row('gpus').detail, 'Radeon Pro Graphics');
      expect(c.row('cuda').state, CheckState.missing);
      expect(c.row('cuda').detail, S.cudaNoNvidia);
      expect(c.row('dml').state, CheckState.found);
      expect(c.row('dml').stateText, S.stWorks);
      expect(c.row('test').state, CheckState.found);
      expect(c.row('test').detail, contains('0,021'));
      expect(c.row('test').detail, contains('0,25'));
      expect(r.device.speedUp, closeTo(12, 0.01));
      expect(fmtTimes(r.device.speedUp!), '12');
      expect(c.rows.every((x) => x.done), isTrue);
    });

    test('NVIDIA with the CUDA build: CUDA, main build asked without a second test run', () async {
      final calls = <String>[];
      Future<Map<String, dynamic>> run(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        calls.add('${cmd.label} ${args.join(' ')}'.trim());
        if (cmd.label == 'packaged-cuda') {
          for (final e in _sysEvents(['NVIDIA GeForce Test GPU', 'Intel(R) Graphics'])) {
            on(e);
          }
          on(_ev('cuda', 'checking'));
          on(_ev('cuda', 'ok', {'ms': 3.0, 'device': 'NVIDIA GeForce Test GPU'}));
          on(_ev('dml', 'skipped', {'reason': 'not_installed'}));
          on(_ev('test', 'checking', {'backend': 'cuda'}));
          on(_ev('test', 'ok', {'backend': 'cuda', 'ms': 6.0, 'cpu_ms': 300.0}));
          return _result('cuda', device: 'NVIDIA GeForce Test GPU', ok: ['cuda'],
              bench: {'backend': 'cuda', 'ms': 6.0, 'cpu_ms': 300.0});
        }
        for (final e in _sysEvents(['NVIDIA GeForce Test GPU'])) {
          on(e);
        }
        on(_ev('cuda', 'skipped', {'reason': 'not_installed', 'nvidia_present': true}));
        on(_ev('dml', 'checking'));
        on(_ev('dml', 'ok', {'device': 'NVIDIA GeForce Test GPU'}));
        return _result('dml', device: 'NVIDIA GeForce Test GPU', ok: ['dml']);
      }

      final c = SystemCheck(runner: run);
      final r = await c.run([_cuda, _main]);
      expect(r.device.chosen, 'cuda');
      expect(r.cmd, _cuda);
      expect(r.why, S.whyCuda);
      expect(calls, ['packaged-cuda --gpu-test-only', 'packaged --no-test']);
      expect(c.row('cuda').state, CheckState.found); // the main build's "not installed" does not overwrite it
      expect(c.row('dml').state, CheckState.found);
      expect(c.row('test').detail, contains('Ekran kartında'));
    });

    test('CUDA test run fails: falls to DirectML and says why', () async {
      Future<Map<String, dynamic>> run(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        for (final e in _sysEvents(['NVIDIA GeForce Test GPU'])) {
          on(e);
        }
        if (cmd.label == 'packaged-cuda') {
          on(_ev('cuda', 'checking'));
          on(_ev('cuda', 'ok', {'device': 'NVIDIA GeForce Test GPU'}));
          on(_ev('test', 'checking', {'backend': 'cuda'}));
          on(_ev('test', 'failed', {'backend': 'cuda', 'error': 'x'}));
          on(_ev('test', 'checking', {'backend': 'cpu'}));
          on(_ev('test', 'ok', {'backend': 'cpu', 'ms': 300.0, 'cpu_ms': 300.0}));
          return _result('cpu');
        }
        on(_ev('dml', 'checking'));
        on(_ev('dml', 'ok', {'device': 'NVIDIA GeForce Test GPU'}));
        on(_ev('test', 'checking', {'backend': 'dml'}));
        on(_ev('test', 'ok', {'backend': 'dml', 'ms': 15.0, 'cpu_ms': 300.0}));
        return _result('dml', device: 'NVIDIA GeForce Test GPU', ok: ['dml']);
      }

      final c = SystemCheck(runner: run);
      final r = await c.run([_cuda, _main]);
      expect(r.device.chosen, 'dml');
      expect(r.why, S.whyDmlAfterCuda);
      expect(c.row('cuda').state, CheckState.failed);
      expect(c.row('test').state, CheckState.found);
      expect(c.row('test').detail, startsWith(S.testFellBack));
    });

    test('no GPU: CPU, plain reason', () async {
      Future<Map<String, dynamic>> run(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        for (final e in _sysEvents(const [])) {
          on(e);
        }
        on(_ev('cuda', 'skipped', {'reason': 'not_installed', 'nvidia_present': false}));
        on(_ev('dml', 'checking'));
        on(_ev('dml', 'skipped', {'reason': 'software_only'}));
        on(_ev('test', 'checking', {'backend': 'cpu'}));
        on(_ev('test', 'ok', {'backend': 'cpu', 'ms': 250.0, 'cpu_ms': 250.0}));
        return _result('cpu', bench: {'backend': 'cpu', 'ms': 250.0, 'cpu_ms': 250.0});
      }

      final c = SystemCheck(runner: run);
      final r = await c.run([_main]);
      expect(r.device.chosen, 'cpu');
      expect(r.why, S.whyCpu);
      expect(r.device.speedUp, isNull);
      expect(c.row('gpus').state, CheckState.missing);
      expect(c.row('dml').detail, S.dmlSoftwareOnly);
      expect(c.row('test').detail, contains('İşlemcide'));
    });

    test('engines time out / crash: never blocks, ends on the CPU with a clear note', () async {
      Future<Map<String, dynamic>> run(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        on(_ev('cpu', 'checking'));
        throw EngineTimeout();
      }

      final c = SystemCheck(runner: run);
      final r = await c.run([_cuda, _main]);
      expect(r.device.chosen, 'cpu');
      expect(r.probeFailed, isTrue);
      expect(r.why, S.whyCpuFailed);
      expect(c.rows.every((x) => x.done), isTrue);
      expect(c.row('cpu').detail, S.noAnswer);
    });

    test('DeviceInfo survives the settings file', () {
      final d = DeviceInfo.fromJson(_result('dml',
          device: 'Radeon (TM) Pro Graphics', ok: ['dml'], bench: {'backend': 'dml', 'ms': 20.0, 'cpu_ms': 200.0}));
      final back = DeviceInfo.fromJson(d.toJson());
      expect(back.chosen, 'dml');
      expect(back.deviceName, d.deviceName);
      expect(back.gpuBackends, ['dml']);
      expect(back.fingerprint, 'cpu|gpu');
      expect(back.speedUp, closeTo(10, 0.01));
      expect(back.cpuName, contains('Test CPU'));
    });
  });

  group('done panel and timeline count the same moments', () {
    test('QA-only risky moments are on the timeline too; the panel counts the timeline', () {
      final st = AppState()
        ..deviceChecking = false
        ..video = VideoInfo(path: r'C:\v\t.mp4', width: 640, height: 360, fps: 25, frameCount: 400, durationS: 16)
        ..projectDir = r'C:\p'
        ..stage = Stage.exporting;
      st.persons = [
        Person(id: 1, name: 'Person 1', trackIds: const [1], seconds: 9, lowConfidence: false, thumbnails: const []),
        Person(id: 2, name: 'Person 2', trackIds: const [2], seconds: 5, lowConfidence: false, thumbnails: const []),
      ];
      st.selected.add(1);
      st.risks = [
        Risk(type: 'occlusion', startS: 1.0, endS: 2.0, persons: const [1]),
        Risk(type: 'track_break', startS: 2.2, endS: 2.6, persons: const [1]), // < 0.5 s later: same moment
        Risk(type: 'track_break', startS: 6.0, endS: 6.5, persons: const [2]), // kept person: not shown
        Risk(type: 'face_merge', startS: 9.0, endS: 9.4, persons: const []),
      ];
      Map<String, dynamic> r(String type, double a, double b, List<int> p, bool rel) =>
          {'type': type, 'start_s': a, 'end_s': b, 'persons': p, 'relevant': rel};
      st.takeExportResult({
        'output': r'C:\v\t_bulanik.mp4',
        'qa': {
          'leak_ranges': [],
          'risky': [
            r('occlusion', 1.0, 2.0, [1], true),
            r('track_break', 2.2, 2.6, [1], true),
            r('track_break', 6.0, 6.5, [2], false),
            r('face_merge', 9.0, 9.4, [], true),
            r('unknown_face', 12.0, 12.4, [], true), // only the QA knows this one (0.1.6: panel 3, timeline 2)
          ],
        },
      }, r'C:\v\t_bulanik.mp4');
      st.stage = Stage.done;
      expect(st.riskMoments.length, 3);
      expect(st.qaSummary, S.qaRiskOnly(3));
    });

    test('leaks: the panel counts the red marks of the timeline (two ranges 0.3 s apart are one)', () {
      final st = AppState()
        ..deviceChecking = false
        ..video = VideoInfo(path: r'C:\v\t.mp4', width: 640, height: 360, fps: 25, frameCount: 400, durationS: 16)
        ..projectDir = r'C:\p'
        ..stage = Stage.exporting;
      st.persons = [
        Person(id: 1, name: 'Person 1', trackIds: const [1], seconds: 9, lowConfidence: false, thumbnails: const []),
      ];
      st.selected.add(1);
      Map<String, dynamic> l(double a, double b) => {'type': 'leak', 'start_s': a, 'end_s': b, 'persons': <int>[]};
      st.takeExportResult({
        'output': r'C:\v\t_bulanik.mp4',
        'qa': {
          'leak_ranges': [l(3.0, 3.4), l(3.7, 4.0), l(10.0, 10.2)],
          'risky': [],
        },
      }, r'C:\v\t_bulanik.mp4');
      st.stage = Stage.done;
      expect(st.riskMoments.where((m) => m.severe).length, 2);
      expect(st.leakMoments.length, 2);
      expect(st.qaHasIssues, isTrue);
      expect(st.qaSummary, S.qaLeaks(2));
    });

    test('a long yellow stretch never swallows a second leak (each leak keeps its own mark and button)', () {
      final st = AppState()
        ..deviceChecking = false
        ..video = VideoInfo(path: r'C:\v\t.mp4', width: 640, height: 360, fps: 25, frameCount: 750, durationS: 30)
        ..projectDir = r'C:\p'
        ..stage = Stage.exporting;
      st.persons = [
        Person(id: 1, name: 'Person 1', trackIds: const [1], seconds: 9, lowConfidence: false, thumbnails: const []),
      ];
      st.selected.add(1);
      st.risks = [
        for (var t = 2.0; t < 21; t += 0.8) Risk(type: 'low_confidence', startS: t, endS: t + 0.6, persons: const [1]),
      ];
      Map<String, dynamic> l(double a, double b) => {'type': 'leak', 'start_s': a, 'end_s': b, 'persons': <int>[]};
      st.takeExportResult({
        'output': r'C:\v\t_bulanik.mp4',
        'qa': {'leak_ranges': [l(3.0, 3.2), l(20.0, 20.3)], 'risky': []},
      }, r'C:\v\t_bulanik.mp4');
      st.stage = Stage.done;
      expect(st.leakMoments.map((m) => m.startS), [3.0, 20.0]);
      expect(st.riskMoments.where((m) => m.severe).length, 2);
      expect(st.riskMoments.last.severe, isTrue); // red marks are drawn on top
      expect(st.qaSummary, S.qaLeaks(2));
    });

    test('a different file at a recent path is not given the old project (same size, frames, pixels needed)', () {
      final a = VideoInfo(path: 'a', width: 1920, height: 1080, fps: 30, frameCount: 900, durationS: 30);
      expect(AppState.sameVideo(a, VideoInfo(path: 'a', width: 1920, height: 1080, fps: 30, frameCount: 900, durationS: 30)),
          isTrue);
      // ffprobe's estimate vs the decoded count (test.mp4: 968 vs 966) is the same video
      expect(AppState.sameVideo(a, VideoInfo(path: 'a', width: 1920, height: 1080, fps: 30, frameCount: 898, durationS: 30)),
          isTrue);
      expect(AppState.sameVideo(a, VideoInfo(path: 'a', width: 1920, height: 1080, fps: 30, frameCount: 960, durationS: 32)),
          isFalse);
      expect(AppState.sameVideo(a, VideoInfo(path: 'a', width: 1920, height: 1080, fps: 25, frameCount: 900, durationS: 36)),
          isFalse);
      expect(AppState.sameVideo(a, VideoInfo(path: 'a', width: 1280, height: 720, fps: 30, frameCount: 900, durationS: 30)),
          isFalse);
    });
  });

  group('first-run setup', () {
    Future<void> pump(WidgetTester tester, AppState st, Size size, Brightness b) async {
      tester.view.physicalSize = size;
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(MaterialApp(theme: buildTheme(b), home: HomeScreen(state: st)));
      await tester.pump();
    }

    AppState setupState() => AppState(settings: AppSettings(), check: SystemCheck(runner: _radeon))
      ..engines = [_main]
      ..wizardOpen = true;

    for (final b in Brightness.values) {
      for (final size in const [Size(1024, 640), Size(1360, 860)]) {
        testWidgets('three steps fit without overflow: ${size.width.toInt()}x${size.height.toInt()} ${b.name}',
            (tester) async {
          final st = setupState();
          await pump(tester, st, size, b);
          expect(find.text(S.welcomeTitle), findsOneWidget);
          expect(find.text(S.welcomePrivacy), findsOneWidget);
          expect(find.byKey(const ValueKey('setup-steps')), findsOneWidget);
          await tester.tap(find.byKey(const ValueKey('setup-next')));
          await tester.pump();
          expect(st.wizardStep, 1);
          expect(st.check.started, isTrue);
          // İleri waits until every row has been shown
          await tester.pump(const Duration(milliseconds: 100));
          expect(tester.widget<FilledButton>(find.byKey(const ValueKey('setup-next'))).onPressed, isNull);
          for (var i = 0; i < 10; i++) {
            await tester.pump(const Duration(milliseconds: 400));
          }
          expect(st.check.finished, isTrue);
          expect(find.byKey(const ValueKey('check-note')), findsOneWidget);
          expect(find.text(S.stWorks), findsOneWidget); // DirectML row
          expect(tester.widget<FilledButton>(find.byKey(const ValueKey('setup-next'))).onPressed, isNotNull);
          await tester.tap(find.byKey(const ValueKey('setup-next')));
          await tester.pump();
          expect(st.wizardStep, 2);
          expect(find.text('Radeon Pro Graphics · DirectML'), findsOneWidget);
          expect(find.text(S.speedHint('12')), findsOneWidget);
          expect(find.text(S.outBesideVideo), findsOneWidget);
          await tester.tap(find.byKey(const ValueKey('setup-finish')));
          await tester.pump();
          expect(st.wizardOpen, isFalse);
          expect(st.settings.setupDone, isTrue);
          expect(st.engine?.backend, 'dml');
          expect(st.deviceChecking, isFalse);
          expect(find.text(S.pickVideo), findsOneWidget); // the normal empty window
        });
      }
    }

    for (final b in Brightness.values) {
      testWidgets('editor at 1024x640 with a long GPU name in the top bar: no overflow (${b.name})', (tester) async {
        final st = AppState()
          ..deviceChecking = false
          ..device = DeviceInfo(chosen: 'dml', deviceName: 'Radeon (TM) Pro Graphics', gpuBackends: const ['dml'])
          ..video = VideoInfo(path: r'C:\v\uzun bir video adı.mp4', width: 640, height: 360, fps: 25, frameCount: 100,
              durationS: 4)
          ..projectDir = r'C:\p'
          ..stage = Stage.editing;
        await pump(tester, st, const Size(1024, 640), b);
        expect(find.byKey(const ValueKey('device-chip')), findsOneWidget);
        expect(tester.takeException(), isNull);
      });
    }

    testWidgets('skip: setup closes at once, the check still runs and is used', (tester) async {
      final gate = Completer<void>();
      Future<Map<String, dynamic>> slow(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        await gate.future;
        return _radeon(cmd, args, on, t);
      }

      final st = AppState(settings: AppSettings(), check: SystemCheck(runner: slow))
        ..engines = [_main]
        ..wizardOpen = true;
      await pump(tester, st, const Size(1280, 720), Brightness.light);
      await tester.tap(find.byKey(const ValueKey('setup-skip')));
      await tester.pump();
      expect(st.wizardOpen, isFalse);
      expect(st.settings.setupDone, isTrue);
      expect(st.deviceChecking, isTrue);
      gate.complete();
      await tester.pump(const Duration(milliseconds: 50));
      expect(st.deviceChecking, isFalse);
      expect(st.engine?.backend, 'dml');
      expect(st.settings.deviceCache?['chosen'], 'dml');
    });

    test('no engine found: the setup closes and the main window says so (0.1.8 stayed on a check that never ran)',
        () async {
      final st = AppState(settings: AppSettings(), check: SystemCheck(runner: _radeon))..wizardOpen = true;
      await st.init(); // no engine next to the test runner
      expect(st.wizardOpen, isFalse);
      expect(st.error, S.errEngineMissing);
      expect(st.settings.setupDone, isFalse); // a repaired install still gets the setup
    });

    test('a video dropped / picked during the check waits instead of failing; a non-video is refused', () async {
      final gate = Completer<void>();
      Future<Map<String, dynamic>> slow(EngineCommand cmd, List<String> args, EventCallback on, Duration t) async {
        await gate.future;
        return _radeon(cmd, args, on, t);
      }

      final st = AppState(settings: AppSettings(), check: SystemCheck(runner: slow))..engines = [_main];
      final checking = st.startSystemCheck();
      expect(st.deviceChecking, isTrue);
      await st.openPicked(r'C:\v\clip.mp4');
      expect(st.error, isNull); // 0.1.8: "engine missing" for a drop during the first check
      expect(st.stage, Stage.start);
      await st.openPicked(r'C:\v\photo.jpg');
      expect(st.error, S.notAVideo);
      gate.complete();
      await checking;
      expect(st.deviceChecking, isFalse);
    });

    testWidgets('Yardım menu re-runs the check on the Sistem step', (tester) async {
      final st = setupState()..wizardOpen = false;
      await st.startSystemCheck();
      await pump(tester, st, const Size(1360, 860), Brightness.dark);
      await tester.tap(find.text(S.menuHelp));
      await tester.pumpAndSettle();
      expect(find.text(S.rerunCheck), findsOneWidget);
      await tester.tap(find.text(S.rerunCheck));
      await tester.pump();
      expect(st.wizardOpen, isTrue);
      expect(st.wizardStep, 1);
      for (var i = 0; i < 10; i++) {
        await tester.pump(const Duration(milliseconds: 400));
      }
      expect(st.check.finished, isTrue);
      expect(find.byKey(const ValueKey('check-note')), findsOneWidget);
    });
  });
}
