import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:zoria_blur/app_state.dart';
import 'package:zoria_blur/live_player.dart';
import 'package:zoria_blur/models.dart';
import 'package:zoria_blur/pickers.dart';
import 'package:zoria_blur/screens/home.dart';
import 'package:zoria_blur/strings.dart';
import 'package:zoria_blur/settings.dart';
import 'package:zoria_blur/theme.dart';
import 'package:zoria_blur/version.dart';


VideoInfo _video() =>
    VideoInfo(path: r'C:\v\test.mp4', width: 640, height: 360, fps: 25, frameCount: 100, durationS: 4);

Future<void> _pump(WidgetTester tester, AppState st,
    {Size size = const Size(1360, 860),
    Brightness brightness = Brightness.light,
    Future<String?> Function()? pick}) async {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
      theme: buildTheme(brightness),
      home: pick == null ? HomeScreen(state: st) : HomeScreen(state: st, pickVideo: pick)));
  await tester.pump();
}

Person _person(int id, {String? name, int tracks = 1}) => Person(
    id: id,
    name: name ?? 'Person $id',
    trackIds: [for (var i = 0; i < tracks; i++) id * 10 + i],
    seconds: 12.5 - id,
    lowConfidence: false,
    thumbnails: const []);

AppState _editing({int persons = 8}) {
  final st = AppState()
    ..deviceChecking = false
    ..video = _video()
    ..projectDir = r'C:\p'
    ..stage = Stage.editing;
  st.persons = [for (var i = 1; i <= persons; i++) _person(i, name: i == 2 ? 'Çok uzun bir kişi adı, taşmamalı' : null)];
  st.risks = [
    Risk(type: 'occlusion', startS: 0.4, endS: 1.2, persons: const []),
    Risk(type: 'track_break', startS: 2.5, endS: 3.0, persons: const []),
  ];
  st.presence = {
    for (var i = 1; i <= persons; i++)
      i: [
        [0.2 * i, 1.5 + 0.2 * i],
        [2.4, 3.9],
      ]
  };
  return st;
}

int _filledButtons(WidgetTester tester) =>
    tester.widgetList(find.byWidgetPredicate((w) => w is FilledButton)).length;

void main() {
  group('video files only', () {
    test('picker filter lists video extensions only', () {
      final ext = videoTypeGroup.extensions!;
      for (final v in ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'wmv']) {
        expect(ext, contains(v));
      }
      for (final img in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'heic', 'tif']) {
        expect(ext, isNot(contains(img)));
      }
      expect(videoTypeGroup.label, S.videoFiles);
    });

    test('isVideoPath', () {
      expect(isVideoPath(r'C:\a\b.MP4'), isTrue);
      expect(isVideoPath('x.webm'), isTrue);
      expect(isVideoPath('photo.jpg'), isFalse);
      expect(isVideoPath('photo.PNG'), isFalse);
      expect(isVideoPath('noext'), isFalse);
    });

    test('openVideo refuses an image with the short Turkish message', () async {
      final st = AppState();
      await st.openVideo(r'C:\p\photo.jpg');
      expect(st.error, S.notAVideo);
      expect(st.stage, Stage.start);
    });
  });

  group('one clear action per step', () {
    testWidgets('start: only "Video seç" is primary', (tester) async {
      final st = AppState()..deviceChecking = false;
      await _pump(tester, st);
      expect(find.text(S.pickVideo), findsOneWidget);
      expect(_filledButtons(tester), 1);
    });

    testWidgets('video chosen: a "Başlat" button, nothing else primary', (tester) async {
      final st = AppState()
        ..deviceChecking = false
        ..video = _video()
        ..stage = Stage.ready;
      await _pump(tester, st);
      expect(find.text('Başlat'), findsOneWidget);
      expect(find.text(S.export), findsNothing);
      expect(_filledButtons(tester), 1);
      expect(find.text(S.step1), findsOneWidget);
    });

    testWidgets('while the PC is checked, Başlat says why it waits', (tester) async {
      final st = AppState()
        ..video = _video()
        ..stage = Stage.ready; // deviceChecking stays true
      await _pump(tester, st);
      expect(find.text(S.analyzeWait), findsOneWidget);
      final b = tester.widget<FilledButton>(find.byWidgetPredicate((w) => w is FilledButton));
      expect(b.onPressed, isNull);
    });

    testWidgets('people step: final button says what it does; disabled until someone is chosen', (tester) async {
      final st = AppState()
        ..deviceChecking = false
        ..video = _video()
        ..projectDir = r'C:\p'
        ..stage = Stage.editing;
      await _pump(tester, st);
      expect(find.text('Bulanıklaştır ve kaydet'), findsOneWidget); // inspector button (menu item is closed)
      expect(find.text('Dışa aktar'), findsNothing);
      expect(_filledButtons(tester), 1);
      expect(find.text(S.needSelection), findsOneWidget);
      expect(find.byKey(const ValueKey('preview-play')), findsOneWidget);
      final b = tester.widget<FilledButton>(find.byWidgetPredicate((w) => w is FilledButton));
      expect(b.onPressed, isNull);
      st.setMode(BlurMode.except);
      await tester.pump(const Duration(milliseconds: 300)); // preview debounce
      expect(find.text(S.selectHintExcept), findsOneWidget);
      final b2 = tester.widget<FilledButton>(find.byWidgetPredicate((w) => w is FilledButton));
      expect(b2.onPressed, isNotNull);
    });
  });

  group('live preview fail-safe', () {
    test('plan parse + neighbour frames', () {
      final plan = LivePlan.fromJson({
        'frames': {
          '10': [
            [100, 100, 30, 40]
          ],
          '11': [
            [104, 100, 30, 40]
          ],
        }
      });
      expect(plan.at(10).length, 2); // own + next frame
      expect(plan.at(12).length, 1); // previous frame's region still blurred
      expect(plan.at(50), isEmpty);
    });

    test('regions: nothing selected in "Seçilenleri" = no blur; otherwise no plan = blur everything', () {
      final st = AppState();
      expect(st.liveRegionsAt(5), isEmpty);
      st.selected.add(1);
      expect(st.liveRegionsAt(5), isNull); // no plan yet -> whole frame blurred
      expect(st.livePlanPending, isTrue);
      st.mode = BlurMode.except;
      st.selected.clear();
      expect(st.liveRegionsAt(5), isNull);
    });

    testWidgets('painter blurs inside the region and leaves the rest', (tester) async {
      // left half black, right half white; a region on the edge must mix them, a region-free spot must not
      final w = 64, h = 32;
      final px = Uint8List(w * h * 4);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final v = x < w / 2 ? 0 : 255;
          final i = (y * w + x) * 4;
          px[i] = px[i + 1] = px[i + 2] = v;
          px[i + 3] = 255;
        }
      }
      final img = await tester.runAsync(() async {
        final c = Completer<ui.Image>();
        ui.decodeImageFromPixels(px, w, h, ui.PixelFormat.rgba8888, c.complete);
        return c.future;
      });
      final rec = ui.PictureRecorder();
      LiveFramePainter(image: img!, regions: [
        [32, 16, 10, 10]
      ], videoW: w, videoH: h, strength: 2)
          .paint(Canvas(rec), Size(w.toDouble(), h.toDouble()));
      final out = await tester.runAsync(() async {
        final i = await rec.endRecording().toImage(w, h);
        return (await i.toByteData())!;
      });
      int at(int x, int y) => out!.getUint8((y * w + x) * 4);
      expect(at(31, 16), inInclusiveRange(30, 225)); // edge inside the region: blurred (mixed)
      expect(at(5, 16), 0); // outside: untouched
      expect(at(60, 16), 255);
    });
  });

  group('desktop shell', () {
    testWidgets('editor fits without overflow: 1360x860, 1280x720, 1024x640, light and dark', (tester) async {
      for (final size in const [Size(1360, 860), Size(1280, 720), Size(1024, 640)]) {
        for (final b in Brightness.values) {
          final st = _editing();
          st.selected.addAll([1, 3]);
          await _pump(tester, st, size: size, brightness: b);
          await tester.pump(const Duration(milliseconds: 300));
          expect(tester.takeException(), isNull, reason: '$size $b');
          expect(find.byKey(const ValueKey('preview-timeline')), findsOneWidget);
          expect(find.byKey(const ValueKey('timeline-lane-1')), findsOneWidget);
          expect(find.byKey(const ValueKey('export')), findsOneWidget);
          expect(find.byKey(const ValueKey('device-chip')), findsOneWidget);
          expect(find.text(S.menuFile), findsOneWidget);
          // the result screen at the same size, worst case: long file name and several leak moments
          st
            ..stage = Stage.done
            ..outputPath = 'C:\\Users\\kullanici\\Videolar\\cok uzun bir video dosyasi adi ile birlikte_bulanik.mp4'
            ..qa = {'leak_ranges': [], 'risky': []}
            ..leaks = [
              for (var i = 0; i < 6; i++)
                Risk.fromJson({'start_s': 1.0 + i * 2, 'end_s': 1.5 + i * 2}, severe: true)
            ];
          st.dismissError();
          await tester.pump(const Duration(milliseconds: 300));
          expect(tester.takeException(), isNull, reason: 'done $size $b');
          expect(find.byKey(const ValueKey('export-done')), findsOneWidget);
          expect(find.byKey(const ValueKey('done-fix')), findsOneWidget);
        }
      }
    });

    testWidgets('start / ready / analyzing fit in both themes at 1280x720', (tester) async {
      for (final b in Brightness.values) {
        final settings = AppSettings();
        for (var i = 0; i < 6; i++) {
          settings.recents.add(RecentProject(
              video: 'C:\\v\\video_$i uzun dosya adı ile birlikte.mp4',
              project: 'C:\\p\\$i',
              opened: DateTime.now().subtract(Duration(days: i)),
              durationS: 75,
              persons: 3));
        }
        final st = AppState(settings: settings)..deviceChecking = false;
        await _pump(tester, st, size: const Size(1280, 720), brightness: b);
        expect(find.byKey(const ValueKey('drop-zone')), findsOneWidget);
        expect(find.text(S.recent), findsWidgets);
        expect(tester.takeException(), isNull);
        st
          ..video = _video()
          ..stage = Stage.ready;
        st.dismissError();
        await tester.pump();
        expect(find.text(S.analyze), findsOneWidget);
        expect(tester.takeException(), isNull);
        st.stage = Stage.analyzing;
        st.progress = 0.42;
        st.phase = S.phaseFaces;
        st.dismissError();
        await tester.pump();
        expect(find.byKey(const ValueKey('analyze-progress')), findsOneWidget);
        expect(find.text('42%'), findsOneWidget);
        expect(tester.takeException(), isNull);
      }
    });

    testWidgets('export progress is a panel over the editor; the result gets its own screen', (tester) async {
      final st = _editing()..selected.add(1);
      st.stage = Stage.exporting;
      st.progress = 0.3;
      st.phase = S.phaseRender;
      await _pump(tester, st);
      expect(find.byKey(const ValueKey('export-progress')), findsOneWidget);
      expect(find.byKey(const ValueKey('preview-timeline')), findsOneWidget); // editor stays underneath
      expect(find.text(S.cancel), findsOneWidget);
      st
        ..stage = Stage.done
        ..outputPath = r'C:\v\test_bulanik.mp4'
        ..qa = {'leak_ranges': [], 'risky': []};
      st.dismissError();
      await tester.pump();
      expect(find.byKey(const ValueKey('export-done')), findsOneWidget);
      // the panel counts the timeline's moments (2 here), never a separate list
      expect(st.riskMoments.length, 2);
      expect(find.text(S.qaRiskOnly(2)), findsOneWidget);
      st.risks = [];
      st.dismissError();
      await tester.pump();
      expect(find.text(S.qaClean), findsOneWidget);
      // no modal any more: the sidebar's "Video aç…" stays usable, and the result has its own next steps
      expect(find.byKey(const ValueKey('export-progress')), findsNothing);
      expect(find.byKey(const ValueKey('done-new-video')), findsOneWidget);
      expect(find.text(S.newVideo), findsOneWidget);
      expect(find.text(S.openFolder), findsOneWidget);
      expect(find.byKey(const ValueKey('result-source')), findsOneWidget);
      final open = tester.widget<OutlinedButton>(find.byKey(const ValueKey('sidebar-open')));
      expect(open.onPressed, isNotNull);
      // a leak: the main action becomes the fix, "Yeni video" stays one click away
      st.leaks = [Risk.fromJson({'type': 'leak', 'start_s': 1.0, 'end_s': 1.4, 'start_frame': 25, 'end_frame': 35}, severe: true)];
      st.dismissError();
      await tester.pump();
      expect(find.byKey(const ValueKey('done-fix')), findsOneWidget);
      expect(find.byKey(const ValueKey('done-new-video')), findsOneWidget);
      st.leaks = [];
      st.dismissError();
      await tester.pump();
      await tester.tap(find.byKey(const ValueKey('done-back')));
      await tester.pump(const Duration(milliseconds: 300));
      expect(st.stage, Stage.editing);
      expect(find.byKey(const ValueKey('export-done')), findsNothing);
    });

    testWidgets('keyboard: arrows step frames, Home/End, Ctrl+B toggles the sidebar', (tester) async {
      final st = _editing();
      await _pump(tester, st);
      await tester.sendKeyEvent(LogicalKeyboardKey.arrowRight);
      await tester.sendKeyEvent(LogicalKeyboardKey.arrowRight);
      await tester.pump(const Duration(milliseconds: 300));
      expect(st.frame, 2);
      await tester.sendKeyEvent(LogicalKeyboardKey.arrowLeft);
      expect(st.frame, 1);
      await tester.sendKeyEvent(LogicalKeyboardKey.end);
      expect(st.frame, 99);
      await tester.sendKeyEvent(LogicalKeyboardKey.home);
      expect(st.frame, 0);
      final open = st.settings.sidebarOpen;
      await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
      await tester.sendKeyEvent(LogicalKeyboardKey.keyB);
      await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
      await tester.pump(const Duration(milliseconds: 300));
      expect(st.settings.sidebarOpen, !open);
      expect(tester.takeException(), isNull);
    });

    testWidgets('menus show shortcuts; Görünüm > Tema > Koyu switches the theme', (tester) async {
      final st = _editing();
      await _pump(tester, st);
      await tester.tap(find.text(S.menuFile));
      await tester.pumpAndSettle();
      expect(find.byKey(const ValueKey('menu-open-video')), findsOneWidget);
      expect(find.text('Yeni video'), findsNothing); // merged into "Video aç…" in 0.1.8
      expect(find.text('Ctrl+O'), findsOneWidget);
      expect(find.text('Ctrl+N'), findsNothing);
      expect(find.text('Ctrl+S'), findsWidgets);
      await tester.tap(find.text(S.menuView));
      await tester.pumpAndSettle();
      await tester.tap(find.text(S.theme));
      await tester.pumpAndSettle();
      await tester.tap(find.text(S.themeDark));
      await tester.pumpAndSettle();
      expect(st.settings.themeMode, ThemeMode.dark);
    });

    testWidgets('sidebar "Video aç…" opens the video picker; cancel changes nothing', (tester) async {
      final st = _editing()..selected.addAll([1, 3]);
      var picks = 0;
      await _pump(tester, st, pick: () async {
        picks++;
        return null; // the user pressed İptal
      });
      expect(find.text('Yeni video'), findsNothing);
      final btn = find.byKey(const ValueKey('sidebar-open'));
      expect(find.descendant(of: btn, matching: find.text(S.openVideoMenu)), findsOneWidget);
      expect(find.descendant(of: btn, matching: find.byIcon(Icons.folder_open_outlined)), findsOneWidget);
      await tester.tap(btn);
      await tester.pump(const Duration(milliseconds: 300));
      expect(picks, 1);
      expect(st.stage, Stage.editing);
      expect(st.video?.path, r'C:\v\test.mp4');
      expect(st.persons.length, 8);
      expect(st.selected, {1, 3});
      // Ctrl+O and the old Ctrl+N both open the same dialog
      for (final k in [LogicalKeyboardKey.keyO, LogicalKeyboardKey.keyN]) {
        await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
        await tester.sendKeyEvent(k);
        await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
        await tester.pump(const Duration(milliseconds: 300));
      }
      expect(picks, 3);
      expect(st.stage, Stage.editing);
      expect(st.selected, {1, 3});
      // collapsed strip: same action, same icon, tooltip says what it does
      st.settings.toggleSidebar();
      await tester.pump(const Duration(milliseconds: 300));
      final strip = find.byKey(const ValueKey('sidebar-open-strip'));
      expect(tester.widget<IconButton>(strip).tooltip, '${S.openVideoMenu}  (Ctrl+O)');
      await tester.tap(strip);
      await tester.pump(const Duration(milliseconds: 300));
      expect(picks, 4);
      expect(tester.takeException(), isNull);
    });

    testWidgets('Video aç… is disabled while a job runs', (tester) async {
      final st = _editing()..stage = Stage.analyzing;
      st.video = _video();
      var picks = 0;
      await _pump(tester, st, pick: () async {
        picks++;
        return null;
      });
      final b = tester.widget<OutlinedButton>(find.byKey(const ValueKey('sidebar-open')));
      expect(b.onPressed, isNull);
      await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
      await tester.sendKeyEvent(LogicalKeyboardKey.keyO);
      await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
      await tester.pump(const Duration(milliseconds: 300));
      expect(picks, 0);
    });

    testWidgets('a picked file that fails to open keeps the open project as it was', (tester) async {
      final st = _editing()..selected.add(2); // no engine here: opening fails
      await _pump(tester, st, pick: () async => r'C:\v\other.mp4');
      await tester.tap(find.byKey(const ValueKey('sidebar-open')));
      await tester.pump(const Duration(milliseconds: 300));
      expect(st.stage, Stage.editing);
      expect(st.video?.path, r'C:\v\test.mp4');
      expect(st.selected, {2});
      expect(st.persons.length, 8);
    });

    testWidgets('inspector: merge needs 2+, split needs one person with several parts', (tester) async {
      final st = _editing(persons: 3);
      st.persons = [_person(1, tracks: 3), _person(2), _person(3)];
      await _pump(tester, st);
      ButtonStyleButton btn(String k) => tester.widget<ButtonStyleButton>(find.byKey(ValueKey(k)));
      expect(find.byKey(const ValueKey('merge')), findsOneWidget);
      expect(btn('merge').onPressed, isNull);
      expect(btn('split').onPressed, isNull);
      st.toggle(1);
      await tester.pump(const Duration(milliseconds: 300));
      expect(btn('split').onPressed, isNotNull);
      st.toggle(2);
      await tester.pump(const Duration(milliseconds: 300));
      expect(btn('merge').onPressed, isNotNull);
      expect(btn('split').onPressed, isNull);
    });
  });

  group('settings', () {
    test('recents and theme persist', () {
      final dir = Directory.systemTemp.createTempSync('zb_settings');
      addTearDown(() => dir.deleteSync(recursive: true));
      final f = File('${dir.path}/settings.json');
      final a = AppSettings.load(f);
      a.setThemeMode(ThemeMode.dark);
      a.touchRecent(RecentProject(video: r'C:\v\a.mp4', project: r'C:\p\a', opened: DateTime(2026, 10, 15, 9), durationS: 16));
      a.touchRecent(RecentProject(video: r'C:\v\b.mp4', project: r'C:\p\b', opened: DateTime(2026, 10, 15, 10)));
      a.touchRecent(RecentProject(video: r'C:\V\A.MP4', project: r'C:\p\a', opened: DateTime(2026, 10, 15, 11), thumb: 't.png'));
      final b = AppSettings.load(f);
      expect(b.themeMode, ThemeMode.dark);
      expect(b.recents.map((r) => r.name), ['a.mp4', 'b.mp4']); // one entry per video, case-insensitive
      expect(b.recents.first.thumb, 't.png');
      expect(b.recents.first.durationS, 16); // kept from the first entry
    });

    test('chosen people are kept per project in "Son projeler" and come back', () {
      final dir = Directory.systemTemp.createTempSync('zb_settings');
      addTearDown(() => dir.deleteSync(recursive: true));
      final f = File('${dir.path}/settings.json');
      final s = AppSettings.load(f);
      s.touchRecent(RecentProject(video: r'C:\v\test.mp4', project: r'C:\p', opened: DateTime(2026, 10, 15, 9)));
      s.touchRecent(RecentProject(video: r'C:\v\b.mp4', project: r'C:\p\b', opened: DateTime(2026, 10, 15, 10)));
      final st = AppState(settings: s)
        ..deviceChecking = false
        ..video = _video()
        ..projectDir = r'C:\p'
        ..stage = Stage.editing;
      st.persons = [for (var i = 1; i <= 5; i++) _person(i)];
      st.toggle(4);
      st.toggle(2);
      st.setMode(BlurMode.except);
      expect(s.recents.map((r) => r.name), ['b.mp4', 'test.mp4']); // saving does not reorder the list
      // a later touch (thumbnail, reopen) keeps the saved selection
      s.touchRecent(RecentProject(video: r'C:\v\test.mp4', project: r'C:\p', opened: DateTime(2026, 10, 15, 11)));
      final again = AppSettings.load(f);
      final r = again.recentFor(r'C:\V\TEST.MP4')!;
      expect(r.selected, [2, 4]);
      expect(r.mode, 'except');
      // reopened: ids that no longer exist are dropped, the rest come back with the mode
      final st2 = AppState(settings: again)..video = _video();
      st2.persons = [_person(2), _person(3)];
      st2.restoreSelectionFor(r'C:\v\test.mp4');
      expect(st2.selected, {2});
      expect(st2.mode, BlurMode.except);
    });

    test('"Video aç…" on a video that is already in "Son projeler" reopens that project', () async {
      final dir = Directory.systemTemp.createTempSync('zb_recent');
      addTearDown(() => dir.deleteSync(recursive: true));
      final proj = Directory('${dir.path}/p')..createSync();
      File('${proj.path}/project.json').writeAsStringSync('{}');
      final s = AppSettings();
      final video = '${dir.path}/gone.mp4'; // the file was moved away: the recent path says so plainly
      s.touchRecent(RecentProject(video: video, project: proj.path, opened: DateTime(2026, 10, 15, 9)));
      final st = AppState(settings: s)..deviceChecking = false;
      await st.openPicked(video.toUpperCase());
      expect(st.error, S.errNoFile); // the recent-project path ran (a fresh open would say "engine missing")
      // a video that is not in the list opens fresh
      final st2 = AppState(settings: AppSettings())..deviceChecking = false;
      await st2.openPicked('${dir.path}/new.mp4');
      expect(st2.error, S.errEngineMissing);
    });

    test('a new project at the same path replaces the recent entry whole (old people never carry over)', () {
      final s = AppSettings();
      s.touchRecent(RecentProject(video: r'C:\v\GOPR0001.MP4', project: r'C:\p\a', opened: DateTime(2026, 10, 15, 9)));
      expect(s.saveSelection(r'C:\v\GOPR0001.MP4', [3, 1], 'except'), isTrue);
      s.touchRecent(RecentProject(video: r'C:\v\GOPR0001.MP4', project: r'C:\p\a', opened: DateTime(2026, 10, 15, 10)));
      expect(s.recents.single.selected, [1, 3]); // same project: kept
      s.touchRecent(RecentProject(video: r'C:\v\gopr0001.mp4', project: r'C:\p\b', opened: DateTime(2026, 10, 16, 9)));
      expect(s.recents.single.project, r'C:\p\b');
      expect(s.recents.single.selected, isNull);
      expect(s.recents.single.mode, isNull);
      // same project folder, but the file there changed: its saved people are forgotten
      s.saveSelection(r'C:\v\GOPR0001.MP4', [2], 'except');
      s.clearSelection(r'C:\V\GOPR0001.MP4');
      expect(s.recents.single.selected, isNull);
      expect(s.recents.single.mode, isNull);
      expect(s.recents.single.project, r'C:\p\b');
    });

    test('broken settings file = defaults', () {
      final dir = Directory.systemTemp.createTempSync('zb_settings');
      addTearDown(() => dir.deleteSync(recursive: true));
      final f = File('${dir.path}/settings.json')..writeAsStringSync('{not json');
      final s = AppSettings.load(f);
      expect(s.themeMode, ThemeMode.system);
      expect(s.recents, isEmpty);
    });

    test('version.dart matches pubspec', () {
      final v = RegExp(r'^version:\s*([\d.]+)', multiLine: true).firstMatch(File('pubspec.yaml').readAsStringSync())!;
      expect(appVersion, v.group(1));
    });

    test('dates in plain Turkish', () {
      final now = DateTime(2026, 10, 15, 12);
      expect(S.when(DateTime(2026, 10, 15, 9, 5), now), 'Bugün 09:05');
      expect(S.when(DateTime(2026, 10, 14, 23, 0), now), 'Dün 23:00');
      expect(S.when(DateTime(2026, 3, 2, 8), now), '2 Mar');
      expect(S.when(DateTime(2025, 3, 2, 8), now), '2 Mar 2025');
    });
  });
}
