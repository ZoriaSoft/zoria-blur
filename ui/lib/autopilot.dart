// Self-test mode for build machines (no mouse, no desktop capture): drives the real app through
// pick video -> analyze -> select persons -> preview -> export and saves a PNG of the window at each
// step, rendered by the app itself. Off unless ZORIA_AUTOPILOT_VIDEO is set.
//
//   ZORIA_AUTOPILOT_VIDEO   video to open
//   ZORIA_AUTOPILOT_OUT     folder for shots + autopilot.json (default %TEMP%\zoria_autopilot)
//   ZORIA_AUTOPILOT_SELECT  person ids to select, e.g. 1,3,4 (default: the two seen longest)
//   ZORIA_AUTOPILOT_MODE    selected | except (default selected)
//   ZORIA_AUTOPILOT_EXPORT  output mp4 (default <out>\export.mp4)
//   ZORIA_AUTOPILOT_DET     fast | balanced | thorough
//   ZORIA_AUTOPILOT_THEME   light | dark (main.dart; ZORIA_AUTOPILOT_SIZE=WxH sets the window, runner)
//   After export: back to the start page (shot 09), Dosya menu open (09b), the project reopened from
//   "Son projeler" with its chosen people restored (10, step recent_restored).
//   ZORIA_AUTOPILOT_RERUN   1 = at the end also run Yardım -> Sistem kontrolünü tekrar çalıştır (shot w4_rerun)
//   First run (no setup_done in the settings folder): the setup is driven first (shots w1_welcome,
//   w2_checking, w2_system, w3_ready), then the normal flow. With a kept settings folder the saved device
//   choice is used and checked quietly (autopilot.json: start_path, quiet_check).
//   ZORIA_AUTOPILOT_PICKER_DIR  first open the real "Video seç" dialog in this folder (an outside UI
//                           Automation script lists what it shows and closes it)
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:path/path.dart' as p;

import 'app_state.dart';
import 'pickers.dart';
import 'screens/home.dart';

final shotKey = GlobalKey();

class Autopilot {
  Autopilot(this.st, this.env);
  final AppState st;
  final Map<String, String> env;
  final log = <String, Object?>{'steps': <Map<String, Object?>>[]};
  late final String out = env['ZORIA_AUTOPILOT_OUT'] ?? p.join(Directory.systemTemp.path, 'zoria_autopilot');

  static Autopilot? fromEnvironment(AppState st) {
    final env = Platform.environment;
    return env['ZORIA_AUTOPILOT_VIDEO'] == null ? null : Autopilot(st, env);
  }

  void _step(String name, [Map<String, Object?> extra = const {}]) {
    (log['steps'] as List).add({'step': name, 't': DateTime.now().toIso8601String(), ...extra});
    _save();
  }

  void _save() {
    try {
      File(p.join(out, 'autopilot.json')).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(log));
    } catch (_) {}
  }

  Future<void> shot(String name) async {
    await WidgetsBinding.instance.endOfFrame;
    await Future<void>.delayed(const Duration(milliseconds: 300));
    await WidgetsBinding.instance.endOfFrame;
    final boundary = shotKey.currentContext?.findRenderObject();
    if (boundary is! RenderRepaintBoundary) return;
    final img = await boundary.toImage(pixelRatio: 1.0);
    final bytes = await img.toByteData(format: ui.ImageByteFormat.png);
    if (bytes == null) return;
    File(p.join(out, '$name.png')).writeAsBytesSync(bytes.buffer.asUint8List());
    _step('shot', {'name': name});
  }

  Future<bool> _until(bool Function() cond, Duration timeout) async {
    final end = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(end)) {
      if (cond()) return true;
      await Future<void>.delayed(const Duration(milliseconds: 200));
    }
    return cond();
  }

  List<Map<String, Object?>> _rows() => [
        for (final r in st.check.rows) {'id': r.id, 'state': r.state.name, 'label': r.stateText, 'detail': r.detail}
      ];

  /// First-run setup: welcome -> system check (live) -> ready -> Bitir.
  Future<void> _wizard() async {
    _step('wizard_open');
    await shot('w1_welcome');
    st.wizardGo(1);
    await _until(() => st.check.doneCount >= 3 || st.check.finished, const Duration(minutes: 2));
    await Future<void>.delayed(const Duration(milliseconds: 700));
    await shot('w2_checking');
    final ok = await _until(() => st.check.finished, const Duration(minutes: 5));
    if (!ok) throw 'system check did not finish';
    // the list reveals one row every 380 ms
    await Future<void>.delayed(const Duration(milliseconds: 3200));
    await shot('w2_system');
    final res = st.check.result;
    _step('wizard_checked', {
      'rows': _rows(),
      'chosen': res?.device.chosen,
      'device_name': res?.device.deviceName,
      'engine': res?.cmd.label,
      'why': res?.why,
      'probe_failed': res?.probeFailed,
      'bench_ms': res?.device.benchMs,
      'bench_cpu_ms': res?.device.benchCpuMs,
    });
    st.wizardGo(2);
    await shot('w3_ready');
    st.finishWizard(besideVideo: true);
    await Future<void>.delayed(const Duration(milliseconds: 300));
    _step('wizard_done', {'setup_done': st.settings.setupDone, 'cache_saved': st.settings.deviceCache != null});
  }

  Future<void> run() async {
    Directory(out).createSync(recursive: true);
    final sw = Stopwatch()..start();
    try {
      final wizard = st.wizardOpen;
      if (wizard) await _wizard();
      await _until(() => !st.deviceChecking, const Duration(minutes: 4));
      log['start_path'] = wizard ? 'wizard' : (st.check.started ? 'full_check' : 'saved_choice');
      log['device'] = st.deviceText;
      log['backend'] = st.engine?.backend;
      log['engine'] = st.engine?.cmd.exe;
      final pickDir = env['ZORIA_AUTOPILOT_PICKER_DIR'];
      if (pickDir != null) {
        _step('picker_open', {'dir': pickDir});
        final picked = await pickVideoFile(initialDirectory: pickDir);
        _step('picker_closed', {'picked': picked});
      }
      await shot('01_start');

      await st.openVideo(env['ZORIA_AUTOPILOT_VIDEO']!);
      if (st.stage != Stage.ready) throw 'open failed: ${st.error} ${st.errorDetail}';
      if (env['ZORIA_AUTOPILOT_DET'] != null) st.setDetMode(env['ZORIA_AUTOPILOT_DET']!);
      await shot('02_ready');

      final t0 = sw.elapsedMilliseconds;
      final analyzing = st.analyze();
      await _until(() => st.progress > 0.35 || st.stage != Stage.analyzing, const Duration(minutes: 30));
      if (st.stage == Stage.analyzing) await shot('03_analyzing');
      await analyzing;
      if (st.stage != Stage.editing) throw 'analyze failed: ${st.error} ${st.errorDetail}';
      _step('analyzed', {'seconds': (sw.elapsedMilliseconds - t0) / 1000, 'persons': st.persons.length});
      await _until(() => st.previewBytes != null && !st.previewBusy, const Duration(minutes: 2));
      await shot('04_persons');

      final want = (env['ZORIA_AUTOPILOT_SELECT'] ?? '')
          .split(',')
          .map((s) => int.tryParse(s.trim()))
          .whereType<int>()
          .toList();
      final ids = want.isNotEmpty ? want : st.persons.take(2).map((x) => x.id).toList();
      if (env['ZORIA_AUTOPILOT_MODE'] == 'except') st.setMode(BlurMode.except);
      for (final id in ids) {
        if (st.persons.any((x) => x.id == id)) st.toggle(id);
      }
      await Future<void>.delayed(const Duration(milliseconds: 400));
      await _until(() => !st.previewBusy, const Duration(minutes: 2));
      await shot('05_selected');

      // live preview: play, change the selection while playing, pause
      st.setFrame(0, commit: true);
      await _until(() => !st.previewBusy, const Duration(seconds: 30));
      final pf0 = st.frame;
      await st.play();
      final tp = DateTime.now();
      await _until(() => st.player.image != null && !st.livePlanPending && st.player.frame > pf0 + 30,
          const Duration(seconds: 40));
      await shot('05b_playing');
      final played = st.player.frame - pf0;
      final secs = DateTime.now().difference(tp).inMilliseconds / 1000;
      final other = st.persons.map((x) => x.id).where((id) => !ids.contains(id)).firstOrNull;
      if (other != null) st.toggle(other);
      final pendingRight = st.livePlanPending;
      await _until(() => !st.livePlanPending, const Duration(seconds: 30));
      await Future<void>.delayed(const Duration(milliseconds: 300));
      await shot('05c_playing_changed');
      _step('live', {
        'frames_advanced': played,
        'seconds': secs,
        'playing': st.player.playing,
        'toggled': other,
        'pending_right_after_toggle': pendingRight,
        'regions_now': st.liveRegionsAt(st.player.frame)?.length,
      });
      if (other != null) st.toggle(other);
      await st.pause();
      await _until(() => !st.livePlanPending && !st.previewBusy, const Duration(seconds: 30));

      final risks = st.visibleRisks;
      if (risks.isNotEmpty) {
        st.setFrame((risks.first.startS * st.video!.fps).round(), commit: true);
        await Future<void>.delayed(const Duration(milliseconds: 400));
        await _until(() => !st.previewBusy, const Duration(minutes: 2));
        await shot('06_risky_moment');
      }
      _step('selected', {'ids': ids, 'risks_shown': risks.length});

      final outMp4 = env['ZORIA_AUTOPILOT_EXPORT'] ?? p.join(out, 'export.mp4');
      final t1 = sw.elapsedMilliseconds;
      final exporting = st.export(outMp4);
      await _until(
          () => (st.phaseKey == 'render' && st.progress > 0.3) || st.stage == Stage.done || st.error != null,
          const Duration(minutes: 30));
      if (st.stage == Stage.exporting) await shot('07_exporting');
      await exporting;
      if (st.stage != Stage.done) throw 'export failed: ${st.error} ${st.errorDetail}';
      _step('exported', {
        'seconds': (sw.elapsedMilliseconds - t1) / 1000,
        'output': st.outputPath,
        'qa_summary': st.qaSummary,
        'timeline_moments': st.riskMoments.length,
        'qa_extra': st.qaExtra.length,
        'leak_ranges': st.qa?['leak_ranges'],
        'leak_moments': st.leakMoments.length,
      });
      await shot('08_done');
      // zero leaks first: a self-test whose export still shows a sharp face fails
      if (st.leakMoments.isNotEmpty) throw 'export QA found ${st.leakMoments.length} leak moment(s)';
      // back to an empty window: the welcome page now lists this video under "Son projeler"
      st.backToEdit();
      await Future<void>.delayed(const Duration(milliseconds: 300));
      st.reset();
      await _until(() => st.settings.recents.isNotEmpty && st.settings.recents.first.thumb != null,
          const Duration(seconds: 20));
      _step('recent', {'recents': st.settings.recents.length, 'thumb': st.settings.recents.firstOrNull?.thumb});
      await Future<void>.delayed(const Duration(milliseconds: 500));
      await shot('09_start_recent');
      // Dosya menu: one "Video aç…" entry (0.1.8 merged "Yeni video" into it)
      fileMenuController.open();
      await Future<void>.delayed(const Duration(milliseconds: 400));
      await shot('09b_file_menu');
      fileMenuController.close();
      // back into the project from "Son projeler": the chosen people must come back
      final want2 = (ids.where((id) => st.persons.any((x) => x.id == id)).toList()..sort());
      await st.openRecent(st.settings.recents.first);
      await _until(() => st.stage == Stage.editing && st.previewBytes != null && !st.previewBusy,
          const Duration(minutes: 1));
      final got = st.selected.toList()..sort();
      final restored = st.stage == Stage.editing && got.join(',') == want2.join(',');
      _step('recent_restored', {'stage': st.stage.name, 'selected': got, 'expected': want2, 'ok': restored});
      await shot('10_recent_restored');
      if (!restored) throw 'selection not restored from Son projeler: $got != $want2';
      if (env['ZORIA_AUTOPILOT_RERUN'] == '1') {
        st.rerunSystemCheck();
        await _until(() => st.check.finished, const Duration(minutes: 4));
        await Future<void>.delayed(const Duration(milliseconds: 3200));
        await shot('w4_rerun');
        _step('rerun', {'rows': _rows(), 'chosen': st.check.result?.device.chosen});
        st.finishWizard();
        await _until(() => !st.deviceChecking, const Duration(minutes: 1));
      }
      log['quiet_check'] = st.quietCheckResult;
      log['ok'] = true;
    } catch (e) {
      log['ok'] = false;
      log['error'] = '$e';
      await shot('99_error');
    }
    _save();
    await st.shutdown(); // no engine job may outlive the app (a preview writing to a closed pipe)
    exit(log['ok'] == true ? 0 : 1);
  }
}
