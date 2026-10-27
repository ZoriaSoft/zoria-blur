// The system check of the first-run setup (and of Yardım -> Sistem kontrolünü tekrar çalıştır): runs the
// engine's `devices` probe on every shipped engine with --json-progress and turns its live events into six plain
// rows (processor, memory, display adapters, NVIDIA CUDA, DirectML, test run) and one decision:
// NVIDIA CUDA -> DirectML -> CPU, the same rule as the start-up probe and the runtime fallback.
// Nothing here may block the app: every engine call has a timeout, any failure ends on the CPU with a clear note.
import 'dart:async';

import 'package:flutter/foundation.dart';

import 'engine.dart';
import 'models.dart';
import 'strings.dart';

enum CheckState { waiting, checking, found, missing, failed }

class CheckRow {
  CheckRow(this.id, this.title);
  final String id;
  final String title;
  CheckState state = CheckState.waiting;
  String detail = '';

  /// Status word when the default one for [state] would read oddly ("Çalışıyor", "Başarılı", "Bilinmiyor").
  String? label;

  bool get done => state != CheckState.waiting && state != CheckState.checking;

  String get stateText =>
      label ??
      switch (state) {
        CheckState.waiting => S.stWaiting,
        CheckState.checking => S.stChecking,
        CheckState.found => S.stFound,
        CheckState.missing => S.stMissing,
        CheckState.failed => S.stFailed,
      };

  void set(CheckState s, String d, {String? label}) {
    state = s;
    detail = d;
    this.label = label;
  }
}

/// Runs `devices` (args, live events, timeout) on one engine build; tests inject a fake.
typedef DevicesRunner = Future<Map<String, dynamic>> Function(
    EngineCommand cmd, List<String> args, EventCallback onEvent, Duration timeout);

Future<Map<String, dynamic>> _realRunner(EngineCommand cmd, List<String> args, EventCallback onEvent, Duration timeout) =>
    Engine(cmd).run(['devices', ...args], onEvent: onEvent, withBackend: false, timeout: timeout);

/// What the user can do to get a faster (or any) graphics card path, shown under the check with one button.
enum AdviceKind { nvidiaDriver, gpuDriver, nvidiaPack }

class Advice {
  const Advice(this.kind, this.text, this.url, this.button);
  final AdviceKind kind;
  final String text; // one or two plain sentences: what is wrong, what it gives
  final String url; // the official page the button opens
  final String button;
}

/// Official driver pages by vendor (the adapter name decides).
const nvidiaDriverUrl = 'https://www.nvidia.com/tr-tr/drivers/';
const amdDriverUrl = 'https://www.amd.com/en/support/download/drivers.html';
const intelDriverUrl = 'https://www.intel.com/content/www/us/en/support/detect.html';
const releasesUrl = 'https://github.com/ZoriaSoft/zoria-blur/releases/latest';

String driverUrlFor(String adapterName) {
  final n = adapterName.toLowerCase();
  if (n.contains('nvidia') || n.contains('geforce') || n.contains('quadro') || n.contains('rtx')) return nvidiaDriverUrl;
  if (n.contains('amd') || n.contains('radeon')) return amdDriverUrl;
  return intelDriverUrl;
}

class CheckResult {
  CheckResult(this.cmd, this.device, {required this.why, this.probeFailed = false});
  final EngineCommand cmd;
  final DeviceInfo device;
  final String why; // one plain sentence: why this device
  final bool probeFailed; // nothing answered: CPU as the safe choice
}

String fmtSec(double ms) {
  final s = ms / 1000;
  return (s < 0.1 ? s.toStringAsFixed(3) : s.toStringAsFixed(2)).replaceAll('.', ',');
}

String fmtTimes(double r) => (r >= 3 ? r.round().toString() : r.toStringAsFixed(1)).replaceAll('.', ',');

class SystemCheck extends ChangeNotifier {
  SystemCheck({DevicesRunner? runner, this.cudaTimeout = const Duration(seconds: 150),
      this.mainTimeout = const Duration(seconds: 90)})
      : _runner = runner ?? _realRunner {
    reset();
  }

  final DevicesRunner _runner;
  final Duration cudaTimeout, mainTimeout;

  late List<CheckRow> rows;
  bool running = false;
  bool started = false;
  CheckResult? result;
  bool _nvidia = false;
  bool _cudaFromCudaEngine = false;
  final List<String> notes = []; // fallbacks that happened during the test run
  String? nvidiaDriver; // NVIDIA's own driver number ('560.94'), when an NVIDIA card is in
  String? _cudaReason; // last reason the engine gave for the CUDA row
  List<String> _gpuNames = const [];

  /// One thing the user can do for a faster path (null when the best path already works).
  Advice? advice;

  CheckRow row(String id) => rows.firstWhere((r) => r.id == id);
  int get doneCount => rows.where((r) => r.done).length;
  bool get finished => result != null && !running;

  void reset() {
    rows = [
      CheckRow('cpu', S.rowCpu),
      CheckRow('memory', S.rowMemory),
      CheckRow('gpus', S.rowGpus),
      CheckRow('cuda', S.rowCuda),
      CheckRow('dml', S.rowDml),
      CheckRow('test', S.rowTest),
    ];
    running = false;
    started = false;
    result = null;
    _nvidia = false;
    _cudaFromCudaEngine = false;
    notes.clear();
    nvidiaDriver = null;
    _cudaReason = null;
    _gpuNames = const [];
    advice = null;
  }

  /// Full check over [engines] (EngineLocator order: optional CUDA build first, then the main build).
  Future<CheckResult> run(List<EngineCommand> engines) async {
    if (running) {
      while (running) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
      return result!;
    }
    reset();
    running = true;
    started = true;
    notifyListeners();
    final cudaCmd = engines.where((e) => e.label == 'packaged-cuda').firstOrNull;
    final mainCmd = engines.where((e) => e.label != 'packaged-cuda').firstOrNull;
    Map<String, dynamic>? cudaRes, mainRes;
    var anyAnswered = false;

    if (cudaCmd != null) {
      try {
        cudaRes = await _runner(cudaCmd, const ['--gpu-test-only'], (ev) => _onEvent(ev, cuda: true), cudaTimeout);
        anyAnswered = true;
      } catch (_) {
        _closeOpen(fromCuda: true);
      }
    }
    final cudaWorks = cudaRes != null && cudaRes['chosen'] == 'cuda';
    if (mainCmd != null) {
      try {
        // with CUDA already chosen the main build only reports DirectML (no second test run)
        mainRes = await _runner(mainCmd, cudaWorks ? const ['--no-test'] : const [], (ev) => _onEvent(ev, cuda: false),
            mainTimeout);
        anyAnswered = true;
      } catch (_) {
        _closeOpen(fromCuda: false);
      }
    }

    CheckResult res;
    if (cudaWorks) {
      res = CheckResult(cudaCmd!, DeviceInfo.fromJson(cudaRes), why: S.whyCuda);
      final d = row('dml');
      if (!d.done) d.set(CheckState.missing, S.dmlNotNeeded, label: S.stSkipped);
    } else if (mainRes != null && mainCmd != null) {
      final dev = DeviceInfo.fromJson(mainRes);
      final why = switch (dev.chosen) {
        'dml' => _nvidia ? S.whyDmlAfterCuda : S.whyDml,
        'cuda' => S.whyCuda,
        _ => S.whyCpu,
      };
      res = CheckResult(mainCmd, dev, why: why);
    } else {
      final cmd = mainCmd ?? cudaCmd ?? engines.last;
      res = CheckResult(cmd, DeviceInfo(chosen: 'cpu'), why: anyAnswered ? S.whyCpu : S.whyCpuFailed,
          probeFailed: !anyAnswered);
    }
    // anything still open (engine crashed / timed out) gets a plain end state
    for (final r in rows) {
      if (r.done) continue;
      if (r.id == 'test') {
        r.set(CheckState.found, S.testCpuOnly, label: S.stPassed);
      } else {
        r.set(CheckState.missing, S.noAnswer, label: S.stUnknown);
      }
    }
    advice = _adviceFor(res);
    result = res;
    running = false;
    notifyListeners();
    return res;
  }

  /// The one step that would make this PC faster, in the order that matters: an NVIDIA card held back by its driver,
  /// an NVIDIA card this install has no NVIDIA engine for, a card DirectML could not use.
  Advice? _adviceFor(CheckResult res) {
    if (res.device.chosen == 'cuda') return null;
    final nvName = _gpuNames.where((n) => n.toLowerCase().contains('nvidia')).firstOrNull;
    if (_nvidia && _cudaReason == 'old_driver') {
      return Advice(AdviceKind.nvidiaDriver, S.adviceOldNvidia(nvidiaDriver ?? '?'), nvidiaDriverUrl, S.adviceDriverButton);
    }
    if (_nvidia && row('cuda').state == CheckState.failed) {
      return Advice(AdviceKind.nvidiaDriver, S.adviceCudaFailed, nvidiaDriverUrl, S.adviceDriverButton);
    }
    if (_nvidia && (_cudaReason == 'not_installed' || row('cuda').detail == S.cudaNotShipped)) {
      return const Advice(AdviceKind.nvidiaPack, S.adviceNvidiaPack, releasesUrl, S.adviceNvidiaPackButton);
    }
    final dml = row('dml');
    if (res.device.chosen == 'cpu' && dml.state == CheckState.failed && _gpuNames.isNotEmpty) {
      final name = _gpuNames.firstWhere((n) => !n.toLowerCase().contains('nvidia'), orElse: () => nvName ?? _gpuNames.first);
      return Advice(AdviceKind.gpuDriver, S.adviceDmlFailed, driverUrlFor(name), S.adviceDriverButton);
    }
    return null;
  }

  void _closeOpen({required bool fromCuda}) {
    for (final r in rows) {
      if (r.state == CheckState.checking) {
        if (fromCuda && r.id != 'cuda') {
          r.state = CheckState.waiting; // the main build asks again
        } else if (r.id != 'test') {
          r.set(r.id == 'cuda' || r.id == 'dml' ? CheckState.failed : CheckState.missing, S.noAnswer,
              label: r.id == 'cuda' || r.id == 'dml' ? null : S.stUnknown);
        } else {
          r.state = CheckState.waiting;
        }
      }
    }
    notifyListeners();
  }

  /// One `{"event": "probe", "item": ..., "status": ...}` from an engine.
  @visibleForTesting
  void onEventForTest(Map<String, dynamic> ev, {required bool cuda}) => _onEvent(ev, cuda: cuda);

  void _onEvent(Map<String, dynamic> ev, {required bool cuda}) {
    if (ev['event'] != 'probe') return;
    final item = ev['item'] as String? ?? '';
    final status = ev['status'] as String? ?? '';
    switch (item) {
      case 'cpu':
        _info(row('cpu'), status, () {
          final name = ev['name'] as String?;
          final cores = (ev['cores'] as num?)?.toInt() ?? 0, threads = (ev['threads'] as num?)?.toInt() ?? 0;
          final c = cores > 0 || threads > 0 ? S.cpuCores(cores, threads) : '';
          final n = name == null ? S.cpuNoName : DeviceInfo.cleanName(name);
          row('cpu').set(CheckState.found, c.isEmpty ? n : '$n · $c');
        });
      case 'memory':
        _info(row('memory'), status, () {
          final gb = (ev['gb'] as num?)?.toDouble();
          if (gb == null) {
            row('memory').set(CheckState.missing, S.memoryUnknown, label: S.stUnknown);
          } else {
            final t = gb >= 10 ? gb.round().toString() : gb.toStringAsFixed(1).replaceAll('.', ',');
            row('memory').set(CheckState.found, S.memoryGb(t));
          }
        });
      case 'gpus':
        _info(row('gpus'), status, () {
          final names = [for (final n in (ev['names'] as List? ?? const [])) DeviceInfo.cleanName('$n')];
          _gpuNames = names;
          _nvidia = names.any((n) => n.toLowerCase().contains('nvidia'));
          nvidiaDriver ??= ev['nvidia_driver'] as String?;
          if (names.isEmpty) {
            row('gpus').set(CheckState.missing, S.gpusNone);
          } else {
            row('gpus').set(CheckState.found, names.join(', '));
          }
        });
      case 'cuda':
        final r = row('cuda');
        // the CUDA build owns this row; the main build (no CUDA inside) only fills it when there is no CUDA build
        if (!cuda && (_cudaFromCudaEngine || r.done)) return;
        if (cuda) _cudaFromCudaEngine = true;
        if (status != 'checking') _cudaReason = ev['reason'] as String? ?? _cudaReason;
        nvidiaDriver ??= ev['driver'] as String?;
        _backend(r, status, ev, notShipped: _nvidia || ev['nvidia_present'] == true ? S.cudaNotShipped : S.cudaNoNvidia,
            failed: S.cudaFailed, notShippedIsMissing: true);
      case 'dml':
        if (cuda) return; // the CUDA build has no DirectML; the main build reports it
        _backend(row('dml'), status, ev, notShipped: S.dmlNotInstalled, failed: S.dmlFailed);
      case 'test':
        final r = row('test');
        final b = ev['backend'] as String? ?? 'cpu';
        if (cuda && b != 'cuda') return; // the CUDA build's CPU test: the main build decides instead
        if (r.done) return;
        if (status == 'checking') {
          r.set(CheckState.checking, b == 'cpu' ? S.testOnCpu : S.testOnGpu);
        } else if (status == 'ok') {
          final ms = (ev['ms'] as num?)?.toDouble();
          final cpuMs = (ev['cpu_ms'] as num?)?.toDouble();
          final gpu = b != 'cpu';
          var d = ms == null ? S.testCpuOnly : S.testOk(gpu ? S.whereGpu : S.whereCpu, fmtSec(ms));
          if (gpu && cpuMs != null) d = '$d ${S.testCompare(fmtSec(cpuMs))}';
          if (notes.isNotEmpty) d = '${notes.join(' ')} $d';
          r.set(CheckState.found, d, label: S.stPassed);
        } else if (status == 'failed') {
          if (b != 'cpu') notes.add(S.testFellBack);
          r.set(CheckState.checking, S.testFellBack);
          if (cuda) {
            // the CUDA build's test failed: CUDA is not usable after all
            row('cuda').set(CheckState.failed, S.cudaFailed);
            r.state = CheckState.waiting;
          }
        }
    }
    notifyListeners();
  }

  void _info(CheckRow r, String status, void Function() found) {
    if (r.done) return; // the second engine reports the same facts again
    if (status == 'checking') {
      r.set(CheckState.checking, '');
    } else {
      found();
    }
  }

  void _backend(CheckRow r, String status, Map<String, dynamic> ev,
      {required String notShipped, required String failed, bool notShippedIsMissing = true}) {
    final reason = ev['reason'] as String?;
    switch (status) {
      case 'checking':
        r.set(CheckState.checking, '');
      case 'ok':
        final dev = ev['device'] as String?;
        r.set(CheckState.found, dev == null ? S.stWorks : DeviceInfo.cleanName(dev), label: S.stWorks);
      case 'skipped':
        r.set(CheckState.missing, switch (reason) {
          'no_nvidia' => S.cudaNoNvidia,
          'software_only' => S.dmlSoftwareOnly,
          'disabled' => S.cudaDisabled,
          _ => notShipped,
        });
      case 'failed':
        r.set(
            CheckState.failed,
            switch (reason) {
              'no_model' => S.errModels,
              'old_driver' => S.cudaOldDriver('${ev['driver'] ?? '?'}', '${ev['min_driver'] ?? '527.41'}'),
              _ => failed,
            });
    }
  }
}
