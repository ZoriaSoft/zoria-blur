// Bridge to the Python engine: spawns `zoria-blur-engine.exe` (packaged) or `python -m zoria_blur` (dev)
// with --json-progress, streams JSON-lines events and returns the final {"ok": ...} object.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;

import 'strings.dart';

class EngineError implements Exception {
  EngineError(this.message, {this.detail = ''});
  final String message; // plain Turkish for the user
  final String detail; // engine error text (for the details line / log)
  @override
  String toString() => detail.isEmpty ? message : '$message ($detail)';
}

class EngineCancelled implements Exception {}

/// The engine did not answer within the time given to it (the process tree was killed).
class EngineTimeout implements Exception {}

/// How to start one engine build.
class EngineCommand {
  EngineCommand(this.exe, {this.prefix = const [], this.workingDir, this.env = const {}, required this.label});
  final String exe;
  final List<String> prefix;
  final String? workingDir;
  final Map<String, String> env;
  final String label; // 'packaged' / 'packaged-cuda' / 'dev'
}

class EngineLocator {
  /// Engines in the order the user's rule tries them: the optional CUDA build first (only useful on
  /// NVIDIA machines), then the main build (DirectML + CPU).
  static List<EngineCommand> locateAll() {
    final env = Platform.environment;
    final out = <EngineCommand>[];
    final appDir = File(Platform.resolvedExecutable).parent.path;
    final override = env['ZORIA_ENGINE'];
    if (override != null && File(override).existsSync()) {
      out.add(EngineCommand(override, label: 'override'));
      return out;
    }
    final cuda = p.join(appDir, 'engine-cuda', 'zoria-blur-engine.exe');
    if (File(cuda).existsSync()) out.add(EngineCommand(cuda, label: 'packaged-cuda'));
    final main = p.join(appDir, 'engine', 'zoria-blur-engine.exe');
    if (File(main).existsSync()) out.add(EngineCommand(main, label: 'packaged'));
    final dev = env['ZORIA_ENGINE_DEV'];
    if (out.isEmpty && dev != null && Directory(dev).existsSync()) {
      for (final venv in ['.venv-dml', '.venv']) {
        final py = p.join(dev, venv, 'Scripts', 'python.exe');
        if (File(py).existsSync()) {
          out.add(EngineCommand(py, prefix: ['-m', 'zoria_blur'], workingDir: dev, env: {'PYTHONPATH': dev}, label: 'dev'));
          break;
        }
      }
    }
    return out;
  }
}

typedef EventCallback = void Function(Map<String, dynamic> event);

/// One running engine call. [cancel] kills the whole process tree (engine + its ffmpeg).
class EngineJob {
  EngineJob._(this.result, this._kill);
  final Future<Map<String, dynamic>> result;
  final Future<void> Function() _kill;
  bool _cancelled = false;
  bool get cancelled => _cancelled;
  Future<void> cancel() async {
    _cancelled = true;
    await _kill();
  }
}

class Engine {
  Engine(this.cmd, {this.backend = 'cpu'});
  EngineCommand cmd;
  String backend; // cpu | dml | cuda (resolved; never 'auto' - the app probes at startup)

  static String get dataDir {
    final base = Platform.environment['LOCALAPPDATA'] ?? Directory.systemTemp.path;
    return p.join(base, 'ZoriaBlur');
  }

  /// Every engine process this app started and that has not exited yet. Killed when the app closes, so a
  /// job still running (a preview, a plan) never outlives the window and writes to a closed pipe.
  static final Set<Process> _live = {};

  static void _track(Process p) {
    _live.add(p);
    p.exitCode.then((_) => _live.remove(p), onError: (_) => _live.remove(p));
  }

  static Future<void> killAll() async {
    final ps = _live.toList();
    _live.clear();
    await Future.wait([
      for (final p in ps)
        if (Platform.isWindows)
          Process.run('taskkill', ['/T', '/F', '/PID', '${p.pid}']).then((_) {}, onError: (_) {})
        else
          Future(() => p.kill(ProcessSignal.sigkill)),
    ]);
  }

  EngineJob start(List<String> args, {EventCallback? onEvent, bool withBackend = true}) {
    final full = [
      ...cmd.prefix,
      if (withBackend) ...['--backend', backend],
      '--json-progress',
      ...args,
    ];
    late Process proc;
    var started = false;
    var cancelled = false;
    final completer = Completer<Map<String, dynamic>>();
    final errTail = StringBuffer();
    Map<String, dynamic>? finalObj;

    var exited = false;
    Future<void> kill() async {
      cancelled = true;
      if (!started || exited) return;
      if (Platform.isWindows) {
        await Process.run('taskkill', ['/T', '/F', '/PID', '${proc.pid}']);
      } else {
        proc.kill(ProcessSignal.sigkill);
      }
    }

    () async {
      try {
        proc = await Process.start(cmd.exe, full,
            workingDirectory: cmd.workingDir,
            environment: {...cmd.env, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
            runInShell: false);
        started = true;
        _track(proc);
        // the engine never reads stdin; close it so no child (powershell, ffprobe) waits on an open pipe
        unawaited(proc.stdin.close().catchError((_) {}));
        if (cancelled) await kill();
      } catch (e) {
        completer.completeError(EngineError(S.errEngineMissing, detail: '$e'));
        return;
      }
      final errDone = proc.stderr.transform(const Utf8Decoder(allowMalformed: true)).listen((s) {
        errTail.write(s);
        if (errTail.length > 8000) {
          final t = errTail.toString();
          errTail
            ..clear()
            ..write(t.substring(t.length - 4000));
        }
      }).asFuture<void>();
      await proc.stdout
          .transform(const Utf8Decoder(allowMalformed: true))
          .transform(const LineSplitter())
          .forEach((line) {
        final l = line.trim();
        if (!l.startsWith('{')) return;
        Map<String, dynamic> obj;
        try {
          obj = jsonDecode(l) as Map<String, dynamic>;
        } catch (_) {
          return;
        }
        if (obj.containsKey('ok')) {
          finalObj = obj;
        } else if (onEvent != null) {
          onEvent(obj);
        }
      });
      final code = await proc.exitCode;
      exited = true;
      await errDone.catchError((_) {});
      if (cancelled) {
        completer.completeError(EngineCancelled());
        return;
      }
      final f = finalObj;
      if (f != null && f['ok'] == true) {
        completer.complete(f);
        return;
      }
      final detail = (f?['error'] as String?) ?? _lastLine(errTail.toString()) ?? 'exit $code';
      _writeLog(full, errTail.toString(), detail);
      completer.completeError(EngineError(friendlyError(detail), detail: detail));
    }();
    return EngineJob._(completer.future, kill);
  }

  /// The engine with raw stdout (the `frames` stream of the live preview); no JSON parsing.
  Future<Process> startRaw(List<String> args) async {
    final p = await Process.start(cmd.exe, [...cmd.prefix, ...args],
        workingDirectory: cmd.workingDir,
        environment: {...cmd.env, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        runInShell: false);
    _track(p);
    return p;
  }

  Future<Map<String, dynamic>> run(List<String> args,
      {EventCallback? onEvent, bool withBackend = true, Duration? timeout}) {
    final job = start(args, onEvent: onEvent, withBackend: withBackend);
    if (timeout == null) return job.result;
    final c = Completer<Map<String, dynamic>>();
    final t = Timer(timeout, () {
      if (c.isCompleted) return;
      c.completeError(EngineTimeout());
      unawaited(job.cancel());
    });
    job.result.then((v) {
      t.cancel();
      if (!c.isCompleted) c.complete(v);
    }, onError: (Object e) {
      t.cancel();
      if (!c.isCompleted) c.completeError(e);
    });
    return c.future;
  }

  static String? _lastLine(String s) {
    final ls = s.trim().split('\n').where((l) => l.trim().isNotEmpty).toList();
    return ls.isEmpty ? null : ls.last.trim();
  }

  void _writeLog(List<String> args, String err, String detail) {
    try {
      final d = Directory(p.join(dataDir, 'logs'))..createSync(recursive: true);
      File(p.join(d.path, 'last_error.txt')).writeAsStringSync(
          '${DateTime.now()}\n${cmd.exe} ${args.join(' ')}\n$detail\n\n$err');
    } catch (_) {}
  }
}

/// Engine error text -> one plain Turkish sentence.
String friendlyError(String e) {
  final l = e.toLowerCase();
  if (l.contains('ffmpeg not found')) return S.errFfmpeg;
  if (l.contains('model not found')) return S.errModels;
  if (l.contains('memoryerror') || l.contains('out of memory') || l.contains('bad_alloc')) return S.errMemory;
  if (l.contains('permissionerror') || l.contains('access is denied') || l.contains('no space')) return S.errWrite;
  if (l.contains('encode failed')) return S.errEncode;
  if (l.contains('beyond end') || l.contains('out of range')) return S.errFrame;
  if (l.contains('blur-selected needs')) return S.errNeedPersons;
  if (l.contains('cannot decode') || l.contains('invalid data') || l.contains('stopiteration') ||
      l.contains('moov atom')) {
    return S.errOpenVideo;
  }
  if (l.contains('filenotfounderror')) return S.errNoFile;
  return S.errGeneric;
}
