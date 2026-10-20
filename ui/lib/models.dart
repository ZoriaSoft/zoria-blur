import 'package:path/path.dart' as p;

import 'strings.dart';

double _d(Object? v) => (v as num?)?.toDouble() ?? 0.0;
int _i(Object? v) => (v as num?)?.toInt() ?? 0;

class VideoInfo {
  VideoInfo({required this.path, required this.width, required this.height, required this.fps,
      required this.frameCount, required this.durationS});
  final String path;
  final int width, height, frameCount;
  final double fps, durationS;

  factory VideoInfo.fromJson(Map<String, dynamic> j, {String? path}) => VideoInfo(
        path: path ?? j['path'] as String,
        width: _i(j['width']),
        height: _i(j['height']),
        fps: _d(j['fps']),
        frameCount: _i(j['frame_count']),
        durationS: _d(j['duration_s']),
      );

  String get name => p.basename(path);
}

class Person {
  Person({required this.id, required this.name, required this.trackIds, required this.seconds,
      required this.lowConfidence, required this.thumbnails});
  final int id;
  final String name; // engine name ('Person 3' = never renamed)
  final List<int> trackIds;
  final double seconds;
  final bool lowConfidence;
  final List<String> thumbnails; // absolute paths

  factory Person.fromJson(Map<String, dynamic> j, String projectDir) => Person(
        id: _i(j['id']),
        name: (j['name'] as String?) ?? '',
        trackIds: [for (final t in (j['track_ids'] as List? ?? const [])) _i(t)],
        seconds: _d(j['seconds_visible']),
        lowConfidence: j['low_confidence'] == true,
        thumbnails: [for (final t in (j['thumbnails'] as List? ?? const [])) p.join(projectDir, t as String)],
      );

  String get displayName => RegExp(r'^Person \d+$').hasMatch(name) || name.isEmpty ? S.personDefault(id) : name;

  /// Frame of the best thumbnail (file names are NN_fFFFFFF.jpg) - a good first preview frame.
  int? get bestFrame {
    if (thumbnails.isEmpty) return null;
    final m = RegExp(r'_f(\d+)\.jpg$').firstMatch(thumbnails.first);
    return m == null ? null : int.parse(m.group(1)!);
  }
}

class Risk {
  Risk({required this.type, required this.startS, required this.endS, required this.persons, this.severe = false});
  final String type;
  final double startS, endS;
  final List<int> persons;
  final bool severe;

  factory Risk.fromJson(Map<String, dynamic> j, {bool severe = false}) => Risk(
        type: (j['type'] as String?) ?? (severe ? 'leak' : ''),
        startS: _d(j['start_s']),
        endS: _d(j['end_s']),
        persons: [for (final x in (j['persons'] as List? ?? const [])) _i(x)],
        severe: severe,
      );

  String get label => S.risk[type] ?? S.riskOther;
}

class TrackInfo {
  TrackInfo({required this.id, required this.startS, required this.endS, this.thumbnail});
  final int id;
  final double startS, endS;
  final String? thumbnail;
  factory TrackInfo.fromJson(Map<String, dynamic> j, String projectDir) => TrackInfo(
        id: _i(j['id']),
        startS: _d(j['start_s']),
        endS: _d(j['end_s']),
        thumbnail: j['thumbnail'] == null ? null : p.join(projectDir, j['thumbnail'] as String),
      );
}

/// Result of `devices`: which backend the user's rule picked, the adapter's name, the test run timing and the
/// hardware fingerprint (kept with the saved choice; a different fingerprint at start-up = check again).
class DeviceInfo {
  DeviceInfo({
    required this.chosen,
    this.deviceName,
    this.gpuBackends = const [],
    this.benchMs,
    this.benchCpuMs,
    this.fingerprint,
    this.cpuName,
  });
  final String chosen; // cuda | dml | cpu
  final String? deviceName;
  final List<String> gpuBackends; // GPU backends whose test inference worked
  final double? benchMs; // test run (det_10g, one 640x640 frame) on [chosen], ms
  final double? benchCpuMs; // the same test on the CPU
  final String? fingerprint;
  final String? cpuName;

  factory DeviceInfo.fromJson(Map<String, dynamic> j) {
    final bench = j['bench'] is Map ? (j['bench'] as Map).cast<String, dynamic>() : null;
    final sys = j['system'] is Map ? (j['system'] as Map).cast<String, dynamic>() : null;
    final cpu = sys?['cpu'] is Map ? (sys!['cpu'] as Map) : null;
    final chosen = (j['chosen'] as String?) ?? 'cpu';
    return DeviceInfo(
      chosen: chosen,
      deviceName: j['device_name'] as String?,
      gpuBackends: [
        for (final pr in (j['probes'] as List? ?? const []))
          if ((pr as Map)['ok'] == true && pr['backend'] != 'cpu') pr['backend'] as String
      ],
      benchMs: bench != null && bench['backend'] == chosen ? (bench['ms'] as num?)?.toDouble() : null,
      benchCpuMs: (bench?['cpu_ms'] as num?)?.toDouble(),
      fingerprint: sys?['fingerprint'] as String?,
      cpuName: cpu?['name'] as String?,
    );
  }

  /// What the settings file keeps (read back by [fromJson]).
  Map<String, dynamic> toJson() => {
        'chosen': chosen,
        'device_name': deviceName,
        'probes': [
          for (final b in gpuBackends) {'backend': b, 'ok': true}
        ],
        'bench': benchMs == null && benchCpuMs == null ? null : {'backend': chosen, 'ms': benchMs, 'cpu_ms': benchCpuMs},
        'system': {
          'fingerprint': fingerprint,
          'cpu': {'name': cpuName}
        },
      };

  bool get gpu => chosen != 'cpu';
  String get api => chosen == 'cuda' ? 'CUDA' : 'DirectML';
  String get plain {
    if (!gpu) return S.deviceCpu;
    return S.deviceGpu(deviceName ?? S.deviceGpuUnknown, api);
  }

  /// "Radeon (TM) Pro Graphics" -> "Radeon Pro Graphics".
  static String cleanName(String n) => n.replaceAll(RegExp(r'\s*\((TM|R|C)\)', caseSensitive: false), '').trim();

  /// Test-run speed-up of the chosen GPU over the CPU, when it is clear (>= 1.5x); null otherwise.
  double? get speedUp {
    final g = benchMs, c = benchCpuMs;
    if (!gpu || g == null || c == null || g <= 0) return null;
    final r = c / g;
    return r >= 1.5 ? r : null;
  }
}

String fmtDuration(double s) {
  final t = s.isFinite && s > 0 ? s.round() : 0;
  final h = t ~/ 3600, m = (t % 3600) ~/ 60, sec = t % 60;
  final ss = sec.toString().padLeft(2, '0');
  return h > 0 ? '$h:${m.toString().padLeft(2, '0')}:$ss' : '$m:$ss';
}

String fmtSeconds(double s) => s.toStringAsFixed(1).replaceAll('.', ',');
