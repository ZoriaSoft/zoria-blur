// App settings kept between runs: theme, sidebar, recent projects, first-run setup done, default output
// folder and the last system check (device choice + hardware fingerprint, so start-up need not re-probe).
// Stored as JSON in %LOCALAPPDATA%\ZoriaBlur\settings.json (ZORIA_SETTINGS_DIR overrides, used by the
// self-test so a build machine's own list is not touched). Window size / position are kept by the
// Windows runner itself (registry), so the first frame already has the right size.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;

import 'engine.dart';

class RecentProject {
  RecentProject({
    required this.video,
    required this.project,
    required this.opened,
    this.thumb,
    this.durationS = 0,
    this.persons = 0,
    this.selected,
    this.mode,
  });
  final String video;
  final String project;
  final DateTime opened;
  final String? thumb;
  final double durationS;
  final int persons;

  /// The people chosen in this project and the blur mode ('selected' / 'except'), so switching to another
  /// video and back never loses a selection. null = never saved.
  final List<int>? selected;
  final String? mode;

  String get name => p.basename(video);

  Map<String, Object?> toJson() => {
        'video': video,
        'project': project,
        'opened': opened.toIso8601String(),
        'thumb': thumb,
        'duration_s': durationS,
        'persons': persons,
        if (selected != null) 'selected': selected,
        if (mode != null) 'mode': mode,
      };

  static RecentProject? fromJson(Object? j) {
    if (j is! Map) return null;
    final v = j['video'], pr = j['project'];
    if (v is! String || pr is! String) return null;
    return RecentProject(
      video: v,
      project: pr,
      opened: DateTime.tryParse('${j['opened']}') ?? DateTime.fromMillisecondsSinceEpoch(0),
      thumb: j['thumb'] as String?,
      durationS: (j['duration_s'] as num?)?.toDouble() ?? 0,
      persons: (j['persons'] as num?)?.toInt() ?? 0,
      selected: (j['selected'] as List?)?.whereType<num>().map((x) => x.toInt()).toList(),
      mode: j['mode'] as String?,
    );
  }

  RecentProject copyWith(
          {DateTime? opened, String? thumb, double? durationS, int? persons, List<int>? selected, String? mode}) =>
      RecentProject(
        video: video,
        project: project,
        opened: opened ?? this.opened,
        thumb: thumb ?? this.thumb,
        durationS: durationS ?? this.durationS,
        persons: persons ?? this.persons,
        selected: selected ?? this.selected,
        mode: mode ?? this.mode,
      );
}

class AppSettings extends ChangeNotifier {
  AppSettings({this.file});

  /// null = in memory only (tests).
  final File? file;

  ThemeMode themeMode = ThemeMode.system;
  bool sidebarOpen = true;
  final List<RecentProject> recents = [];

  /// The first-run setup (welcome -> system check -> ready) was finished or skipped.
  bool setupDone = false;

  /// Where exports go by default; null = next to the video.
  String? outputDir;

  /// Last system check result (DeviceInfo.toCache + engine + app version); null = never checked.
  Map<String, dynamic>? deviceCache;
  static const maxRecents = 12;

  static File defaultFile() {
    final dir = Platform.environment['ZORIA_SETTINGS_DIR'] ?? Engine.dataDir;
    return File(p.join(dir, 'settings.json'));
  }

  static AppSettings load([File? f]) {
    final s = AppSettings(file: f ?? defaultFile());
    try {
      final file = s.file!;
      if (file.existsSync()) {
        final j = jsonDecode(file.readAsStringSync());
        if (j is Map) {
          s.themeMode = switch (j['theme']) {
            'light' => ThemeMode.light,
            'dark' => ThemeMode.dark,
            _ => ThemeMode.system,
          };
          s.sidebarOpen = j['sidebar'] != false;
          s.setupDone = j['setup_done'] == true;
          final od = j['output_dir'];
          s.outputDir = od is String && od.isNotEmpty ? od : null;
          final dc = j['device'];
          if (dc is Map) s.deviceCache = dc.cast<String, dynamic>();
          for (final r in (j['recents'] as List? ?? const [])) {
            final x = RecentProject.fromJson(r);
            if (x != null) s.recents.add(x);
          }
        }
      }
    } catch (_) {
      // a broken settings file is ignored (defaults), never fatal
    }
    return s;
  }

  void _save() {
    final f = file;
    if (f == null) return;
    try {
      f.parent.createSync(recursive: true);
      final tmp = File('${f.path}.tmp');
      tmp.writeAsStringSync(const JsonEncoder.withIndent('  ').convert({
        'theme': switch (themeMode) { ThemeMode.light => 'light', ThemeMode.dark => 'dark', _ => 'system' },
        'sidebar': sidebarOpen,
        'recents': [for (final r in recents) r.toJson()],
        'setup_done': setupDone,
        'output_dir': outputDir,
        'device': deviceCache,
      }));
      tmp.renameSync(f.path);
    } catch (_) {}
  }

  void setThemeMode(ThemeMode m) {
    themeMode = m;
    _save();
    notifyListeners();
  }

  void setSetupDone() {
    setupDone = true;
    _save();
    notifyListeners();
  }

  void setOutputDir(String? dir) {
    outputDir = dir == null || dir.isEmpty ? null : dir;
    _save();
    notifyListeners();
  }

  void setDeviceCache(Map<String, dynamic>? c) {
    deviceCache = c;
    _save();
  }

  void toggleSidebar() {
    sidebarOpen = !sidebarOpen;
    _save();
    notifyListeners();
  }

  /// Adds or refreshes a project at the top of the list (one entry per video). A new project for the same path
  /// (another file there, searched again) replaces the entry whole: the old project's people never carry over.
  void touchRecent(RecentProject r) {
    final old = recents.where((x) => _same(x.video, r.video)).firstOrNull;
    recents.removeWhere((x) => _same(x.video, r.video));
    recents.insert(
        0,
        old == null || !_same(old.project, r.project)
            ? r
            : old.copyWith(
                opened: r.opened,
                thumb: r.thumb ?? old.thumb,
                durationS: r.durationS > 0 ? r.durationS : old.durationS,
                persons: r.persons > 0 ? r.persons : old.persons,
                selected: r.selected,
                mode: r.mode));
    if (recents.length > maxRecents) recents.removeRange(maxRecents, recents.length);
    _save();
    notifyListeners();
  }

  /// The project's current selection, saved in place (the list order does not change).
  /// Returns false when the video is not in the list.
  bool saveSelection(String video, List<int> selected, String mode) {
    final i = recents.indexWhere((x) => _same(x.video, video));
    if (i < 0) return false;
    final old = recents[i];
    final ids = [...selected]..sort();
    if (old.mode == mode && old.selected != null && _eq(old.selected!, ids)) return true;
    recents[i] = old.copyWith(selected: ids, mode: mode);
    _save();
    return true;
  }

  /// Forgets the chosen people of a project whose video changed (another file at the same path): its ids belong to
  /// the old file and must never be applied to the people of a new face search.
  void clearSelection(String video) {
    final i = recents.indexWhere((x) => _same(x.video, video));
    if (i < 0) return;
    final o = recents[i];
    if (o.selected == null && o.mode == null) return;
    recents[i] = RecentProject(
        video: o.video, project: o.project, opened: o.opened, thumb: o.thumb, durationS: o.durationS, persons: o.persons);
    _save();
  }

  RecentProject? recentFor(String video) => recents.where((x) => _same(x.video, video)).firstOrNull;

  static bool _eq(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  void removeRecent(RecentProject r) {
    recents.removeWhere((x) => _same(x.video, r.video));
    _save();
    notifyListeners();
  }

  static bool _same(String a, String b) => p.normalize(a).toLowerCase() == p.normalize(b).toLowerCase();
}
