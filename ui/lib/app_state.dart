import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';
import 'package:path/path.dart' as p;

import 'engine.dart';
import 'live_player.dart';
import 'models.dart';
import 'settings.dart';
import 'strings.dart';
import 'system_check.dart';
import 'version.dart';

enum Stage { start, ready, analyzing, editing, exporting, done }

enum BlurMode { selected, except }

enum Proc { auto, cpu, gpu }

/// Video files the picker shows and drag-drop accepts (containers ffmpeg reads). Images and other files are refused.
const videoExtensions = [
  'mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'wmv', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts', '3gp', 'flv', 'ogv',
];

bool isVideoPath(String path) =>
    videoExtensions.contains(p.extension(path).toLowerCase().replaceFirst('.', ''));

class AppState extends ChangeNotifier {
  AppState({AppSettings? settings, SystemCheck? check})
      : settings = settings ?? AppSettings(),
        check = check ?? SystemCheck();

  /// Theme, sidebar and recent projects (kept between runs).
  final AppSettings settings;

  Stage stage = Stage.start;
  String? error; // shown as a banner; cleared on the next action
  String? errorDetail;
  String? toast;

  // engine + device
  List<EngineCommand> _engines = [];
  @visibleForTesting
  set engines(List<EngineCommand> e) => _engines = e;
  Engine? engine;
  DeviceInfo? device;
  bool deviceChecking = true;

  /// The live system check (first-run setup, Yardım -> Sistem kontrolünü tekrar çalıştır, or a changed PC).
  final SystemCheck check;

  /// First-run setup on screen; [wizardStep] 0 = Hoş geldiniz, 1 = Sistem, 2 = Hazır.
  bool wizardOpen = false;
  int wizardStep = 0;
  Proc proc = Proc.auto;
  String detMode = 'balanced';

  // video + project
  VideoInfo? video;
  String? projectDir;
  List<Person> persons = [];
  List<Risk> risks = [];

  /// When each person is on screen (seconds), from the engine's `tracks`: person id -> [[start, end], ...].
  Map<int, List<List<double>>> presence = {};
  int _presenceGen = 0;
  final Set<int> selected = {};
  BlurMode mode = BlurMode.selected;
  double strength = 2.0;

  // progress
  double progress = 0; // 0..1, <0 = indeterminate
  String phase = '';
  String etaText = '';
  int stepNo = 1, stepCount = 1;
  EngineJob? _job;
  DateTime? _phaseStart;
  String _phaseKey = '';

  // preview
  int frame = 0;
  Uint8List? previewBytes;
  bool previewBusy = false;
  EngineJob? _previewJob;
  Timer? _previewDebounce;
  int _previewSeq = 0;
  bool drawMode = false;

  // live preview (playing)
  final player = LivePlayer();
  final Map<String, LivePlan> _plans = {};
  final Set<String> _planJobs = {};
  int _planGen = 0;

  // a video passed on the command line / by a second launch before the engine was ready
  String? _pendingOpen;

  // export result
  String? outputPath;
  Map<String, dynamic>? qa;
  List<Risk> leaks = [];

  /// Risky moments only the export's QA can know (a sharp face nobody was tracked for, a face-like spot left
  /// open). They join the timeline so the done panel and the timeline count the same moments.
  List<Risk> qaExtra = [];
  static const qaOnlyTypes = {'unknown_face', 'unconfirmed_face'};

  bool get busy => stage == Stage.analyzing || stage == Stage.exporting;

  /// Engine phase of the running job (analyze / render / qa); the self-test waits on it.
  String get phaseKey => _phaseKey;

  // ---------------------------------------------------------------- startup
  /// Start-up device choice, fastest path first:
  ///  * first run (setup on screen): the setup's system check decides; it starts on the "Sistem" step (or when
  ///    the setup is skipped), so the user sees it run;
  ///  * a saved choice for this app version and engine: used at once, then checked quietly in the background
  ///    (`devices --only <backend> --no-test` + hardware fingerprint); a changed PC or a failing device runs the
  ///    full check again without the setup;
  ///  * otherwise the full check, quietly.
  Future<void> init() async {
    _engines = EngineLocator.locateAll();
    player.onEnded = _onPlayEnded;
    if (_engines.isEmpty) {
      deviceChecking = false;
      // the setup's check has nothing to run: show the error on the main window instead of a check that never
      // starts (setup_done stays false, so a repaired install still gets the setup)
      wizardOpen = false;
      _fail(EngineError(S.errEngineMissing));
      return;
    }
    if (wizardOpen) {
      notifyListeners();
      return; // the setup starts the check (startSystemCheck)
    }
    if (_applyCache()) {
      deviceChecking = false;
      _applyProc();
      notifyListeners();
      unawaited(_verifyQuietly());
      await _openPending();
      return;
    }
    await startSystemCheck();
  }

  /// Runs (or joins) the full system check and applies its choice. Never throws; ends on the CPU at worst.
  Future<void> startSystemCheck() async {
    if (_engines.isEmpty) return;
    if (check.running) {
      await check.run(_engines); // joins the running one
      return;
    }
    deviceChecking = true;
    notifyListeners();
    CheckResult r;
    try {
      r = await check.run(_engines);
    } catch (_) {
      r = CheckResult(_engines.last, DeviceInfo(chosen: 'cpu'), why: S.whyCpuFailed, probeFailed: true);
    }
    _applyResult(r.cmd, r.device, save: !r.probeFailed);
    deviceChecking = false;
    _applyProc();
    notifyListeners();
    await _openPending();
  }

  void _applyResult(EngineCommand cmd, DeviceInfo d, {bool save = true}) {
    engine = Engine(cmd, backend: d.chosen);
    device = d;
    if (save) {
      settings.setDeviceCache({
        'app_version': appVersion,
        'engine': cmd.exe,
        'engine_label': cmd.label,
        'checked': DateTime.now().toIso8601String(),
        ...d.toJson(),
      });
    }
  }

  /// The saved choice, if it fits this install (same app version, engine still there).
  bool _applyCache() {
    final c = settings.deviceCache;
    if (c == null || c['app_version'] != appVersion) return false;
    final cmd = _engines.where((e) => e.exe == c['engine'] && e.label == c['engine_label']).firstOrNull;
    if (cmd == null) return false;
    final d = DeviceInfo.fromJson(c);
    engine = Engine(cmd, backend: d.chosen);
    device = d;
    return true;
  }

  /// Background check of a saved choice: same hardware and the device still works, else the full check again.
  Future<void> _verifyQuietly() async {
    final e = engine, d = device;
    if (e == null || d == null) return;
    var ok = false;
    try {
      final r = await Engine(e.cmd).run(['devices', '--only', d.chosen, '--no-test'],
          withBackend: false, timeout: const Duration(seconds: 60));
      final now = DeviceInfo.fromJson(r);
      final fpSame = d.fingerprint == null || now.fingerprint == d.fingerprint;
      final works = d.chosen == 'cpu' || now.gpuBackends.contains(d.chosen);
      ok = fpSame && works;
    } catch (_) {
      ok = false;
    }
    quietCheckResult = ok ? 'same' : 'changed';
    if (ok) return;
    while (busy) {
      await Future<void>.delayed(const Duration(seconds: 1)); // never swap the engine under a running job
    }
    final before = device?.chosen;
    final beforeName = device?.deviceName;
    await startSystemCheck();
    if (device?.chosen != before || device?.deviceName != beforeName) {
      toast = S.deviceChanged;
      notifyListeners();
    }
  }

  /// Outcome of the background check of a saved choice ('same' / 'changed'); null = not done (self-test log).
  String? quietCheckResult;

  Future<void> _openPending() async {
    final pend = _pendingOpen;
    _pendingOpen = null;
    if (pend != null) await openExternal(pend);
  }

  // ---------------------------------------------------------------- first-run setup
  void wizardGo(int step) {
    wizardStep = step.clamp(0, 2);
    if (wizardStep >= 1 && !check.started) unawaited(startSystemCheck());
    notifyListeners();
  }

  /// Setup closed ("Bitir" or "Şimdilik geç"): never shown again; a check not started yet runs in the background.
  void finishWizard({bool? besideVideo, String? outputDir}) {
    if (besideVideo == true) settings.setOutputDir(null);
    if (besideVideo == false && outputDir != null) settings.setOutputDir(outputDir);
    settings.setSetupDone();
    wizardOpen = false;
    if (!check.started) unawaited(startSystemCheck());
    notifyListeners();
  }

  /// Yardım -> Sistem kontrolünü tekrar çalıştır: the setup's "Sistem" step with a fresh check.
  void rerunSystemCheck() {
    if (busy || check.running) return;
    unawaited(player.stop());
    check.reset();
    wizardOpen = true;
    wizardStep = 1;
    notifyListeners();
    unawaited(startSystemCheck());
  }

  /// A video from the command line or forwarded by a second launch of the app.
  Future<void> openExternal(String path) => openPicked(path);

  /// A video chosen by the user ("Video aç…", Ctrl+O, drag-drop, command line, a second launch):
  ///  * while the PC is still being checked it waits and opens as soon as the check ends (0.1.8 showed
  ///    "engine missing" for a video dropped during the first check);
  ///  * a video that is already in "Son projeler" with a finished face search reopens that project with its
  ///    chosen people, the same as clicking it in the list (no second face search, no lost choices);
  ///  * anything else opens fresh.
  Future<void> openPicked(String path) async {
    if (deviceChecking) {
      if (isVideoPath(path)) {
        _pendingOpen = path;
      } else {
        _clearError();
        error = S.notAVideo;
        notifyListeners();
      }
      return;
    }
    if (busy) return;
    await player.stop();
    final r = settings.recentFor(path);
    if (r != null && isVideoPath(path) && File(p.join(r.project, 'project.json')).existsSync()) {
      await openRecent(r);
    } else {
      await openVideo(path);
    }
  }

  void setProc(Proc v) {
    proc = v;
    _applyProc();
    notifyListeners();
  }

  void _applyProc() {
    final d = device;
    final e = engine;
    if (d == null || e == null) return;
    e.backend = switch (proc) {
      Proc.auto => d.chosen,
      Proc.cpu => 'cpu',
      Proc.gpu => d.gpuBackends.isNotEmpty ? d.gpuBackends.first : 'dml',
    };
  }

  String get deviceText {
    if (deviceChecking) return S.deviceChecking;
    final d = device;
    if (d == null) return '';
    if (proc == Proc.cpu) return S.deviceCpu;
    if (proc == Proc.gpu && d.gpuBackends.isEmpty) return S.gpuNotUsable;
    return d.plain;
  }

  void setDetMode(String m) {
    detMode = m;
    notifyListeners();
  }

  // ---------------------------------------------------------------- video
  Future<void> openVideo(String path) async {
    if (busy) return;
    _clearError();
    if (!isVideoPath(path)) {
      error = S.notAVideo;
      notifyListeners();
      return;
    }
    final e = engine;
    if (e == null) {
      _fail(EngineError(S.errEngineMissing));
      return;
    }
    try {
      final r = await e.run(['probe', path], withBackend: false);
      _rememberSelection(); // the project being left keeps its chosen people in "Son projeler"
      video = VideoInfo.fromJson(r, path: path);
      await player.stop();
      projectDir = null;
      _plans.clear();
      _planGen++;
      persons = [];
      risks = [];
      presence = {};
      selected.clear();
      previewBytes = null;
      outputPath = null;
      qa = null;
      leaks = [];
      qaExtra = [];
      stage = Stage.ready;
      notifyListeners();
    } catch (err) {
      _fail(err);
    }
  }

  void reset() {
    if (busy) return;
    _rememberSelection();
    player.stop();
    _clearError();
    video = null;
    stage = Stage.start;
    notifyListeners();
  }

  String _projectDirFor(String video) {
    final f = File(video);
    final stat = f.existsSync() ? f.statSync() : null;
    final key = '${p.basenameWithoutExtension(video)}_${(video.hashCode ^ (stat?.size ?? 0)).toUnsigned(32).toRadixString(16)}';
    return p.join(Engine.dataDir, 'projects', key);
  }

  // ---------------------------------------------------------------- analyze
  Future<void> analyze() async {
    final v = video, e = engine;
    if (v == null || e == null || busy) return;
    _clearError();
    final pdir = _projectDirFor(v.path);
    stage = Stage.analyzing;
    _beginPhase('start', S.phaseStart, step: 1, of: 1);
    notifyListeners();
    final job = e.start(['analyze', v.path, '-p', pdir, '--det-mode', detMode], onEvent: _onEvent);
    _job = job;
    try {
      final r = await job.result;
      _loadProject(r);
      selected.clear();
      leaks = [];
      qaExtra = [];
      qa = null;
      frame = _initialFrame();
      stage = Stage.editing;
      _rememberProject();
      notifyListeners();
      requestPreview(immediate: true);
      unawaited(_makeRecentThumb(frame));
    } on EngineCancelled {
      stage = Stage.ready;
      toast = S.cancelled;
      notifyListeners();
    } catch (err) {
      stage = Stage.ready;
      _fail(err);
    } finally {
      _job = null;
    }
  }

  Future<void> cancelJob() async {
    await _job?.cancel();
  }

  int _initialFrame() {
    if (persons.isEmpty) return 0;
    final top = [...persons]..sort((a, b) => b.seconds.compareTo(a.seconds));
    return (top.first.bestFrame ?? 0).clamp(0, (video?.frameCount ?? 1) - 1);
  }

  void _beginPhase(String key, String text, {int? step, int? of}) {
    _phaseKey = key;
    phase = text;
    progress = -1;
    etaText = '';
    _phaseStart = DateTime.now();
    if (step != null) stepNo = step;
    if (of != null) stepCount = of;
  }

  void _onEvent(Map<String, dynamic> ev) {
    final kind = ev['event'];
    if (kind == 'progress') {
      final ph = ev['phase'] as String? ?? '';
      final f = (ev['frame'] as num?)?.toDouble() ?? 0, tot = (ev['total'] as num?)?.toDouble() ?? 0;
      if (ph != _phaseKey) {
        final text = switch (ph) {
          'analyze' => S.phaseFaces,
          'render' => S.phaseRender,
          'qa' => S.phaseQa,
          _ => S.phaseStart,
        };
        _beginPhase(ph, text, step: stage == Stage.exporting ? (ph == 'qa' ? 2 : 1) : null);
      }
      if (tot > 0) {
        progress = (f / tot).clamp(0.0, 1.0);
        if (ph == 'analyze' && progress >= 0.995) {
          phase = S.phaseGrouping;
          etaText = '';
          progress = -1;
        } else {
          _updateEta();
        }
      }
      notifyListeners();
    } else if (kind == 'done' && ev['phase'] == 'analyze') {
      phase = S.phaseGrouping;
      progress = -1;
      notifyListeners();
    }
  }

  void _updateEta() {
    final s = _phaseStart;
    if (s == null || progress < 0.03) {
      etaText = S.etaCalc;
      return;
    }
    final el = DateTime.now().difference(s).inMilliseconds / 1000.0;
    final left = el * (1 - progress) / progress;
    etaText = S.eta(left < 60 ? S.lessThanMinute : S.minutes((left / 60).ceil()));
  }

  void _loadProject(Map<String, dynamic> r) {
    final pdir = r['project'] as String;
    projectDir = pdir;
    persons = [for (final x in (r['persons'] as List? ?? const [])) Person.fromJson(x as Map<String, dynamic>, pdir)]
      ..sort((a, b) => b.seconds.compareTo(a.seconds));
    risks = [for (final x in (r['risks'] as List? ?? const [])) Risk.fromJson(x as Map<String, dynamic>)];
    final vj = r['video'] as Map<String, dynamic>?;
    if (vj != null && video != null) video = VideoInfo.fromJson(vj, path: video!.path);
    selected.removeWhere((id) => !persons.any((x) => x.id == id));
    PaintingBinding.instance.imageCache.clear(); // thumbnails are rewritten after every edit
    _plans.clear(); // merge / split / addbox change the plan
    _planGen++;
    unawaited(_loadPresence());
  }

  Future<void> _loadPresence() async {
    final e = engine, pd = projectDir;
    if (e == null || pd == null) return;
    final gen = ++_presenceGen;
    try {
      final r = await e.run(['tracks', pd], withBackend: false);
      final out = <int, List<List<double>>>{};
      for (final t in (r['tracks'] as List? ?? const [])) {
        final m = t as Map<String, dynamic>;
        final pid = (m['person_id'] as num?)?.toInt(), a = (m['start_s'] as num?)?.toDouble();
        final b = (m['end_s'] as num?)?.toDouble();
        if (pid == null || a == null || b == null) continue;
        (out[pid] ??= []).add([a, b]);
      }
      for (final l in out.values) {
        l.sort((x, y) => x[0].compareTo(y[0]));
      }
      if (gen == _presenceGen && pd == projectDir) {
        presence = out;
        notifyListeners();
      }
    } catch (_) {
      // presence lanes are a help, not a must: the timeline just shows no lanes
    }
  }

  // ---------------------------------------------------------------- recent projects
  void _rememberProject() {
    final v = video, pd = projectDir;
    if (v == null || pd == null) return;
    settings.touchRecent(RecentProject(
        video: v.path, project: pd, opened: DateTime.now(), durationS: v.durationS, persons: persons.length));
  }

  /// Small picture of the video for the recent list: a frame with everyone blurred.
  Future<void> _makeRecentThumb(int f) async {
    final e = engine, v = video, pd = projectDir;
    if (e == null || v == null || pd == null) return;
    final out = p.join(pd, 'ui_cache', 'recent.png');
    try {
      await e.run(['preview', v.path, '-p', pd, '--frame', '$f', '--mode', 'blur-except', '-o', out],
          withBackend: false);
      PaintingBinding.instance.imageCache.evict(FileImage(File(out)));
      settings.touchRecent(RecentProject(video: v.path, project: pd, opened: DateTime.now(), thumb: out));
    } catch (_) {}
  }

  /// Saves the open project's chosen people + mode into its "Son projeler" entry (no-op when unchanged).
  void _rememberSelection() {
    final v = video;
    if (v == null || projectDir == null || persons.isEmpty) return;
    settings.saveSelection(v.path, selected.toList(), mode == BlurMode.selected ? 'selected' : 'except');
  }

  /// Restores a saved selection (only ids that still exist in the project).
  void _restoreSelection(String videoPath) {
    final r = settings.recentFor(videoPath);
    final ids = r?.selected;
    if (r == null || ids == null) return;
    selected
      ..clear()
      ..addAll(ids.where((id) => persons.any((x) => x.id == id)));
    if (r.mode == 'except') mode = BlurMode.except;
    if (r.mode == 'selected') mode = BlurMode.selected;
  }

  @visibleForTesting
  void restoreSelectionFor(String videoPath) => _restoreSelection(videoPath);

  bool isCurrent(RecentProject r) =>
      video != null && p.normalize(video!.path).toLowerCase() == p.normalize(r.video).toLowerCase();

  /// Opens a recent project: the video, and its people straight away when the face search was done before.
  Future<void> openRecent(RecentProject r) async {
    if (busy || deviceChecking) return;
    _clearError();
    if (!File(r.video).existsSync()) {
      error = S.errNoFile;
      errorDetail = r.video;
      notifyListeners();
      return;
    }
    await player.stop();
    await openVideo(r.video);
    if (stage != Stage.ready) return;
    final e = engine, probed = video;
    if (e == null || probed == null || !File(p.join(r.project, 'project.json')).existsSync()) return;
    // the saved people and tracks belong to the file that was searched then: a different file at the same path
    // (a camera reusing GOPR0001.MP4, an overwritten recording) must never get them, or its real faces stay sharp
    if (!p.equals(_projectDirFor(r.video), r.project)) {
      // file size is part of the project key: another file now
      settings.clearSelection(r.video);
      toast = S.videoChanged;
      notifyListeners();
      return;
    }
    try {
      final res = await e.run(['persons', r.project], withBackend: false);
      // something else was opened (or the face search started) while the project loaded: never put this
      // project's people and tracks on another video
      if (!identical(video, probed) || stage != Stage.ready) return;
      final vj = res['video'];
      if (vj is! Map || !sameVideo(probed, VideoInfo.fromJson(vj.cast<String, dynamic>(), path: probed.path))) {
        settings.clearSelection(r.video); // the new face search writes the same project folder: old ids must go
        toast = S.videoChanged; // stays on "Başlat": a new face search
        notifyListeners();
        return;
      }
      _loadProject(res);
      _restoreSelection(r.video);
      frame = _initialFrame();
      stage = Stage.editing;
      _rememberProject();
      notifyListeners();
      requestPreview(immediate: true);
    } catch (_) {
      // an old / broken project: stay on "Başlat" (a new face search)
    }
  }

  /// Same video as the one a project was made from: size in pixels, frame rate and frame count (the file size is
  /// checked through the project key). [probed] is ffprobe's estimate, the project holds the decoded count, which
  /// can be a few frames less (test.mp4: 968 vs 966), so the count may differ by 1 % (at least 5 frames).
  static bool sameVideo(VideoInfo probed, VideoInfo project) {
    if (probed.width != project.width || probed.height != project.height) return false;
    if (probed.fps > 0 && project.fps > 0 && (probed.fps - project.fps).abs() > 0.5) return false;
    final a = probed.frameCount, b = project.frameCount;
    if (a <= 0 || b <= 0) return false;
    final tol = (a * 0.01).ceil() < 5 ? 5 : (a * 0.01).ceil();
    return (a - b).abs() <= tol;
  }

  // ---------------------------------------------------------------- persons
  void toggle(int id) {
    if (!selected.remove(id)) selected.add(id);
    notifyListeners();
    requestPreview();
  }

  // ---------------------------------------------------------------- live preview
  String _planKey(BlurMode m, Iterable<int> ids) =>
      '${m == BlurMode.selected ? 'blur-selected' : 'blur-except'}:${(ids.toList()..sort()).join(',')}';
  String get _currentPlanKey => _planKey(mode, selected);
  String get _allPlanKey => _planKey(BlurMode.except, const []);

  /// Regions to blur at [f] while playing; null = blur the whole frame (no plan yet).
  /// The current selection's plan, else the "everyone blurred" plan (fail-safe while it is computed).
  List<List<double>>? liveRegionsAt(int f) {
    if (mode == BlurMode.selected && selected.isEmpty) return const [];
    final plan = _plans[_currentPlanKey] ?? _plans[_allPlanKey];
    return plan?.at(f);
  }

  /// True while the preview shows the fail-safe plan instead of the current selection's.
  bool get livePlanPending => !(mode == BlurMode.selected && selected.isEmpty) && !_plans.containsKey(_currentPlanKey);

  void _ensurePlans() {
    if (projectDir == null) return;
    _fetchPlan(_allPlanKey, BlurMode.except, const []);
    if (!(mode == BlurMode.selected && selected.isEmpty)) _fetchPlan(_currentPlanKey, mode, selected.toList());
  }

  Future<void> _fetchPlan(String key, BlurMode m, List<int> ids) async {
    final e = engine, pd = projectDir;
    if (e == null || pd == null || _plans.containsKey(key) || _planJobs.contains(key)) return;
    final gen = _planGen;
    _planJobs.add(key);
    final out = p.join(pd, 'ui_cache', 'plan_${key.hashCode.toUnsigned(32).toRadixString(16)}.json');
    try {
      await e.run([
        'plan', pd, '-o', out,
        '--mode', m == BlurMode.selected ? 'blur-selected' : 'blur-except',
        if (ids.isNotEmpty) ...['--persons', (ids.toList()..sort()).join(',')],
      ], withBackend: false);
      final txt = await File(out).readAsString();
      final plan = await compute(_parsePlan, txt);
      if (gen == _planGen) {
        _plans[key] = plan;
        notifyListeners();
      }
    } catch (_) {
      // no plan = the preview keeps the fail-safe (everyone / whole frame blurred)
    } finally {
      _planJobs.remove(key);
    }
  }

  Future<void> play() async {
    final e = engine, v = video;
    if (e == null || v == null || stage != Stage.editing || player.playing) return;
    drawMode = false;
    _previewDebounce?.cancel();
    await _previewJob?.cancel();
    previewBusy = false;
    _ensurePlans();
    final start = frame >= v.frameCount - 1 ? 0 : frame;
    notifyListeners();
    await player.play(e, v.path,
        startFrame: start, frameCount: v.frameCount, fps: v.fps, width: v.width, height: v.height);
  }

  Future<void> pause() async {
    if (!player.playing) return;
    final f = player.frame;
    await player.stop();
    frame = f;
    notifyListeners();
    requestPreview(immediate: true);
  }

  void _onPlayEnded() {
    frame = player.frame;
    notifyListeners();
    requestPreview(immediate: true);
  }

  void selectAll(bool all) {
    selected
      ..clear()
      ..addAll(all ? persons.map((x) => x.id) : const <int>[]);
    notifyListeners();
    requestPreview();
  }

  void setMode(BlurMode m) {
    mode = m;
    notifyListeners();
    requestPreview();
  }

  void setStrength(double v, {bool commit = false}) {
    strength = v;
    notifyListeners();
    if (commit) requestPreview();
  }

  Future<void> _projectCmd(List<String> args) async {
    final e = engine;
    if (e == null) return;
    _clearError();
    try {
      final r = await e.run(args, withBackend: false);
      _loadProject(r);
      notifyListeners();
      requestPreview();
    } catch (err) {
      _fail(err);
    }
  }

  Future<void> rename(int id, String name) async {
    final n = name.trim();
    if (n.isEmpty || projectDir == null) return;
    await _projectCmd(['rename', projectDir!, '$id', n]);
  }

  Future<void> mergeSelected() async {
    if (selected.length < 2 || projectDir == null) return;
    // keep the person seen longest as the surviving id
    final ids = persons.where((x) => selected.contains(x.id)).map((x) => x.id).toList();
    await _projectCmd(['merge', projectDir!, ...ids.map((x) => '$x')]);
    selected
      ..clear()
      ..add(ids.first);
    _rememberSelection();
    notifyListeners();
  }

  Future<List<TrackInfo>> tracksOf(int id) async {
    final e = engine, pd = projectDir;
    if (e == null || pd == null) return [];
    try {
      final r = await e.run(['tracks', pd, '--person', '$id'], withBackend: false);
      return [for (final t in (r['tracks'] as List? ?? const [])) TrackInfo.fromJson(t as Map<String, dynamic>, pd)];
    } catch (err) {
      _fail(err);
      return [];
    }
  }

  Future<void> splitTracks(List<int> trackIds) async {
    if (trackIds.isEmpty || projectDir == null) return;
    await _projectCmd(['split', projectDir!, '--tracks', trackIds.join(',')]);
  }

  // ---------------------------------------------------------------- preview
  List<String> _renderArgs() => [
        '--mode',
        mode == BlurMode.selected ? 'blur-selected' : 'blur-except',
        if (selected.isNotEmpty) ...['--persons', (selected.toList()..sort()).join(',')],
        '--strength',
        strength.toStringAsFixed(2),
      ];

  void setFrame(int f, {bool commit = false}) {
    if (player.playing) player.stop(); // scrubbing pauses playback
    final n = video?.frameCount ?? 1;
    frame = f.clamp(0, n > 0 ? n - 1 : 0);
    notifyListeners();
    if (commit) requestPreview();
  }

  /// Arrow keys / step buttons: one frame back or forward (pauses playback).
  void stepFrame(int delta) {
    if (video == null || stage != Stage.editing) return;
    final from = player.playing ? player.frame : frame;
    setFrame(from + delta, commit: true);
  }

  void togglePlay() {
    if (stage != Stage.editing) return;
    player.playing ? pause() : play();
  }

  void requestPreview({bool immediate = false}) {
    if (projectDir == null || video == null || stage != Stage.editing && stage != Stage.done) return;
    _rememberSelection(); // every selection / mode change ends here
    _ensurePlans(); // the live preview follows every selection / mode change
    if (player.playing) return; // the live player shows the frames; the exact frame comes on pause
    _previewDebounce?.cancel();
    // busy from the request on, so the spinner (and the self-test) never shows a stale frame as current
    if (!previewBusy) {
      previewBusy = true;
      notifyListeners();
    }
    _previewDebounce = Timer(Duration(milliseconds: immediate ? 0 : 180), _runPreview);
  }

  Future<void> _runPreview() async {
    final e = engine, v = video, pd = projectDir;
    if (e == null || v == null || pd == null) {
      previewBusy = false;
      return;
    }
    await _previewJob?.cancel();
    final seq = ++_previewSeq;
    final out = p.join(pd, 'ui_cache', 'preview_$seq.png');
    previewBusy = true;
    notifyListeners();
    final job = e.start(['preview', v.path, '-p', pd, '--frame', '$frame', ..._renderArgs(), '-o', out],
        withBackend: false);
    _previewJob = job;
    try {
      await job.result;
      final f = File(out);
      final bytes = await f.readAsBytes();
      unawaited(f.delete().catchError((_) => f));
      if (seq == _previewSeq) previewBytes = bytes;
    } on EngineCancelled {
      return;
    } catch (err) {
      if (seq == _previewSeq) _fail(err);
    } finally {
      if (seq == _previewSeq) {
        previewBusy = false;
        notifyListeners();
      }
    }
  }

  void setDrawMode(bool v) {
    if (v && player.playing) pause();
    drawMode = v;
    notifyListeners();
  }

  Future<void> addBox(int personId, List<double> box) async {
    final e = engine, v = video, pd = projectDir;
    if (e == null || v == null || pd == null) return;
    _clearError();
    try {
      final r = await e.run([
        'addbox', v.path, '-p', pd, '--person', '$personId', '--frame', '$frame',
        '--box', box.map((x) => x.toStringAsFixed(1)).join(','),
      ], withBackend: false);
      _loadProject(r);
      final pid = (r['person'] as num?)?.toInt();
      if (pid != null && mode == BlurMode.selected) selected.add(pid);
      drawMode = false;
      toast = S.boxAdded;
      notifyListeners();
      requestPreview(immediate: true);
    } catch (err) {
      _fail(err);
    }
  }

  /// Timeline marks: analysis risks that involve someone who gets blurred (or nobody known), the QA-only
  /// moments of the last export, and its leaks. The done panel counts exactly these (see [qaSummary]).
  List<Risk> get visibleRisks {
    final blur = mode == BlurMode.selected
        ? selected
        : persons.map((x) => x.id).where((id) => !selected.contains(id)).toSet();
    return [
      ...risks.where((r) => r.persons.isEmpty || r.persons.any(blur.contains)),
      ...qaExtra,
      ...leaks,
    ];
  }

  /// Risky ranges merged into moments for the timeline: overlapping or near (< 0.5 s apart) ranges are one
  /// moment (the engine reports one range per cause, often several for the same second).
  /// Leaks are merged only with each other and drawn last (on top), so a long yellow stretch can never swallow a
  /// second leak into one red mark.
  List<Risk> get riskMoments => [...mergeMoments(_visibleNonLeak), ...leakMoments];

  List<Risk> get _visibleNonLeak {
    final l = leaks.toSet();
    return visibleRisks.where((r) => !l.contains(r)).toList();
  }

  /// Overlapping or near (< 0.5 s apart) ranges become one moment.
  static List<Risk> mergeMoments(List<Risk> risks) {
    final rs = [...risks]..sort((a, b) => a.startS.compareTo(b.startS));
    final out = <Risk>[];
    for (final r in rs) {
      if (out.isNotEmpty && r.startS <= out.last.endS + 0.5) {
        final l = out.removeLast();
        out.add(Risk(
          type: l.severe || r.severe ? 'leak' : (l.type == r.type ? l.type : ''),
          startS: l.startS,
          endS: r.endS > l.endS ? r.endS : l.endS,
          persons: {...l.persons, ...r.persons}.toList(),
          severe: l.severe || r.severe,
        ));
      } else {
        out.add(r);
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- export
  bool get canExport => mode == BlurMode.except || selected.isNotEmpty;

  String suggestedOutputName() {
    final v = video!;
    return '${p.basenameWithoutExtension(v.path)}${S.outSuffix}.mp4';
  }

  /// The folder chosen in the setup, if it still exists; else the video's own folder.
  String suggestedOutputDir() {
    final d = settings.outputDir;
    if (d != null && Directory(d).existsSync()) return d;
    return p.dirname(video!.path);
  }

  Future<void> export(String outPath) async {
    final e = engine, v = video, pd = projectDir;
    if (e == null || v == null || pd == null || busy || !canExport) return;
    _clearError();
    if (!outPath.toLowerCase().endsWith('.mp4')) outPath = '$outPath.mp4';
    await player.stop();
    stage = Stage.exporting;
    _beginPhase('start', S.phaseRender, step: 1, of: 2);
    notifyListeners();
    _previewDebounce?.cancel();
    await _previewJob?.cancel();
    previewBusy = false;
    final job = e.start(['render', v.path, '-p', pd, ..._renderArgs(), '-o', outPath], onEvent: _onEvent);
    _job = job;
    try {
      final r = await job.result;
      takeExportResult(r, outPath);
      stage = Stage.done;
      notifyListeners();
    } on EngineCancelled {
      stage = Stage.editing;
      toast = S.cancelled;
      try {
        final f = File(outPath);
        if (f.existsSync()) f.deleteSync();
      } catch (_) {}
      notifyListeners();
    } catch (err) {
      stage = Stage.editing;
      _fail(err);
    } finally {
      _job = null;
    }
  }

  /// Result of `render` (with its QA) -> output path, leaks, QA-only risky moments.
  @visibleForTesting
  void takeExportResult(Map<String, dynamic> r, String outPath) {
    outputPath = r['output'] as String? ?? outPath;
    qa = r['qa'] as Map<String, dynamic>?;
    leaks = [
      for (final x in (qa?['leak_ranges'] as List? ?? const []))
        Risk.fromJson(x as Map<String, dynamic>, severe: true)
    ];
    qaExtra = [
      for (final x in (qa?['risky'] as List? ?? const []))
        if ((x as Map)['relevant'] == true && qaOnlyTypes.contains(x['type'])) Risk.fromJson(x.cast<String, dynamic>())
    ];
  }

  /// QA in plain words. The number of risky moments is the timeline's own count ([riskMoments]): one list, so the
  /// panel can never say 5 while the "Kontrol edilecek anlar" band shows 4 (0.1.6 counted the QA's `risky` list,
  /// which also held QA-only ranges and could be out of step with the project after merge / split).
  String get qaSummary {
    final q = qa;
    if (q == null) return S.qaSkipped;
    final nLeak = leakMoments.length;
    if (nLeak > 0) return S.qaLeaks(nLeak);
    final nRisk = riskMoments.length;
    if (nRisk > 0) return S.qaRiskOnly(nRisk);
    return S.qaClean;
  }

  bool get qaHasIssues => leaks.isNotEmpty;

  /// The red (leak) moments of the timeline: the done panel counts them and shows one jump button each (at the
  /// leak's own start), so "2 sızıntı" never sits next to 3 buttons or 1 red mark. Two leak ranges 0.3 s apart are
  /// one moment; leaks are never merged with yellow risks.
  List<Risk> get leakMoments => mergeMoments(leaks);

  void backToEdit({double? jumpToS}) {
    stage = Stage.editing;
    if (jumpToS != null && video != null) frame = (jumpToS * video!.fps).round().clamp(0, video!.frameCount - 1);
    notifyListeners();
    requestPreview(immediate: true);
  }

  Future<void> openFolder() async {
    final o = outputPath;
    if (o == null) return;
    await Process.run('explorer.exe', ['/select,', o]);
  }

  Future<void> openOutput() async {
    final o = outputPath;
    if (o == null) return;
    await Process.run('cmd', ['/c', 'start', '', o]);
  }

  /// Before the window closes: stop every engine process this app started.
  Future<void> shutdown() async {
    _rememberSelection();
    _previewDebounce?.cancel();
    try {
      await player.stop();
    } catch (_) {}
    await Engine.killAll(); // analyze / export / preview / plan / tracks - whatever is still running
  }

  // ---------------------------------------------------------------- errors
  void _fail(Object err) {
    if (err is EngineError) {
      error = err.message;
      errorDetail = err.detail;
    } else {
      error = S.errGeneric;
      errorDetail = '$err';
    }
    notifyListeners();
  }

  void _clearError() {
    error = null;
    errorDetail = null;
    toast = null;
  }

  void dismissError() {
    _clearError();
    notifyListeners();
  }
}

LivePlan _parsePlan(String txt) => LivePlan.fromJson(jsonDecode(txt) as Map<String, dynamic>);
