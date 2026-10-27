// After the export: the saved video plays right in the window (with sound), next to what the final check found and
// the next steps (Yeni video, Klasörde göster, Düzenlemeye dön). 0.1.9 showed a modal panel here that covered the
// whole window, so "Video aç…" was out of reach and "Videoyu izle" left the app for an outside player.
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:video_player/video_player.dart';

import '../app_state.dart';
import '../models.dart';
import '../strings.dart';
import '../theme.dart';

/// Opens a video for the result player; tests pass a fake (there is no video plugin in widget tests).
typedef ResultControllerFactory = VideoPlayerController Function(String path);

VideoPlayerController _fileController(String path) => VideoPlayerController.file(File(path));

/// Centre of the window once the video is saved: the saved file (or, one click away, the original) in a player.
class ResultView extends StatefulWidget {
  const ResultView({super.key, required this.st, this.controllerFor = _fileController});
  final AppState st;
  final ResultControllerFactory controllerFor;

  @override
  State<ResultView> createState() => _ResultViewState();
}

class _ResultViewState extends State<ResultView> {
  VideoPlayerController? _ctl;
  bool _original = false;
  bool _failed = false;
  bool _opening = false;
  String? _openedPath;
  int _session = 0;

  AppState get st => widget.st;

  @override
  void initState() {
    super.initState();
    st.resultTogglePlay = _togglePlay;
    st.resultSeekBy = _seekBy;
    st.resultSeekTo = _seekTo;
    _open(autoplay: false);
  }

  @override
  void didUpdateWidget(ResultView old) {
    super.didUpdateWidget(old);
    if (_pathFor(_original) != _openedPath) _open(autoplay: false);
  }

  @override
  void dispose() {
    _session++;
    if (st.resultTogglePlay == _togglePlay) st.resultTogglePlay = null;
    if (st.resultSeekBy == _seekBy) st.resultSeekBy = null;
    if (st.resultSeekTo == _seekTo) st.resultSeekTo = null;
    _ctl?.dispose();
    super.dispose();
  }

  String? _pathFor(bool original) => original ? st.video?.path : st.outputPath;

  /// (Re)opens the saved file or the original, keeping the position and the play state of the one before.
  Future<void> _open({required bool autoplay, Duration? at}) async {
    final path = _pathFor(_original);
    final session = ++_session;
    final old = _ctl;
    final pos = at ?? old?.value.position ?? Duration.zero;
    final wasPlaying = old?.value.isPlaying ?? false;
    st.resultPlayerState = null;
    setState(() {
      _opening = true;
      _failed = false;
      _openedPath = path;
    });
    if (path == null || !File(path).existsSync()) {
      st.resultPlayerState = 'error';
      setState(() {
        _opening = false;
        _failed = true;
      });
      return;
    }
    VideoPlayerController? ctl;
    try {
      ctl = widget.controllerFor(path);
      await ctl.initialize();
      if (session != _session) {
        await ctl.dispose();
        return;
      }
      await ctl.setLooping(false);
      if (pos > Duration.zero) await ctl.seekTo(pos);
      if (autoplay || wasPlaying) await ctl.play();
    } catch (_) {
      await ctl?.dispose().catchError((_) {});
      if (session != _session) return;
      st.resultPlayerState = 'error';
      setState(() {
        _opening = false;
        _failed = true;
      });
      return;
    }
    await old?.dispose();
    if (!mounted || session != _session) {
      await ctl.dispose();
      return;
    }
    ctl.addListener(() => st.resultPlayerPosition = ctl!.value.position);
    st.resultPlayerState = 'ready';
    setState(() {
      _ctl = ctl;
      _opening = false;
    });
  }

  void _togglePlay() {
    final c = _ctl;
    if (c == null || !c.value.isInitialized) return;
    if (c.value.isPlaying) {
      c.pause();
    } else {
      if (c.value.position >= c.value.duration - const Duration(milliseconds: 100)) c.seekTo(Duration.zero);
      c.play();
    }
  }

  void _seekBy(Duration d) {
    final c = _ctl;
    if (c == null || !c.value.isInitialized) return;
    var t = c.value.position + d;
    if (t < Duration.zero) t = Duration.zero;
    if (t > c.value.duration) t = c.value.duration;
    c.seekTo(t);
  }

  void _seekTo(double seconds) {
    final c = _ctl;
    if (c == null || !c.value.isInitialized) return;
    c.seekTo(Duration(milliseconds: (seconds * 1000).round()));
  }

  void _switch(bool original) {
    if (original == _original) return;
    setState(() => _original = original);
    _open(autoplay: false, at: _ctl?.value.position);
  }

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Container(
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: Sp.m),
        decoration: BoxDecoration(color: c.surface, border: Border(bottom: BorderSide(color: c.border))),
        child: Row(children: [
          Icon(Icons.check_circle_outline, size: 18, color: c.accentText),
          const SizedBox(width: Sp.s),
          Text(S.doneTitle, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
          const Spacer(),
          SegmentedButton<bool>(
            key: const ValueKey('result-source'),
            showSelectedIcon: false,
            style: const ButtonStyle(visualDensity: VisualDensity.compact),
            segments: const [
              ButtonSegment(value: false, label: Text(S.resultTab)),
              ButtonSegment(value: true, label: Text(S.originalTab)),
            ],
            selected: {_original},
            onSelectionChanged: (s) => _switch(s.first),
          ),
        ]),
      ),
      Expanded(child: ColoredBox(color: const Color(0xFF0E1013), child: _picture(context))),
      _Transport(ctl: _ctl, marks: st.leakMoments, onToggle: _togglePlay, onSeek: _seekTo),
    ]);
  }

  Widget _picture(BuildContext context) {
    final ctl = _ctl;
    if (_failed) {
      return Center(
        key: const ValueKey('result-cant-play'),
        child: Padding(
          padding: const EdgeInsets.all(Sp.xl),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.videocam_off_outlined, size: 36, color: Color(0xFFB4BAC3)),
            const SizedBox(height: Sp.m),
            const Text(S.resultCantPlay,
                textAlign: TextAlign.center, style: TextStyle(color: Color(0xFFE6E8EB), fontSize: 13)),
            const SizedBox(height: Sp.m),
            OutlinedButton.icon(
              onPressed: _original ? null : st.openOutput,
              icon: const Icon(Icons.open_in_new, size: 16),
              label: const Text(S.openExternal),
              style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFFE6E8EB),
                side: const BorderSide(color: Color(0xFF5A616B)),
              ),
            ),
          ]),
        ),
      );
    }
    if (ctl == null || !ctl.value.isInitialized) {
      return const Center(
        child: SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFFB4BAC3))),
      );
    }
    return GestureDetector(
      key: const ValueKey('result-player'),
      onTap: _togglePlay,
      child: Stack(fit: StackFit.expand, children: [
        Center(
          child: AspectRatio(
            aspectRatio: ctl.value.aspectRatio,
            child: st.resultSnapshot
                ? const ColoredBox(
                    key: ValueKey('result-snapshot'),
                    color: Color(0xFF2A2F36),
                    child: Center(child: Icon(Icons.movie_outlined, size: 40, color: Color(0xFF8A929D))))
                : VideoPlayer(ctl),
          ),
        ),
        if (_original)
          Positioned(
            left: Sp.m,
            top: Sp.m,
            child: DecoratedBox(
              decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.66), borderRadius: BorderRadius.circular(3)),
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Text(S.originalBadge, style: TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
              ),
            ),
          ),
        if (_opening)
          const Center(
            child: SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
          ),
      ]),
    );
  }
}

/// Play / pause, position (with the red leak moments on the track), time, sound.
class _Transport extends StatelessWidget {
  const _Transport({required this.ctl, required this.marks, required this.onToggle, required this.onSeek});
  final VideoPlayerController? ctl;
  final List<Risk> marks;
  final VoidCallback onToggle;
  final void Function(double seconds) onSeek;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final ctl = this.ctl;
    Widget bar(VideoPlayerValue? v) {
      final ready = v != null && v.isInitialized;
      final dur = ready ? v.duration.inMilliseconds / 1000 : 0.0;
      final pos = ready ? (v.position.inMilliseconds / 1000).clamp(0.0, dur) : 0.0;
      return Container(
        height: 48,
        padding: const EdgeInsets.symmetric(horizontal: Sp.s),
        decoration: BoxDecoration(color: c.surface, border: Border(top: BorderSide(color: c.border))),
        child: Row(children: [
          IconButton(
            key: const ValueKey('result-play'),
            tooltip: ready && v.isPlaying ? '${S.pauseTip}  (Space)' : '${S.playTip}  (Space)',
            onPressed: ready ? onToggle : null,
            icon: Icon(ready && v.isPlaying ? Icons.pause : Icons.play_arrow),
          ),
          SizedBox(
            width: 92,
            child: Text('${fmtDuration(pos)} / ${fmtDuration(dur)}',
                maxLines: 1,
                style: TextStyle(fontSize: 12, color: c.muted, fontFeatures: const [FontFeature.tabularFigures()])),
          ),
          Expanded(
            child: LayoutBuilder(builder: (context, cons) {
              return Stack(alignment: Alignment.center, children: [
                if (dur > 0)
                  for (final m in marks)
                    Positioned(
                      left: 24 + (cons.maxWidth - 48) * (m.startS / dur).clamp(0.0, 1.0),
                      width: ((cons.maxWidth - 48) * ((m.endS - m.startS) / dur)).clamp(3.0, cons.maxWidth),
                      top: cons.maxHeight / 2 + 6,
                      height: 3,
                      child: ColoredBox(color: c.riskSevere),
                    ),
                Slider(
                  key: const ValueKey('result-seek'),
                  value: dur > 0 ? pos : 0,
                  max: dur > 0 ? dur : 1,
                  onChanged: ready ? onSeek : null,
                ),
              ]);
            }),
          ),
          IconButton(
            tooltip: ready && v.volume == 0 ? S.unmute : S.mute,
            onPressed: ready ? () => ctl!.setVolume(v.volume == 0 ? 1 : 0) : null,
            icon: Icon(ready && v.volume == 0 ? Icons.volume_off_outlined : Icons.volume_up_outlined, size: 20),
          ),
        ]),
      );
    }

    if (ctl == null) return bar(null);
    return ValueListenableBuilder<VideoPlayerValue>(valueListenable: ctl, builder: (context, v, _) => bar(v));
  }
}

/// Right panel once the video is saved: where it went, what the final check found, what to do next.
class ResultPanel extends StatelessWidget {
  const ResultPanel({super.key, required this.st, required this.onNewVideo, this.width = 320});
  final AppState st;
  final VoidCallback onNewVideo;
  final double width;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final issues = st.qaHasIssues;
    final q = st.qa;
    final riskOnly = !issues && q != null && st.qaSummary != S.qaClean;
    final boxColor = issues ? c.dangerSoft : (riskOnly ? c.warnSoft : c.accentSoft);
    final fg = issues ? c.danger : (riskOnly ? c.warn : c.accentText);
    final out = st.outputPath ?? '';
    return Container(
      key: const ValueKey('export-done'),
      width: width,
      decoration: BoxDecoration(color: c.surface, border: Border(left: BorderSide(color: c.border))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Expanded(
          child: ListView(padding: const EdgeInsets.fromLTRB(Sp.m, Sp.s, Sp.m, Sp.m), children: [
            const SectionLabel(S.done),
            Text(p.basename(out),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
            const SizedBox(height: 2),
            Tooltip(
              message: out,
              child: Text(p.dirname(out),
                  maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: c.muted)),
            ),
            const SizedBox(height: Sp.s),
            Row(children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: st.openFolder,
                  icon: const Icon(Icons.folder_open_outlined, size: 16),
                  label: const Text(S.openFolder, maxLines: 1, overflow: TextOverflow.ellipsis),
                ),
              ),
              const SizedBox(width: Sp.s),
              Tooltip(
                message: S.openExternal,
                child: OutlinedButton(
                  key: const ValueKey('done-open-external'),
                  onPressed: st.openOutput,
                  style: OutlinedButton.styleFrom(minimumSize: const Size(40, 36), padding: EdgeInsets.zero),
                  child: const Icon(Icons.open_in_new, size: 16),
                ),
              ),
            ]),
            const SizedBox(height: Sp.l),
            const SectionLabel(S.finalCheck),
            Container(
              key: const ValueKey('qa-box'),
              padding: const EdgeInsets.all(Sp.m),
              decoration: BoxDecoration(
                color: boxColor,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: fg.withValues(alpha: 0.35)),
              ),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(issues || riskOnly ? Icons.warning_amber_rounded : Icons.verified_outlined, color: fg, size: 18),
                const SizedBox(width: Sp.s),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(st.qaSummary, style: TextStyle(color: fg, fontWeight: FontWeight.w600, fontSize: 13)),
                    if (issues) ...[
                      const SizedBox(height: Sp.s),
                      Text(S.leakWatchHint, style: TextStyle(color: c.muted, fontSize: 12)),
                      const SizedBox(height: Sp.s),
                      Wrap(spacing: Sp.s, runSpacing: Sp.s, children: [
                        for (final r in st.leakMoments)
                          ActionChip(
                            avatar: Icon(Icons.play_arrow, size: 16, color: c.danger),
                            label: Text(fmtDuration(r.startS)),
                            visualDensity: VisualDensity.compact,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(3)),
                            side: BorderSide(color: c.borderStrong),
                            onPressed: () => st.resultSeekTo?.call(r.startS),
                          ),
                      ]),
                    ],
                  ]),
                ),
              ]),
            ),
          ]),
        ),
        Container(
          padding: const EdgeInsets.all(Sp.m),
          decoration: BoxDecoration(border: Border(top: BorderSide(color: c.border))),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            if (issues) ...[
              FilledButton.icon(
                key: const ValueKey('done-fix'),
                onPressed: () => st.backToEdit(jumpToS: st.leakMoments.firstOrNull?.startS),
                icon: const Icon(Icons.edit_outlined, size: 18),
                label: const Text(S.fixInEditor),
                style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(40)),
              ),
              const SizedBox(height: Sp.s),
              OutlinedButton.icon(
                key: const ValueKey('done-new-video'),
                onPressed: onNewVideo,
                icon: const Icon(Icons.add, size: 18),
                label: const Text(S.newVideo),
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(40)),
              ),
            ] else ...[
              FilledButton.icon(
                key: const ValueKey('done-new-video'),
                onPressed: onNewVideo,
                icon: const Icon(Icons.add, size: 18),
                label: const Text(S.newVideo),
                style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(40)),
              ),
              const SizedBox(height: Sp.s),
              TextButton(
                key: const ValueKey('done-back'),
                onPressed: () => st.backToEdit(),
                child: const Text(S.backToEdit),
              ),
            ],
          ]),
        ),
      ]),
    );
  }
}
