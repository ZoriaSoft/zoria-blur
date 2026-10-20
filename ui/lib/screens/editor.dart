import 'package:flutter/material.dart';

import '../app_state.dart';
import '../live_player.dart';
import '../models.dart';
import '../strings.dart';
import '../theme.dart';
import '../widgets/dialogs.dart';
import '../widgets/person_card.dart';
import '../widgets/preview_pane.dart';
import '../widgets/timeline.dart';

/// Centre of the editor: preview toolbar, the preview with its transport bar, and the timeline.
class EditorCenter extends StatelessWidget {
  const EditorCenter({super.key, required this.st});
  final AppState st;

  Future<void> _boxDrawn(BuildContext context, Rect r) async {
    final pid = await askWhoseFace(context, st);
    if (pid != null) await st.addBox(pid, [r.left, r.top, r.right, r.bottom]);
  }

  Widget? _live(BuildContext context, VideoInfo v) {
    final pl = st.player;
    if (!pl.playing) return null;
    final img = pl.image;
    return Stack(fit: StackFit.expand, children: [
      if (img != null)
        CustomPaint(
          painter: LiveFramePainter(
            image: img,
            regions: st.liveRegionsAt(pl.frame),
            videoW: v.width,
            videoH: v.height,
            strength: st.strength,
          ),
        ),
      if (pl.loading || st.livePlanPending)
        Positioned(
          left: 12,
          right: 12,
          bottom: 12,
          child: IgnorePointer(
            child: Center(
              child: DecoratedBox(
                decoration:
                    BoxDecoration(color: Colors.black.withValues(alpha: 0.66), borderRadius: BorderRadius.circular(4)),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  child: Text(pl.loading ? S.liveLoading : S.livePending,
                      style: const TextStyle(color: Colors.white, fontSize: 12.5)),
                ),
              ),
            ),
          ),
        ),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    final v = st.video!;
    final c = context.pal;
    return ListenableBuilder(
      listenable: st.player,
      builder: (context, _) {
        final playing = st.player.playing;
        final frame = playing ? st.player.frame : st.frame;
        final blur = st.mode == BlurMode.selected
            ? st.selected
            : st.persons.map((x) => x.id).where((id) => !st.selected.contains(id)).toSet();
        final lanes = [
          for (final p in st.persons)
            TimelineLane(
              id: p.id,
              label: p.displayName,
              thumbnail: p.thumbnails.firstOrNull,
              segments: st.presence[p.id] ?? const [],
              willBlur: blur.contains(p.id),
              selected: st.selected.contains(p.id),
            ),
        ];
        return LayoutBuilder(builder: (context, cons) {
          // keep the picture big on small windows: fewer lane rows when height is short
          final maxLanes = cons.maxHeight < 560 ? 2 : (cons.maxHeight < 700 ? 3 : 4);
          return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Container(
              height: 40,
              padding: const EdgeInsets.symmetric(horizontal: Sp.m),
              decoration: BoxDecoration(color: c.surface, border: Border(bottom: BorderSide(color: c.border))),
              child: Row(children: [
                Text(S.preview, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
                const SizedBox(width: Sp.m),
                Flexible(
                  child: Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: playing ? c.accentSoft : c.surfaceAlt,
                    borderRadius: BorderRadius.circular(3),
                    border: Border.all(color: playing ? c.accent.withValues(alpha: 0.5) : c.border),
                  ),
                  child: Text(playing ? S.previewLive : S.previewExact,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 11.5, color: playing ? c.accentText : c.muted)),
                  ),
                  ),
                ),
                const SizedBox(width: Sp.m),
                // the action gets most of the room and keeps a one-line label; the note on its left gives way first
                // (0.1.6 split the room 2:3 and wrapped the label to two lines at 1024 px)
                Flexible(
                  flex: 3,
                  child: Align(
                    alignment: Alignment.centerRight,
                    child: OutlinedButton.icon(
                      key: const ValueKey('mark-face'),
                      onPressed: () => st.setDrawMode(!st.drawMode),
                      icon: Icon(st.drawMode ? Icons.check : Icons.crop_free, size: 16),
                      label: Text(st.drawMode ? S.markFaceStop : S.markFace,
                          maxLines: 1, softWrap: false, overflow: TextOverflow.ellipsis),
                      style: st.drawMode
                          ? OutlinedButton.styleFrom(
                              backgroundColor: c.accentSoft, foregroundColor: c.accentText, side: BorderSide(color: c.accent))
                          : null,
                    ),
                  ),
                ),
              ]),
            ),
            Expanded(
              child: PreviewPane(
                image: st.previewBytes,
                videoWidth: v.width,
                videoHeight: v.height,
                fps: v.fps,
                frameCount: v.frameCount,
                frame: frame,
                onFrameChanged: (f) => st.setFrame(f),
                onFrameChangeEnd: (f) => st.setFrame(f, commit: true),
                busy: !playing && st.previewBusy,
                drawMode: st.drawMode,
                onBoxDrawn: (r) => _boxDrawn(context, r),
                live: _live(context, v),
                playing: playing,
                onPlayPause: playing ? st.pause : st.play,
              ),
            ),
            Timeline(
              fps: v.fps,
              frameCount: v.frameCount,
              frame: frame,
              onFrameChanged: (f) => st.setFrame(f),
              onFrameChangeEnd: (f) => st.setFrame(f, commit: true),
              marks: [
                for (final r in st.riskMoments)
                  RiskMark(startS: r.startS, endS: r.endS, label: r.label, severe: r.severe),
              ],
              lanes: lanes,
              maxVisibleLanes: maxLanes,
            ),
          ]);
        });
      },
    );
  }
}

/// Right inspector: what to do, strength, people (select / rename / merge / split), Gelişmiş, and the
/// one primary action anchored at the bottom.
class Inspector extends StatelessWidget {
  const Inspector({super.key, required this.st, required this.onExport, this.width = 320});
  final AppState st;
  final VoidCallback onExport;
  final double width;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final sel = st.selected;
    final single = sel.length == 1 ? st.persons.where((p) => p.id == sel.first).firstOrNull : null;
    final canSplit = single != null && single.trackIds.length >= 2;
    final hint = !st.canExport
        ? S.needSelection
        : st.mode == BlurMode.except && st.selected.isEmpty
            ? S.exceptNone
            : '';
    return Container(
      width: width,
      decoration: BoxDecoration(color: c.surface, border: Border(left: BorderSide(color: c.border))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(Sp.m, Sp.s, Sp.m, 0),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            const SectionLabel(S.whatToDo),
            _Radio(
              key: const ValueKey('mode-selected'),
              selected: st.mode == BlurMode.selected,
              text: S.modeSelected,
              onTap: () => st.setMode(BlurMode.selected),
            ),
            _Radio(
              key: const ValueKey('mode-except'),
              selected: st.mode == BlurMode.except,
              text: S.modeExcept,
              onTap: () => st.setMode(BlurMode.except),
            ),
            const SizedBox(height: Sp.s),
            const SectionLabel(S.strength),
            Row(children: [
              Text(S.strengthLow, style: TextStyle(fontSize: 11.5, color: c.faint)),
              Expanded(
                child: Slider(
                  value: st.strength,
                  min: 1.0,
                  max: 3.0,
                  onChanged: (v) => st.setStrength(v),
                  onChangeEnd: (v) => st.setStrength(v, commit: true),
                ),
              ),
              Text(S.strengthHigh, style: TextStyle(fontSize: 11.5, color: c.faint)),
            ]),
          ]),
        ),
        Divider(height: 1, color: c.border),
        Padding(
          padding: const EdgeInsets.fromLTRB(Sp.m, Sp.s, Sp.s, 0),
          child: SectionLabel(
            sel.isEmpty ? '${S.people} · ${S.peopleCount(st.persons.length)}' : '${S.people} · ${S.selectedCount(sel.length)}',
            trailing: TextButton(
              onPressed: st.persons.isEmpty ? null : () => st.selectAll(sel.isEmpty),
              child: Text(sel.isEmpty ? S.selectAll : S.selectNone),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(Sp.m, 0, Sp.m, Sp.s),
          child: Text(st.mode == BlurMode.except ? S.selectHintExcept : S.selectHint,
              style: TextStyle(fontSize: 12, color: c.muted)),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(Sp.m, 0, Sp.m, Sp.s),
          child: Row(children: [
            Tooltip(
              message: S.mergeHelp,
              child: OutlinedButton.icon(
                key: const ValueKey('merge'),
                onPressed: sel.length >= 2 ? st.mergeSelected : null,
                icon: const Icon(Icons.call_merge, size: 16),
                label: const Text(S.merge),
              ),
            ),
            const SizedBox(width: Sp.s),
            Tooltip(
              message: S.splitTip,
              child: OutlinedButton.icon(
                key: const ValueKey('split'),
                onPressed: canSplit ? () => splitPerson(context, st, single) : null,
                icon: const Icon(Icons.call_split, size: 16),
                label: const Text(S.split),
              ),
            ),
          ]),
        ),
        Expanded(
          child: LayoutBuilder(builder: (context, cons) {
            final cols = ((cons.maxWidth - Sp.m * 2 + Sp.s) / 100).floor().clamp(2, 4);
            return GridView.builder(
              padding: const EdgeInsets.fromLTRB(Sp.m, 0, Sp.m, Sp.m),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: cols,
                mainAxisSpacing: Sp.s,
                crossAxisSpacing: Sp.s,
                // short windows: squarer cards so a full row of faces stays in view
                childAspectRatio: cons.maxHeight < 400 ? 0.9 : 0.76,
              ),
              itemCount: st.persons.length,
              itemBuilder: (context, i) {
                final p = st.persons[i];
                return PersonCard(
                  key: ValueKey('person-${p.id}'),
                  person: p,
                  selected: sel.contains(p.id),
                  willBlur: st.mode == BlurMode.selected ? sel.contains(p.id) : !sel.contains(p.id),
                  onTap: () => st.toggle(p.id),
                  onRename: () => renamePerson(context, st, p),
                  onShow: p.bestFrame == null ? null : () => st.setFrame(p.bestFrame!, commit: true),
                );
              },
            );
          }),
        ),
        Divider(height: 1, color: c.border),
        AdvancedSection(st: st, showDetMode: false),
        Container(
          padding: const EdgeInsets.all(Sp.m),
          decoration: BoxDecoration(color: c.surfaceAlt, border: Border(top: BorderSide(color: c.border))),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            if (hint.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: Sp.s),
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Icon(st.canExport ? Icons.info_outline : Icons.touch_app_outlined, size: 16, color: c.muted),
                  const SizedBox(width: Sp.s - 2),
                  Expanded(child: Text(hint, style: TextStyle(fontSize: 12, color: c.muted))),
                ]),
              ),
            Tooltip(
              message: 'Ctrl+S',
              child: FilledButton.icon(
                key: const ValueKey('export'),
                onPressed: st.canExport ? onExport : null,
                icon: const Icon(Icons.save_alt, size: 18),
                label: const Text(S.export),
                style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(40)),
              ),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _Radio extends StatelessWidget {
  const _Radio({super.key, required this.selected, required this.text, required this.onTap});
  final bool selected;
  final String text;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return InkWell(
      borderRadius: BorderRadius.circular(4),
      onTap: onTap,
      child: Container(
        constraints: const BoxConstraints(minHeight: 32),
        padding: const EdgeInsets.symmetric(horizontal: Sp.xs, vertical: Sp.xs),
        child: Row(children: [
          Icon(selected ? Icons.radio_button_checked : Icons.radio_button_unchecked,
              size: 18, color: selected ? c.accent : c.faint),
          const SizedBox(width: Sp.s),
          Expanded(
            child: Text(text,
                style: TextStyle(
                    fontSize: 13, color: c.text, fontWeight: selected ? FontWeight.w600 : FontWeight.w400)),
          ),
        ]),
      ),
    );
  }
}

/// "Gelişmiş": collapsed by default. Face search mode (before the search) and hardware.
class AdvancedSection extends StatelessWidget {
  const AdvancedSection({super.key, required this.st, this.showDetMode = true});
  final AppState st;
  final bool showDetMode;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final help = switch (st.detMode) {
      'fast' => S.detFastHelp,
      'thorough' => S.detThoroughHelp,
      _ => S.detBalancedHelp,
    };
    return Material(
      type: MaterialType.transparency, // ExpansionTile's ListTile paints on the nearest Material
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          key: const PageStorageKey('advanced'),
          dense: true,
          tilePadding: const EdgeInsets.symmetric(horizontal: Sp.m),
          childrenPadding: const EdgeInsets.fromLTRB(Sp.m, 0, Sp.m, Sp.m),
          visualDensity: VisualDensity.compact,
          title: Text(S.advanced, style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600, color: c.text)),
          children: [
            if (showDetMode) ...[
              _Labeled(
                label: S.detMode,
                help: help,
                child: SegmentedButton<String>(
                  showSelectedIcon: false,
                  segments: const [
                    ButtonSegment(value: 'fast', label: Text(S.detFast)),
                    ButtonSegment(value: 'balanced', label: Text(S.detBalanced)),
                    ButtonSegment(value: 'thorough', label: Text(S.detThorough)),
                  ],
                  selected: {st.detMode},
                  onSelectionChanged: (s) => st.setDetMode(s.first),
                ),
              ),
              const SizedBox(height: Sp.m),
            ],
            _Labeled(
              label: S.processor,
              help: st.deviceText,
              child: SegmentedButton<Proc>(
                showSelectedIcon: false,
                segments: const [
                  ButtonSegment(value: Proc.auto, label: Text(S.procAuto)),
                  ButtonSegment(value: Proc.cpu, label: Text(S.procCpu)),
                  ButtonSegment(value: Proc.gpu, label: Text(S.procGpu)),
                ],
                selected: {st.proc},
                onSelectionChanged: (s) => st.setProc(s.first),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Labeled extends StatelessWidget {
  const _Labeled({required this.label, required this.child, this.help});
  final String label;
  final Widget child;
  final String? help;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: TextStyle(fontSize: 12, color: c.muted, fontWeight: FontWeight.w600)),
      const SizedBox(height: Sp.xs),
      SizedBox(width: double.infinity, child: child),
      if (help != null && help!.isNotEmpty) ...[
        const SizedBox(height: Sp.xs),
        Text(help!, style: TextStyle(fontSize: 12, color: c.muted)),
      ],
    ]);
  }
}
