import 'dart:io';

import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_state.dart';
import '../models.dart';
import '../pickers.dart';
import '../strings.dart';
import '../theme.dart';
import '../widgets/dialogs.dart';
import '../widgets/export_panel.dart';
import '../widgets/sidebar.dart';
import 'editor.dart';
import 'setup_wizard.dart';

/// The one window: menu bar + title strip, sidebar (recent projects), centre (welcome / start / progress /
/// editor), inspector, status bar. Export runs in a modal panel over the editor.
/// The Dosya menu, so the self-test can open it for a screenshot.
final fileMenuController = MenuController();

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.state, this.pickVideo = pickVideoFile});
  final AppState state;

  /// The video-only open dialog (tests pass a fake). null result = cancelled.
  final Future<String?> Function() pickVideo;
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  AppState get st => widget.state;
  bool _dragging = false;
  String? _shownToast;

  @override
  void initState() {
    super.initState();
    st.addListener(_onState);
  }

  @override
  void dispose() {
    st.removeListener(_onState);
    super.dispose();
  }

  void _onState() {
    final t = st.toast;
    if (t != null && t != _shownToast) {
      _shownToast = t;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(t),
          behavior: SnackBarBehavior.floating,
          width: 420,
          duration: const Duration(seconds: 3),
        ));
        st.toast = null;
        _shownToast = null;
      });
    }
  }

  bool get _canOpen => !st.busy && !st.deviceChecking;

  /// "Video aç…" (sidebar, Dosya menu, Ctrl+O / Ctrl+N, "Video seç"): the video-only open dialog.
  /// Cancel leaves everything as it was; the project being left keeps its place (and chosen people) in
  /// "Son projeler".
  Future<void> _pickVideo() async {
    if (!_canOpen) return;
    final path = await widget.pickVideo();
    if (path != null && _canOpen) await st.openPicked(path);
  }

  Future<void> _export() async {
    if (st.stage != Stage.editing || !st.canExport) return;
    final loc = await getSaveLocation(
      suggestedName: st.suggestedOutputName(),
      initialDirectory: st.suggestedOutputDir(),
      confirmButtonText: S.save,
      acceptedTypeGroups: const [XTypeGroup(label: 'MP4', extensions: ['mp4'])],
    );
    if (loc != null) await st.export(loc.path);
  }

  Future<void> _quit() async {
    await st.shutdown();
    exit(0);
  }

  Map<ShortcutActivator, VoidCallback> get _shortcuts => {
        const SingleActivator(LogicalKeyboardKey.space): st.togglePlay,
        const SingleActivator(LogicalKeyboardKey.arrowLeft): () => st.stepFrame(-1),
        const SingleActivator(LogicalKeyboardKey.arrowRight): () => st.stepFrame(1),
        const SingleActivator(LogicalKeyboardKey.home): () {
          if (st.stage == Stage.editing) st.setFrame(0, commit: true);
        },
        const SingleActivator(LogicalKeyboardKey.end): () {
          if (st.stage == Stage.editing) st.setFrame((st.video?.frameCount ?? 1) - 1, commit: true);
        },
        const SingleActivator(LogicalKeyboardKey.keyO, control: true): _pickVideo,
        const SingleActivator(LogicalKeyboardKey.keyN, control: true): _pickVideo, // old "Yeni video" key
        const SingleActivator(LogicalKeyboardKey.keyS, control: true): _export,
        const SingleActivator(LogicalKeyboardKey.keyB, control: true): st.settings.toggleSidebar,
      };

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([st, st.settings]),
      builder: (context, _) {
        if (st.wizardOpen) return SetupWizard(st: st);
        final c = context.pal;
        final editing = st.stage == Stage.editing || st.stage == Stage.exporting || st.stage == Stage.done;
        final dropEnabled = st.stage == Stage.start || st.stage == Stage.ready;
        Widget center = switch (st.stage) {
          Stage.start => _Welcome(st: st, dragging: _dragging, onPick: _pickVideo),
          Stage.ready => _scroll(_ReadyView(st: st, onPick: _pickVideo)),
          Stage.analyzing => _scroll(_AnalyzeView(st: st)),
          _ => EditorCenter(st: st),
        };
        center = DropTarget(
          enable: dropEnabled,
          onDragEntered: (_) => setState(() => _dragging = true),
          onDragExited: (_) => setState(() => _dragging = false),
          onDragDone: (d) {
            setState(() => _dragging = false);
            if (d.files.isEmpty) return;
            // first video among the dropped files; anything else is refused with a short message
            final paths = d.files.map((f) => f.path).toList();
            st.openPicked(paths.firstWhere(isVideoPath, orElse: () => paths.first));
          },
          child: ColoredBox(color: editing ? c.canvas : c.chrome, child: center),
        );
        return CallbackShortcuts(
          bindings: _shortcuts,
          child: Focus(
            autofocus: true,
            child: Scaffold(
              backgroundColor: c.chrome,
              body: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                _TopBar(
                  st: st,
                  onOpen: _pickVideo,
                  onExport: _export,
                  onQuit: _quit,
                ),
                if (st.error != null) _ErrorBanner(st: st),
                Expanded(
                  child: Stack(children: [
                    Positioned.fill(
                      child: LayoutBuilder(builder: (context, cons) {
                        final inspectorW = cons.maxWidth >= 1500 ? 344.0 : 312.0;
                        return Row(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                          Sidebar(st: st, onOpenVideo: _pickVideo),
                          Expanded(child: center),
                          if (editing) Inspector(st: st, onExport: _export, width: inspectorW),
                        ]);
                      }),
                    ),
                    if (st.stage == Stage.exporting || st.stage == Stage.done)
                      Positioned.fill(child: ExportOverlay(st: st)),
                  ]),
                ),
                _StatusBar(st: st),
              ]),
            ),
          ),
        );
      },
    );
  }

  Widget _scroll(Widget child) => Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(Sp.xl),
          child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 560), child: child),
        ),
      );
}

// ------------------------------------------------------------------ top bar: menus, file, steps, device
class _TopBar extends StatelessWidget {
  const _TopBar({required this.st, required this.onOpen, required this.onExport, required this.onQuit});
  final AppState st;
  final VoidCallback onOpen, onExport, onQuit;

  Widget _check(bool on) => SizedBox(width: 18, child: on ? const Icon(Icons.check, size: 16) : null);

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final s = st.settings;
    final v = st.video;
    final canOpen = !st.busy && !st.deviceChecking;
    final menus = MenuBar(children: [
      SubmenuButton(
        controller: fileMenuController,
        menuChildren: [
          MenuItemButton(
            key: const ValueKey('menu-open-video'),
            leadingIcon: const Icon(Icons.folder_open_outlined),
            shortcut: const SingleActivator(LogicalKeyboardKey.keyO, control: true),
            onPressed: canOpen ? onOpen : null,
            child: const Text(S.openVideoMenu),
          ),
          SubmenuButton(
            leadingIcon: const Icon(Icons.history),
            menuChildren: s.recents.isEmpty
                ? [const MenuItemButton(child: Text(S.recentEmpty))]
                : [
                    for (final r in s.recents.take(8))
                      MenuItemButton(
                        onPressed: canOpen ? () => st.openRecent(r) : null,
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 320),
                          child: Text(r.name, overflow: TextOverflow.ellipsis),
                        ),
                      ),
                  ],
            child: const Text(S.recentMenu),
          ),
          const Divider(height: 9),
          MenuItemButton(
            leadingIcon: const Icon(Icons.save_alt),
            shortcut: const SingleActivator(LogicalKeyboardKey.keyS, control: true),
            onPressed: st.stage == Stage.editing && st.canExport ? onExport : null,
            child: const Text(S.exportMenu),
          ),
          const Divider(height: 9),
          MenuItemButton(
            leadingIcon: const Icon(Icons.logout),
            shortcut: const SingleActivator(LogicalKeyboardKey.f4, alt: true),
            onPressed: onQuit,
            child: const Text(S.quit),
          ),
        ],
        child: const Text(S.menuFile),
      ),
      SubmenuButton(
        menuChildren: [
          MenuItemButton(
            leadingIcon: _check(s.sidebarOpen),
            shortcut: const SingleActivator(LogicalKeyboardKey.keyB, control: true),
            onPressed: s.toggleSidebar,
            child: const Text(S.sidebar),
          ),
          SubmenuButton(
            leadingIcon: const Icon(Icons.contrast),
            menuChildren: [
              for (final (m, t) in const [
                (ThemeMode.system, S.themeSystem),
                (ThemeMode.light, S.themeLight),
                (ThemeMode.dark, S.themeDark),
              ])
                MenuItemButton(
                  leadingIcon: _check(s.themeMode == m),
                  onPressed: () => s.setThemeMode(m),
                  child: Text(t),
                ),
            ],
            child: const Text(S.theme),
          ),
        ],
        child: const Text(S.menuView),
      ),
      SubmenuButton(
        menuChildren: [
          MenuItemButton(
            leadingIcon: const Icon(Icons.keyboard_outlined),
            onPressed: () => showShortcuts(context),
            child: const Text(S.shortcuts),
          ),
          MenuItemButton(
            key: const ValueKey('menu-rerun-check'),
            leadingIcon: const Icon(Icons.fact_check_outlined),
            onPressed: st.busy || st.check.running ? null : st.rerunSystemCheck,
            child: const Text(S.rerunCheck),
          ),
          const Divider(height: 9),
          MenuItemButton(
            leadingIcon: const Icon(Icons.info_outline),
            onPressed: () => showAbout(context, st),
            child: const Text(S.about),
          ),
        ],
        child: const Text(S.menuHelp),
      ),
    ]);
    return Container(
      height: 40,
      padding: const EdgeInsets.only(left: Sp.m, right: Sp.m),
      decoration: BoxDecoration(color: c.chrome, border: Border(bottom: BorderSide(color: c.border))),
      child: Row(children: [
        const AppMark(size: 20),
        const SizedBox(width: Sp.s),
        menus,
        const SizedBox(width: Sp.s),
        Container(width: 1, height: 20, color: c.border),
        const SizedBox(width: Sp.m),
        Expanded(
          child: v == null || st.stage == Stage.start
              ? Text(S.appName, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.muted))
              : Row(children: [
                  Icon(Icons.movie_outlined, size: 16, color: c.muted),
                  const SizedBox(width: Sp.s - 2),
                  Flexible(
                    child: Tooltip(
                      message: v.path,
                      child: Text(v.name,
                          key: const ValueKey('title-file'),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
                    ),
                  ),
                  const SizedBox(width: Sp.s),
                  Flexible(
                    child: Text(fmtDuration(v.durationS),
                        maxLines: 1, overflow: TextOverflow.clip, softWrap: false,
                        style: TextStyle(fontSize: 12, color: c.muted)),
                  ),
                ]),
        ),
        const SizedBox(width: Sp.m),
        _Steps(current: switch (st.stage) {
          Stage.start || Stage.ready || Stage.analyzing => 1,
          Stage.editing => 2,
          Stage.exporting || Stage.done => 3,
        }),
        const SizedBox(width: Sp.m),
        // a long adapter name shrinks (ellipsis, full text in the tooltip) instead of pushing the bar off screen
        // at 1024 px (0.1.6 overflowed by 77 px with a long GPU name)
        Flexible(child: _DeviceChip(st: st)),
      ]),
    );
  }
}

/// Where the user is: 1 Video -> 2 Kişiler -> 3 Kaydet.
class _Steps extends StatelessWidget {
  const _Steps({required this.current});
  final int current;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    const labels = [S.step1, S.step2, S.step3];
    final items = <Widget>[];
    for (var i = 0; i < labels.length; i++) {
      final n = i + 1;
      final active = n == current, done = n < current;
      if (i > 0) {
        items.add(Container(width: 16, height: 1, margin: const EdgeInsets.symmetric(horizontal: 6), color: c.borderStrong));
      }
      items.add(Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 18,
          height: 18,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: active || done ? c.accent : Colors.transparent,
            shape: BoxShape.circle,
            border: Border.all(color: active || done ? c.accent : c.borderStrong),
          ),
          child: done
              ? Icon(Icons.check, size: 12, color: c.onAccent)
              : Text('$n',
                  style: TextStyle(
                      fontSize: 10.5, fontWeight: FontWeight.w700, color: active ? c.onAccent : c.muted, height: 1.1)),
        ),
        const SizedBox(width: 6),
        Text(labels[i],
            style: TextStyle(
                fontSize: 12.5, fontWeight: active ? FontWeight.w600 : FontWeight.w400, color: active ? c.text : c.muted)),
      ]));
    }
    return Row(mainAxisSize: MainAxisSize.min, children: items);
  }
}

class _DeviceChip extends StatelessWidget {
  const _DeviceChip({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final d = st.device;
    final gpu = !st.deviceChecking && d != null && d.gpu && st.proc != Proc.cpu;
    final String label, value;
    if (st.deviceChecking) {
      label = '';
      value = S.deviceChecking;
    } else if (gpu) {
      label = S.procGpu;
      // "Radeon (TM) Pro Graphics" -> "Radeon Pro Graphics"; the tooltip keeps the full text
      value = (d.deviceName ?? S.deviceGpuUnknown).replaceAll(RegExp(r'\s*\((TM|R|C)\)', caseSensitive: false), '');
    } else {
      label = '';
      value = st.deviceText;
    }
    return Tooltip(
      message: st.deviceText,
      child: Container(
        key: const ValueKey('device-chip'),
        height: 26,
        constraints: const BoxConstraints(maxWidth: 340),
        padding: const EdgeInsets.symmetric(horizontal: Sp.s),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(color: c.border),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          if (st.deviceChecking)
            SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: c.muted))
          else
            Icon(gpu ? Icons.developer_board : Icons.memory, size: 15, color: gpu ? c.accentText : c.muted),
          const SizedBox(width: 6),
          // label + adapter name in one text, so a narrow window shortens it with an ellipsis instead of overflowing
          Flexible(
            child: Text.rich(
              TextSpan(children: [
                if (label.isNotEmpty)
                  TextSpan(text: '$label  ', style: TextStyle(fontWeight: FontWeight.w600, color: c.text)),
                TextSpan(text: value),
              ]),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 12, color: c.muted),
            ),
          ),
        ]),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final detail = st.errorDetail ?? '';
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(color: c.dangerSoft, border: Border(bottom: BorderSide(color: c.danger.withValues(alpha: 0.4)))),
      padding: const EdgeInsets.fromLTRB(Sp.l, Sp.s, Sp.s, Sp.s),
      child: Row(children: [
        Icon(Icons.error_outline, color: c.danger, size: 18),
        const SizedBox(width: Sp.s),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(st.error ?? '', style: TextStyle(color: c.danger, fontWeight: FontWeight.w600, fontSize: 13)),
            if (detail.isNotEmpty)
              Text('${S.errDetails}: $detail',
                  maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(color: c.muted, fontSize: 12)),
          ]),
        ),
        IconButton(onPressed: st.dismissError, icon: const Icon(Icons.close, size: 16), tooltip: S.close),
      ]),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final v = st.video;
    final pct = st.progress >= 0 ? ' · ${(st.progress * 100).round()}%' : '';
    final text = switch (st.stage) {
      Stage.start => S.statusReady,
      Stage.ready => S.statusReady,
      Stage.analyzing => '${st.phase}$pct',
      Stage.editing => st.selected.isEmpty
          ? '${S.peopleCount(st.persons.length)} · ${S.statusPickPeople}'
          : '${S.peopleCount(st.persons.length)} · ${S.selectedCount(st.selected.length)}',
      Stage.exporting => '${S.exporting} · ${st.phase}$pct',
      Stage.done => S.done,
    };
    final style = TextStyle(fontSize: 11.5, color: c.muted);
    return Container(
      height: 24,
      padding: const EdgeInsets.symmetric(horizontal: Sp.m),
      decoration: BoxDecoration(color: c.chrome, border: Border(top: BorderSide(color: c.border))),
      child: Row(children: [
        Expanded(child: Text(text, key: const ValueKey('status-text'), maxLines: 1, overflow: TextOverflow.ellipsis, style: style)),
        if (v != null && st.stage != Stage.start) ...[
          Text('${v.width}×${v.height} · ${v.fps.toStringAsFixed(v.fps.truncateToDouble() == v.fps ? 0 : 2)} fps · ${fmtDuration(v.durationS)}',
              style: style),
          const SizedBox(width: Sp.l),
        ],
        Icon(Icons.lock_outline, size: 12, color: c.faint),
        const SizedBox(width: Sp.xs),
        Text(S.privacy, style: style),
      ]),
    );
  }
}

// ------------------------------------------------------------------ welcome (no video yet)
class _Welcome extends StatelessWidget {
  const _Welcome({required this.st, required this.dragging, required this.onPick});
  final AppState st;
  final bool dragging;
  final VoidCallback onPick;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final recents = st.settings.recents.take(4).toList();
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(Sp.xl),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 600),
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            AnimatedContainer(
              key: const ValueKey('drop-zone'),
              duration: const Duration(milliseconds: 120),
              padding: const EdgeInsets.symmetric(vertical: 40, horizontal: Sp.xl),
              decoration: BoxDecoration(
                color: dragging ? c.accentSoft : c.surface,
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: dragging ? c.accent : c.borderStrong, width: dragging ? 2 : 1),
              ),
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.video_file_outlined, size: 40, color: dragging ? c.accentText : c.muted),
                const SizedBox(height: Sp.l),
                Text(dragging ? S.dropRelease : S.dropTitle,
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: c.text)),
                const SizedBox(height: Sp.s),
                Text(S.dropOr, style: TextStyle(fontSize: 13, color: c.muted)),
                const SizedBox(height: Sp.m),
                Tooltip(
                  message: 'Ctrl+O',
                  child: FilledButton.icon(
                    key: const ValueKey('pick-video'),
                    onPressed: st.deviceChecking ? null : onPick,
                    icon: const Icon(Icons.folder_open_outlined, size: 18),
                    label: const Text(S.pickVideo),
                    style: FilledButton.styleFrom(minimumSize: const Size(160, 38)),
                  ),
                ),
                const SizedBox(height: Sp.l + Sp.xs),
                Text(S.dropFoot, textAlign: TextAlign.center, style: TextStyle(fontSize: 12, color: c.faint)),
              ]),
            ),
            if (recents.isNotEmpty) ...[
              const SizedBox(height: Sp.xl),
              const SectionLabel(S.recent),
              Container(
                decoration: BoxDecoration(
                  color: c.surface,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: c.border),
                ),
                clipBehavior: Clip.antiAlias,
                child: Column(children: [
                  for (var i = 0; i < recents.length; i++) ...[
                    if (i > 0) Divider(height: 1, color: c.border),
                    RecentRow(
                      r: recents[i],
                      dense: false,
                      onOpen: st.deviceChecking ? null : () => st.openRecent(recents[i]),
                    ),
                  ],
                ]),
              ),
            ],
          ]),
        ),
      ),
    );
  }
}

// ------------------------------------------------------------------ video chosen: Başlat
class _ReadyView extends StatelessWidget {
  const _ReadyView({required this.st, required this.onPick});
  final AppState st;
  final VoidCallback onPick;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final v = st.video!;
    return Container(
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: c.border),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, mainAxisSize: MainAxisSize.min, children: [
        Padding(
          padding: const EdgeInsets.all(Sp.l),
          child: Row(children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: c.surfaceAlt,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: c.border),
              ),
              child: Icon(Icons.movie_outlined, color: c.muted, size: 22),
            ),
            const SizedBox(width: Sp.m),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(v.name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: c.text)),
                const SizedBox(height: 2),
                Text('${S.duration(fmtDuration(v.durationS))}  ·  ${v.width}×${v.height}',
                    style: TextStyle(fontSize: 12.5, color: c.muted)),
              ]),
            ),
            TextButton(onPressed: onPick, child: const Text(S.otherVideo)),
          ]),
        ),
        Divider(height: 1, color: c.border),
        Padding(
          padding: const EdgeInsets.all(Sp.l),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            FilledButton.icon(
              key: const ValueKey('analyze'),
              onPressed: st.deviceChecking ? null : st.analyze,
              icon: st.deviceChecking
                  ? SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: c.faint))
                  : const Icon(Icons.play_arrow, size: 20),
              label: const Text(S.analyze),
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(40)),
            ),
            const SizedBox(height: Sp.s),
            Text(st.deviceChecking ? S.analyzeWait : S.analyzeHint,
                textAlign: TextAlign.center, style: TextStyle(fontSize: 12.5, color: c.muted)),
          ]),
        ),
        Divider(height: 1, color: c.border),
        AdvancedSection(st: st),
      ]),
    );
  }
}

// ------------------------------------------------------------------ face search progress
class _AnalyzeView extends StatelessWidget {
  const _AnalyzeView({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final pct = st.progress >= 0 ? '${(st.progress * 100).round()}%' : '';
    return Container(
      key: const ValueKey('analyze-progress'),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: c.border),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, mainAxisSize: MainAxisSize.min, children: [
        Container(
          height: 44,
          padding: const EdgeInsets.symmetric(horizontal: Sp.l),
          decoration: BoxDecoration(border: Border(bottom: BorderSide(color: c.border))),
          child: Row(children: [
            Icon(Icons.face_retouching_natural, size: 18, color: c.accentText),
            const SizedBox(width: Sp.s),
            Text(S.analyzing, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text)),
          ]),
        ),
        Padding(
          padding: const EdgeInsets.all(Sp.l),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Text(st.video?.name ?? '',
                maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12.5, color: c.muted)),
            const SizedBox(height: Sp.m),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(value: st.progress >= 0 ? st.progress : null, minHeight: 6),
            ),
            const SizedBox(height: Sp.s),
            Row(children: [
              Expanded(
                child: Text(st.phase, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
              ),
              Text(pct,
                  style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: c.text,
                      fontFeatures: const [FontFeature.tabularFigures()])),
            ]),
            const SizedBox(height: 2),
            Text(st.etaText.isEmpty ? ' ' : st.etaText, style: TextStyle(fontSize: 12, color: c.muted)),
            const SizedBox(height: Sp.l),
            Align(
              alignment: Alignment.centerRight,
              child: OutlinedButton(onPressed: st.cancelJob, child: const Text(S.cancel)),
            ),
          ]),
        ),
      ]),
    );
  }
}
