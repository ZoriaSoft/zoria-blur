import 'dart:io';

import 'package:flutter/material.dart';

import '../app_state.dart';
import '../models.dart';
import '../settings.dart';
import '../strings.dart';
import '../theme.dart';

/// Left sidebar: "Video aç…" and the recent projects. Collapses to a thin icon strip (Ctrl+B).
class Sidebar extends StatelessWidget {
  const Sidebar({super.key, required this.st, required this.onOpenVideo});
  final AppState st;
  final VoidCallback onOpenVideo;

  static const openWidth = 240.0;
  static const closedWidth = 44.0;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final s = st.settings;
    final open = s.sidebarOpen;
    final enabled = !st.busy && !st.deviceChecking;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 140),
      curve: Curves.easeOut,
      width: open ? openWidth : closedWidth,
      decoration: BoxDecoration(color: c.surfaceAlt, border: Border(right: BorderSide(color: c.border))),
      child: ClipRect(
        child: OverflowBox(
          alignment: Alignment.topLeft,
          minWidth: 0,
          maxWidth: open ? openWidth : closedWidth,
          child: open ? _open(context, c, s, enabled) : _closed(context, c, enabled),
        ),
      ),
    );
  }

  Widget _closed(BuildContext context, Pal c, bool enabled) => Column(children: [
        const SizedBox(height: Sp.xs),
        IconButton(
          key: const ValueKey('sidebar-expand'),
          tooltip: '${S.expand}  (Ctrl+B)',
          onPressed: st.settings.toggleSidebar,
          icon: const Icon(Icons.chevron_right),
        ),
        IconButton(
          key: const ValueKey('sidebar-open-strip'),
          tooltip: '${S.openVideoMenu}  (Ctrl+O)',
          onPressed: enabled ? onOpenVideo : null,
          icon: const Icon(Icons.folder_open_outlined),
        ),
        const SizedBox(height: Sp.xs),
        Container(width: 24, height: 1, color: c.border),
        const SizedBox(height: Sp.xs),
        for (final r in st.settings.recents.take(8))
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 2),
            child: Tooltip(
              message: r.name,
              child: InkWell(
                borderRadius: BorderRadius.circular(3),
                onTap: enabled ? () => st.openRecent(r) : null,
                child: Container(
                  width: 32,
                  height: 24,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(3),
                    border: Border.all(color: st.isCurrent(r) ? c.accent : c.border, width: st.isCurrent(r) ? 1.5 : 1),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: _Thumb(r: r, iconSize: 14),
                ),
              ),
            ),
          ),
      ]);

  Widget _open(BuildContext context, Pal c, AppSettings s, bool enabled) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(Sp.m, Sp.s, Sp.xs, Sp.xs),
            child: Row(children: [
              Expanded(
                child: Text(S.recent, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: c.muted)),
              ),
              IconButton(
                key: const ValueKey('sidebar-collapse'),
                tooltip: '${S.collapse}  (Ctrl+B)',
                onPressed: s.toggleSidebar,
                icon: const Icon(Icons.chevron_left),
              ),
            ]),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(Sp.m, 0, Sp.m, Sp.s),
            child: OutlinedButton.icon(
              key: const ValueKey('sidebar-open'),
              onPressed: enabled ? onOpenVideo : null,
              icon: const Icon(Icons.folder_open_outlined, size: 18),
              label: const Text(S.openVideoMenu),
              style: OutlinedButton.styleFrom(
                alignment: Alignment.centerLeft,
                backgroundColor: c.surface,
              ),
            ),
          ),
          Divider(height: 1, color: c.border),
          Expanded(
            child: s.recents.isEmpty
                ? Padding(
                    padding: const EdgeInsets.all(Sp.m),
                    child: Text(S.recentNone, style: TextStyle(fontSize: 12, color: c.faint)),
                  )
                : ListView(
                    padding: const EdgeInsets.symmetric(vertical: Sp.xs),
                    children: [
                      for (final r in s.recents)
                        RecentRow(
                          r: r,
                          current: st.isCurrent(r),
                          onOpen: enabled ? () => st.openRecent(r) : null,
                          onRemove: () => s.removeRecent(r),
                        ),
                    ],
                  ),
          ),
        ],
      );
}

/// One recent project: thumbnail, name, when, length.
class RecentRow extends StatefulWidget {
  const RecentRow({super.key, required this.r, this.current = false, this.onOpen, this.onRemove, this.dense = true});
  final RecentProject r;
  final bool current;
  final VoidCallback? onOpen;
  final VoidCallback? onRemove;
  final bool dense;
  @override
  State<RecentRow> createState() => _RecentRowState();
}

class _RecentRowState extends State<RecentRow> {
  bool _hover = false;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final r = widget.r;
    final meta = [
      S.when(r.opened, DateTime.now()),
      if (r.durationS > 0) fmtDuration(r.durationS),
      if (r.persons > 0) S.persons(r.persons),
    ].join(' · ');
    final tw = widget.dense ? 64.0 : 80.0;
    return MouseRegion(
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: Material(
        color: widget.current ? c.accentSoft : (_hover ? c.hover : Colors.transparent),
        child: InkWell(
          onTap: widget.onOpen,
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: Sp.m, vertical: widget.dense ? 6 : Sp.s),
            decoration: widget.current
                ? BoxDecoration(border: Border(left: BorderSide(color: c.accent, width: 2)))
                : null,
            child: Row(children: [
              Container(
                width: tw,
                height: tw * 9 / 16,
                decoration: BoxDecoration(borderRadius: BorderRadius.circular(3), border: Border.all(color: c.border)),
                clipBehavior: Clip.antiAlias,
                child: _Thumb(r: r, iconSize: 18),
              ),
              const SizedBox(width: Sp.s + 2),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(r.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600, color: c.text)),
                  const SizedBox(height: 2),
                  Text(meta,
                      maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 11.5, color: c.muted)),
                ]),
              ),
              if (_hover && widget.onRemove != null)
                Tooltip(
                  message: S.removeRecent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(3),
                    onTap: widget.onRemove,
                    child: Padding(
                      padding: const EdgeInsets.all(2),
                      child: Icon(Icons.close, size: 14, color: c.muted),
                    ),
                  ),
                ),
            ]),
          ),
        ),
      ),
    );
  }
}

class _Thumb extends StatelessWidget {
  const _Thumb({required this.r, required this.iconSize});
  final RecentProject r;
  final double iconSize;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final fallback = ColoredBox(
      color: c.canvas,
      child: Center(child: Icon(Icons.movie_outlined, size: iconSize, color: const Color(0xFF8A929D))),
    );
    final t = r.thumb;
    if (t == null) return fallback;
    return ColoredBox(
      color: c.canvas, // portrait videos are letterboxed, never cropped to a strip
      child: Image.file(File(t), fit: BoxFit.contain, cacheWidth: 192, gaplessPlayback: true, errorBuilder: (_, _, _) => fallback),
    );
  }
}
