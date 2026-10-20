import 'dart:io';

import 'package:flutter/material.dart';

import '../app_state.dart';
import '../models.dart';
import '../strings.dart';
import '../theme.dart';
import '../version.dart';

Future<void> renamePerson(BuildContext context, AppState st, Person p) async {
  final ctl = TextEditingController(text: p.displayName);
  final name = await showDialog<String>(
    context: context,
    builder: (c) => AlertDialog(
      title: const Text(S.renameTitle),
      content: SizedBox(
        width: 340,
        child: TextField(
          controller: ctl,
          autofocus: true,
          onSubmitted: (v) => Navigator.pop(c, v),
        ),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(c), child: const Text(S.cancel)),
        FilledButton(onPressed: () => Navigator.pop(c, ctl.text), child: const Text(S.save)),
      ],
    ),
  );
  if (name != null) await st.rename(p.id, name);
}

Future<void> splitPerson(BuildContext context, AppState st, Person p) async {
  final tracks = await st.tracksOf(p.id);
  if (!context.mounted) return;
  final chosen = <int>{};
  final ids = await showDialog<List<int>>(
    context: context,
    builder: (c) => StatefulBuilder(
      builder: (c, set) {
        final pal = c.pal;
        return AlertDialog(
          title: Text('${S.splitTitle}: ${p.displayName}'),
          content: SizedBox(
            width: 420,
            child: tracks.length < 2
                ? const Text(S.splitNothing)
                : Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(S.splitHelp, style: TextStyle(color: pal.muted)),
                    const SizedBox(height: Sp.m),
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxHeight: 360),
                      child: ListView(shrinkWrap: true, children: [
                        for (final t in tracks)
                          CheckboxListTile(
                            dense: true,
                            value: chosen.contains(t.id),
                            onChanged: (v) => set(() => v == true ? chosen.add(t.id) : chosen.remove(t.id)),
                            secondary: ClipRRect(
                              borderRadius: BorderRadius.circular(3),
                              child: SizedBox(
                                width: 40,
                                height: 40,
                                child: t.thumbnail != null
                                    ? Image.file(File(t.thumbnail!), fit: BoxFit.cover, cacheWidth: 80)
                                    : ColoredBox(color: pal.hover, child: Icon(Icons.person_outline, color: pal.faint)),
                              ),
                            ),
                            title: Text(S.range(fmtDuration(t.startS), fmtDuration(t.endS))),
                            controlAffinity: ListTileControlAffinity.trailing,
                          ),
                      ]),
                    ),
                  ]),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(c), child: const Text(S.close)),
            if (tracks.length >= 2)
              FilledButton(
                onPressed:
                    chosen.isEmpty || chosen.length == tracks.length ? null : () => Navigator.pop(c, chosen.toList()),
                child: const Text(S.splitDo),
              ),
          ],
        );
      },
    ),
  );
  if (ids != null && ids.isNotEmpty) await st.splitTracks(ids);
}

/// After a box is drawn on the preview: whose face is it?
Future<int?> askWhoseFace(BuildContext context, AppState st) => showDialog<int>(
      context: context,
      builder: (c) {
        final pal = c.pal;
        return SimpleDialog(
          title: const Text(S.whoseFace),
          children: [
            SizedBox(
              width: 380,
              height: 360,
              child: ListView(children: [
                for (final p in st.persons)
                  ListTile(
                    dense: true,
                    leading: ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: SizedBox(
                        width: 36,
                        height: 36,
                        child: p.thumbnails.isEmpty
                            ? ColoredBox(color: pal.hover)
                            : Image.file(File(p.thumbnails.first), fit: BoxFit.cover, cacheWidth: 72),
                      ),
                    ),
                    title: Text(p.displayName),
                    onTap: () => Navigator.pop(c, p.id),
                  ),
                ListTile(
                  dense: true,
                  leading: SizedBox(width: 36, height: 36, child: Icon(Icons.person_add_alt, color: pal.accentText)),
                  title: Text(S.newPerson, style: TextStyle(color: pal.accentText, fontWeight: FontWeight.w600)),
                  onTap: () => Navigator.pop(c, 0),
                ),
              ]),
            ),
          ],
        );
      },
    );

/// Keyboard shortcuts in one place (Yardım > Klavye kısayolları); the same keys are in menus and tooltips.
const shortcutList = <(String, String)>[
  ('Space', S.scPlay),
  ('\u2190  \u2192', S.scStep),
  ('Home  End', S.scEnds),
  ('Ctrl+O  Ctrl+N', S.scOpen),
  ('Ctrl+S', S.scExport),
  ('Ctrl+B', S.scSidebar),
];

Future<void> showShortcuts(BuildContext context) => showDialog<void>(
      context: context,
      builder: (c) {
        final pal = c.pal;
        return AlertDialog(
          title: const Text(S.shortcuts),
          content: SizedBox(
            width: 380,
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              for (final (k, v) in shortcutList)
                Container(
                  height: 34,
                  decoration: BoxDecoration(border: Border(bottom: BorderSide(color: pal.border))),
                  child: Row(children: [
                    SizedBox(
                      width: 120,
                      child: Align(
                        alignment: Alignment.centerLeft,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: pal.surfaceAlt,
                            borderRadius: BorderRadius.circular(3),
                            border: Border.all(color: pal.borderStrong),
                          ),
                          child: Text(k,
                              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: pal.text)),
                        ),
                      ),
                    ),
                    Expanded(child: Text(v, style: TextStyle(fontSize: 13, color: pal.text))),
                  ]),
                ),
            ]),
          ),
          actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text(S.ok))],
        );
      },
    );

Future<void> showAbout(BuildContext context, AppState st) => showDialog<void>(
      context: context,
      builder: (c) {
        final pal = c.pal;
        return AlertDialog(
          content: SizedBox(
            width: 380,
            child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                const AppMark(size: 36),
                const SizedBox(width: Sp.m),
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(S.appName, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: pal.text)),
                  Text(S.version(appVersion), style: TextStyle(fontSize: 12, color: pal.muted)),
                ]),
              ]),
              const SizedBox(height: Sp.l),
              Text(S.aboutText, style: TextStyle(fontSize: 13, color: pal.text, height: 1.4)),
              const SizedBox(height: Sp.m),
              Text(st.deviceText, style: TextStyle(fontSize: 12, color: pal.muted)),
              const SizedBox(height: Sp.xs),
              Text('© ${DateTime.now().year} ${S.aboutMaker}', style: TextStyle(fontSize: 12, color: pal.muted)),
            ]),
          ),
          actions: [FilledButton(onPressed: () => Navigator.pop(c), child: const Text(S.close))],
        );
      },
    );

/// The app's mark: a small accent square with the blur glyph (no gradient).
class AppMark extends StatelessWidget {
  const AppMark({super.key, this.size = 20});
  final double size;
  @override
  Widget build(BuildContext context) {
    final pal = context.pal;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(color: pal.accent, borderRadius: BorderRadius.circular(size / 5)),
      child: Icon(Icons.blur_on, color: pal.onAccent, size: size * 0.72),
    );
  }
}
