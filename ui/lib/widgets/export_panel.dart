import 'package:flutter/material.dart';

import '../app_state.dart';
import '../strings.dart';
import '../theme.dart';

/// Export progress as a modal panel over the editor (the editor stays where it was). The result has its own screen
/// (widgets/result_view.dart).
class ExportOverlay extends StatelessWidget {
  const ExportOverlay({super.key, required this.st});
  final AppState st;

  @override
  Widget build(BuildContext context) {
    return Stack(children: [
      ModalBarrier(color: Colors.black.withValues(alpha: 0.38), dismissible: false),
      Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: Padding(
            padding: const EdgeInsets.all(Sp.xl),
            child: Material(
              color: Colors.transparent,
              child: Container(
                key: const ValueKey('export-progress'),
                decoration: BoxDecoration(
                  color: context.pal.surface,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: context.pal.borderStrong),
                  boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.18), blurRadius: 24, offset: const Offset(0, 8))],
                ),
                child: _Progress(st: st),
              ),
            ),
          ),
        ),
      ),
    ]);
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.icon, required this.title, required this.color, this.trailing});
  final IconData icon;
  final String title;
  final Color color;
  final String? trailing;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return Container(
      height: 44,
      padding: const EdgeInsets.symmetric(horizontal: Sp.l),
      decoration: BoxDecoration(border: Border(bottom: BorderSide(color: c.border))),
      child: Row(children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(width: Sp.s),
        Expanded(child: Text(title, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: c.text))),
        if (trailing != null) Text(trailing!, style: TextStyle(fontSize: 12, color: c.muted)),
      ]),
    );
  }
}

class _Progress extends StatelessWidget {
  const _Progress({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final pct = st.progress >= 0 ? '${(st.progress * 100).round()}%' : '';
    return Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _Header(
          icon: Icons.save_alt, title: S.exporting, color: c.accentText, trailing: S.step(st.stepNo, st.stepCount)),
      Padding(
        padding: const EdgeInsets.fromLTRB(Sp.l, Sp.l, Sp.l, Sp.s),
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
              child: Text(st.phase,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
            ),
            Text(pct,
                style: TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600, color: c.text, fontFeatures: const [FontFeature.tabularFigures()])),
          ]),
          const SizedBox(height: 2),
          Text(st.etaText.isEmpty ? ' ' : st.etaText, style: TextStyle(fontSize: 12, color: c.muted)),
        ]),
      ),
      Padding(
        padding: const EdgeInsets.fromLTRB(Sp.l, Sp.s, Sp.l, Sp.l),
        child: Align(
          alignment: Alignment.centerRight,
          child: OutlinedButton(onPressed: st.cancelJob, child: const Text(S.cancel)),
        ),
      ),
    ]);
  }
}
