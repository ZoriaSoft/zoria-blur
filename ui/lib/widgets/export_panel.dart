import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;

import '../app_state.dart';
import '../models.dart';
import '../strings.dart';
import '../theme.dart';

/// Export progress and its result as a modal panel over the editor (the editor stays where it was).
class ExportOverlay extends StatelessWidget {
  const ExportOverlay({super.key, required this.st});
  final AppState st;

  @override
  Widget build(BuildContext context) {
    final done = st.stage == Stage.done;
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
                key: ValueKey(done ? 'export-done' : 'export-progress'),
                decoration: BoxDecoration(
                  color: context.pal.surface,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: context.pal.borderStrong),
                  boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.18), blurRadius: 24, offset: const Offset(0, 8))],
                ),
                child: done ? _Done(st: st) : _Progress(st: st),
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

class _Done extends StatelessWidget {
  const _Done({required this.st});
  final AppState st;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final issues = st.qaHasIssues;
    final q = st.qa;
    final riskOnly = !issues && q != null && st.qaSummary != S.qaClean;
    final boxColor = issues ? c.dangerSoft : (riskOnly ? c.warnSoft : c.accentSoft);
    final fg = issues ? c.danger : (riskOnly ? c.warn : c.accentText);
    final out = st.outputPath ?? '';
    return Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      _Header(icon: Icons.check_circle_outline, title: S.done, color: c.accentText),
      Padding(
        padding: const EdgeInsets.all(Sp.l),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text(p.basename(out),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
          const SizedBox(height: 2),
          Tooltip(
            message: out,
            child: Text(p.dirname(out),
                maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: c.muted)),
          ),
          const SizedBox(height: Sp.m),
          Container(
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
                    Wrap(spacing: Sp.s, runSpacing: Sp.s, children: [
                      for (final r in st.leakMoments)
                        ActionChip(
                          avatar: Icon(Icons.play_arrow, size: 16, color: c.danger),
                          label: Text(fmtDuration(r.startS)),
                          visualDensity: VisualDensity.compact,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(3)),
                          side: BorderSide(color: c.borderStrong),
                          onPressed: () => st.backToEdit(jumpToS: r.startS),
                        ),
                    ]),
                    const SizedBox(height: Sp.xs),
                    Text(S.seeMoments, style: TextStyle(color: c.muted, fontSize: 12)),
                  ],
                ]),
              ),
            ]),
          ),
        ]),
      ),
      Container(
        padding: const EdgeInsets.fromLTRB(Sp.l, Sp.m, Sp.l, Sp.m),
        decoration: BoxDecoration(color: c.surfaceAlt, border: Border(top: BorderSide(color: c.border))),
        child: Row(children: [
          TextButton(onPressed: () => st.backToEdit(), child: const Text(S.close)),
          const Spacer(),
          Flexible(
            flex: 6,
            child: OutlinedButton.icon(
              onPressed: st.openFolder,
              icon: const Icon(Icons.folder_open_outlined, size: 18),
              label: const Text(S.openFolder, overflow: TextOverflow.ellipsis),
            ),
          ),
          const SizedBox(width: Sp.s),
          Flexible(
            flex: 6,
            child: FilledButton.icon(
              key: const ValueKey('done-open-video'),
              onPressed: st.openOutput,
              icon: const Icon(Icons.play_circle_outline, size: 18),
              label: const Text(S.openVideo, overflow: TextOverflow.ellipsis),
            ),
          ),
        ]),
      ),
    ]);
  }
}
