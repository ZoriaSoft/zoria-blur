import 'dart:async';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import '../app_state.dart';
import '../models.dart';
import '../strings.dart';
import '../system_check.dart';
import '../theme.dart';
import '../widgets/dialogs.dart';

/// First-run setup, full window: 1 Hoş geldiniz -> 2 Sistem (the live system check) -> 3 Hazır (the choice,
/// default output folder). Skippable at any step; the check then finishes in the background.
class SetupWizard extends StatefulWidget {
  const SetupWizard({super.key, required this.st});
  final AppState st;
  @override
  State<SetupWizard> createState() => _SetupWizardState();
}

class _SetupWizardState extends State<SetupWizard> {
  AppState get st => widget.st;
  SystemCheck get check => st.check;

  /// Rows shown with their real result; the next one shows "Kontrol ediliyor". One row is revealed at a time
  /// (at least [_revealStep] apart) so a fast PC still reads as a step-by-step check.
  int _revealed = 0;
  Timer? _tick;
  static const _revealStep = Duration(milliseconds: 380);
  late bool _beside = st.settings.outputDir == null;
  double _panelH = 640;
  bool get _compact => _panelH < 600; // 1024x640 window: tighter rows so the whole check fits without scrolling
  late String? _dir = st.settings.outputDir;

  @override
  void initState() {
    super.initState();
    check.addListener(_onCheck);
    _tick = Timer.periodic(_revealStep, (_) => _advance());
  }

  @override
  void dispose() {
    _tick?.cancel();
    check.removeListener(_onCheck);
    super.dispose();
  }

  void _onCheck() {
    if (!check.started || (check.running && check.doneCount == 0)) _revealed = 0;
  }

  void _advance() {
    if (!mounted || st.wizardStep != 1) return;
    final rows = check.rows;
    if (_revealed < rows.length && rows[_revealed].done && check.started) setState(() => _revealed++);
  }

  bool get _allShown => check.finished && _revealed >= check.rows.length;

  CheckState _shownState(int i) {
    final r = check.rows[i];
    if (!check.started) return CheckState.waiting;
    if (i < _revealed) return r.state;
    if (i == _revealed) return check.running || r.done ? CheckState.checking : r.state;
    return CheckState.waiting;
  }

  Future<void> _pickDir() async {
    final d = await getDirectoryPath(initialDirectory: _dir, confirmButtonText: S.outPick.replaceAll('…', ''));
    if (d == null) return;
    setState(() {
      _dir = d;
      _beside = false;
    });
  }

  void _finish() => st.finishWizard(besideVideo: _beside || _dir == null, outputDir: _beside ? null : _dir);

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([st, check]),
      builder: (context, _) {
        final c = context.pal;
        final step = st.wizardStep;
        return Scaffold(
          backgroundColor: c.chrome,
          body: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            _header(c, step),
            Expanded(
              child: LayoutBuilder(builder: (context, cons) {
                return Center(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: Sp.xl, vertical: Sp.l),
                    // one fixed panel size for all three steps, so the stepper and the buttons never jump
                    child: SizedBox(
                      width: 700,
                      height: _panelH = (cons.maxHeight - 2 * Sp.l).clamp(0.0, 640.0),
                      child: Container(
                        key: const ValueKey('setup-panel'),
                        decoration: BoxDecoration(
                          color: c.surface,
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: c.border),
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                          _WizardSteps(current: step),
                          Expanded(
                            child: SingleChildScrollView(
                              padding: EdgeInsets.fromLTRB(Sp.xl, _compact ? Sp.l : Sp.xl, Sp.xl, _compact ? Sp.m : Sp.l),
                              child: switch (step) {
                                0 => const _WelcomeStep(),
                                1 => _systemStep(c),
                                _ => _resultStep(c),
                              },
                            ),
                          ),
                          _footer(c, step),
                        ]),
                      ),
                    ),
                  ),
                );
              }),
            ),
            Container(
              height: 24,
              padding: const EdgeInsets.symmetric(horizontal: Sp.m),
              decoration: BoxDecoration(color: c.chrome, border: Border(top: BorderSide(color: c.border))),
              child: Row(children: [
                Expanded(
                  child: Text(check.running ? S.checkRunning : S.statusReady,
                      maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 11.5, color: c.muted)),
                ),
                Icon(Icons.lock_outline, size: 12, color: c.faint),
                const SizedBox(width: Sp.xs),
                Text(S.privacy, style: TextStyle(fontSize: 11.5, color: c.muted)),
              ]),
            ),
          ]),
        );
      },
    );
  }

  Widget _header(Pal c, int step) => Container(
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: Sp.m),
        decoration: BoxDecoration(color: c.chrome, border: Border(bottom: BorderSide(color: c.border))),
        child: Row(children: [
          const AppMark(size: 20),
          const SizedBox(width: Sp.s),
          Text(S.appName, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: c.text)),
          const SizedBox(width: Sp.s),
          Container(width: 1, height: 16, color: c.border),
          const SizedBox(width: Sp.s),
          Text(S.setupTitle, style: TextStyle(fontSize: 13, color: c.muted)),
          const Spacer(),
          if (step < 2)
            Tooltip(
              message: S.setupSkipTip,
              child: TextButton(
                key: const ValueKey('setup-skip'),
                onPressed: () => st.finishWizard(),
                child: const Text(S.setupSkip),
              ),
            ),
        ]),
      );

  Widget _footer(Pal c, int step) {
    final Widget primary = switch (step) {
      0 => FilledButton.icon(
          key: const ValueKey('setup-next'),
          onPressed: () => st.wizardGo(1),
          icon: const Icon(Icons.arrow_forward, size: 18),
          label: const Text(S.letsStart),
        ),
      1 => FilledButton(
          key: const ValueKey('setup-next'),
          onPressed: _allShown ? () => st.wizardGo(2) : null,
          child: const Text(S.next),
        ),
      _ => FilledButton(key: const ValueKey('setup-finish'), onPressed: _finish, child: const Text(S.finish)),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: Sp.l, vertical: Sp.m),
      decoration: BoxDecoration(color: c.surfaceAlt, border: Border(top: BorderSide(color: c.border))),
      child: Row(children: [
        if (step > 0)
          OutlinedButton.icon(
            key: const ValueKey('setup-back'),
            onPressed: () => st.wizardGo(step - 1),
            icon: const Icon(Icons.arrow_back, size: 16),
            label: const Text(S.back),
          ),
        const Spacer(),
        if (step == 1 && !_allShown) ...[
          SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: c.muted)),
          const SizedBox(width: Sp.s),
          Flexible(
            child: Text(S.checkProgress(_revealed.clamp(0, check.rows.length), check.rows.length),
                maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: c.muted)),
          ),
          const SizedBox(width: Sp.m),
        ],
        primary,
      ]),
    );
  }

  // ------------------------------------------------------------ step 2: system check
  Widget _systemStep(Pal c) {
    final rows = check.rows;
    final res = check.result;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Text(S.checkTitle, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: c.text)),
      const SizedBox(height: Sp.xs),
      Text(S.checkLead, style: TextStyle(fontSize: 13, color: c.muted, height: 1.35)),
      SizedBox(height: _compact ? Sp.m : Sp.l),
      Container(
        key: const ValueKey('check-list'),
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(4), border: Border.all(color: c.border)),
        child: Column(children: [
          for (var i = 0; i < rows.length; i++) ...[
            if (i > 0) Divider(height: 1, color: c.border),
            _CheckRowView(row: rows[i], shown: _shownState(i), compact: _compact),
          ],
        ]),
      ),
      if (_allShown && res != null) ...[
        const SizedBox(height: Sp.m),
        _Note(
          icon: res.probeFailed ? Icons.info_outline : Icons.check_circle_outline,
          warn: res.probeFailed,
          text: '${S.resultTitle}: ${_deviceLine(res)}. ${res.why}',
        ),
      ],
    ]);
  }

  String _deviceLine(CheckResult r) {
    final d = r.device;
    if (d.gpu) return S.resultGpu(DeviceInfo.cleanName(d.deviceName ?? S.deviceGpuUnknown), d.api);
    final cpu = d.cpuName;
    return cpu == null ? S.resultCpuPlain : S.resultCpu(DeviceInfo.cleanName(cpu));
  }

  // ------------------------------------------------------------ step 3: ready
  Widget _resultStep(Pal c) {
    final res = check.result;
    final d = res?.device ?? st.device ?? DeviceInfo(chosen: 'cpu');
    final speed = d.speedUp;
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: res?.probeFailed == true ? c.warnSoft : c.accentSoft,
            borderRadius: BorderRadius.circular(6),
          ),
          child: Icon(d.gpu ? Icons.developer_board : Icons.memory,
              size: 26, color: res?.probeFailed == true ? c.warn : c.accentText),
        ),
        const SizedBox(width: Sp.l),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(S.resultTitle, style: TextStyle(fontSize: 12.5, color: c.muted)),
            const SizedBox(height: 2),
            Text(res == null ? S.resultCpuPlain : _deviceLine(res),
                key: const ValueKey('result-device'),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: c.text)),
            if (res != null) ...[
              const SizedBox(height: Sp.s),
              Text(res.why, style: TextStyle(fontSize: 13, color: c.text, height: 1.35)),
            ],
            if (speed != null) ...[
              const SizedBox(height: Sp.xs),
              Text(S.speedHint(fmtTimes(speed)),
                  style: TextStyle(fontSize: 13, color: c.accentText, fontWeight: FontWeight.w600, height: 1.35)),
            ],
            const SizedBox(height: Sp.s),
            Row(children: [
              Icon(Icons.tune, size: 14, color: c.faint),
              const SizedBox(width: Sp.xs),
              Expanded(child: Text(S.changeLater, style: TextStyle(fontSize: 12, color: c.muted))),
            ]),
          ]),
        ),
      ]),
      const SizedBox(height: Sp.l),
      Divider(height: 1, color: c.border),
      const SizedBox(height: Sp.m),
      const SectionLabel(S.outFolderTitle),
      _Choice(
        key: const ValueKey('out-beside'),
        selected: _beside,
        onTap: () => setState(() => _beside = true),
        child: Text(S.outBesideVideo, style: TextStyle(fontSize: 13, color: c.text)),
      ),
      _Choice(
        key: const ValueKey('out-fixed'),
        selected: !_beside,
        onTap: () => _dir == null ? _pickDir() : setState(() => _beside = false),
        child: Row(children: [
          Text(S.outFixed, style: TextStyle(fontSize: 13, color: c.text)),
          const SizedBox(width: Sp.s),
          Expanded(
            child: Tooltip(
              message: _dir ?? '',
              child: Text(_dir ?? S.outNone,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 12.5, color: _dir == null ? c.faint : c.muted)),
            ),
          ),
          const SizedBox(width: Sp.s),
          OutlinedButton.icon(
            onPressed: _pickDir,
            icon: const Icon(Icons.folder_open_outlined, size: 16),
            label: const Text(S.outPick),
          ),
        ]),
      ),
      const SizedBox(height: Sp.xs),
      Padding(
        padding: const EdgeInsets.only(left: 30),
        child: Text(S.outAskEachTime, style: TextStyle(fontSize: 12, color: c.faint)),
      ),
    ]);
  }
}

/// 1 Hoş geldiniz — 2 Sistem — 3 Hazır
class _WizardSteps extends StatelessWidget {
  const _WizardSteps({required this.current});
  final int current;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    const labels = [S.wStep1, S.wStep2, S.wStep3];
    final items = <Widget>[];
    for (var i = 0; i < labels.length; i++) {
      final active = i == current, done = i < current;
      if (i > 0) {
        items.add(Expanded(
          child: Container(
            height: 1,
            margin: const EdgeInsets.symmetric(horizontal: Sp.m),
            color: done || active ? c.accent : c.borderStrong,
          ),
        ));
      }
      items.add(Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 24,
          height: 24,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: active || done ? c.accent : Colors.transparent,
            shape: BoxShape.circle,
            border: Border.all(color: active || done ? c.accent : c.borderStrong),
          ),
          child: done
              ? Icon(Icons.check, size: 14, color: c.onAccent)
              : Text('${i + 1}',
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: active ? c.onAccent : c.muted)),
        ),
        const SizedBox(width: Sp.s),
        Text(labels[i],
            style: TextStyle(
                fontSize: 13,
                fontWeight: active ? FontWeight.w600 : FontWeight.w400,
                color: active || done ? c.text : c.muted)),
      ]));
    }
    return Container(
      key: const ValueKey('setup-steps'),
      padding: const EdgeInsets.symmetric(horizontal: Sp.xl, vertical: Sp.l),
      decoration: BoxDecoration(border: Border(bottom: BorderSide(color: c.border))),
      child: Row(children: items),
    );
  }
}

class _WelcomeStep extends StatelessWidget {
  const _WelcomeStep();
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    Widget point(IconData icon, String title, String sub) => Padding(
          padding: const EdgeInsets.only(bottom: Sp.l),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: c.surfaceAlt,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: c.border),
              ),
              child: Icon(icon, size: 18, color: c.accentText),
            ),
            const SizedBox(width: Sp.m),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(title, style: TextStyle(fontSize: 13.5, fontWeight: FontWeight.w600, color: c.text)),
                const SizedBox(height: 2),
                Text(sub, style: TextStyle(fontSize: 12.5, color: c.muted, height: 1.35)),
              ]),
            ),
          ]),
        );
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      Row(children: [
        const AppMark(size: 48),
        const SizedBox(width: Sp.l),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(S.welcomeTitle,
                key: const ValueKey('welcome-title'),
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600, color: c.text)),
            const SizedBox(height: 2),
            Text(S.welcomeLead, style: TextStyle(fontSize: 13.5, color: c.muted)),
          ]),
        ),
      ]),
      const SizedBox(height: Sp.xl),
      point(Icons.lock_outline, S.welcomePrivacy, S.welcomePrivacySub),
      point(Icons.fact_check_outlined, S.welcomeCheck, S.welcomeCheckSub),
      point(Icons.tune, S.welcomeLater, S.welcomeLaterSub),
    ]);
  }
}

class _CheckRowView extends StatelessWidget {
  const _CheckRowView({required this.row, required this.shown, this.compact = false});
  final CheckRow row;
  final CheckState shown;
  final bool compact;

  static const _icons = {
    'cpu': Icons.memory,
    'memory': Icons.storage_outlined,
    'gpus': Icons.developer_board,
    'cuda': Icons.bolt_outlined,
    'dml': Icons.layers_outlined,
    'test': Icons.speed,
  };

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final real = shown == row.state;
    final detail = real ? row.detail : '';
    final label = real
        ? row.stateText
        : (shown == CheckState.checking ? S.stChecking : S.stWaiting);
    final (Color fg, Widget icon) = switch (shown) {
      CheckState.waiting => (c.faint, Icon(Icons.radio_button_unchecked, size: 16, color: c.faint)),
      CheckState.checking => (
          c.accentText,
          SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 1.8, color: c.accent))
        ),
      CheckState.found => (c.accentText, Icon(Icons.check_circle_outline, size: 16, color: c.accentText)),
      CheckState.missing => (c.muted, Icon(Icons.remove_circle_outline, size: 16, color: c.muted)),
      CheckState.failed => (c.warn, Icon(Icons.error_outline, size: 16, color: c.warn)),
    };
    return Container(
      key: ValueKey('check-${row.id}'),
      constraints: BoxConstraints(minHeight: compact ? 40 : 48),
      padding: EdgeInsets.symmetric(horizontal: Sp.m, vertical: compact ? 4 : Sp.s),
      child: Row(children: [
        Icon(_icons[row.id], size: 18, color: shown == CheckState.waiting ? c.faint : c.muted),
        const SizedBox(width: Sp.m),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
            Text(row.title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600, color: shown == CheckState.waiting ? c.muted : c.text)),
            if (detail.isNotEmpty) ...[
              const SizedBox(height: 1),
              Text(detail, maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: c.muted)),
            ],
          ]),
        ),
        const SizedBox(width: Sp.m),
        SizedBox(
          width: 128,
          child: Row(mainAxisAlignment: MainAxisAlignment.end, children: [
            icon,
            const SizedBox(width: 6),
            Flexible(
              child: Text(label,
                  key: ValueKey('check-${row.id}-state'),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: fg)),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _Note extends StatelessWidget {
  const _Note({required this.icon, required this.text, this.warn = false});
  final IconData icon;
  final String text;
  final bool warn;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final fg = warn ? c.warn : c.accentText;
    return Container(
      key: const ValueKey('check-note'),
      padding: const EdgeInsets.all(Sp.m),
      decoration: BoxDecoration(
        color: warn ? c.warnSoft : c.accentSoft,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: fg.withValues(alpha: 0.35)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, size: 18, color: fg),
        const SizedBox(width: Sp.s),
        Expanded(child: Text(text, style: TextStyle(fontSize: 13, color: c.text, height: 1.35))),
      ]),
    );
  }
}

/// One option of a single choice (radio look, whole row clickable).
class _Choice extends StatelessWidget {
  const _Choice({super.key, required this.selected, required this.onTap, required this.child});
  final bool selected;
  final VoidCallback onTap;
  final Widget child;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(4),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 2),
        child: Row(children: [
          Icon(selected ? Icons.radio_button_checked : Icons.radio_button_unchecked,
              size: 18, color: selected ? c.accent : c.borderStrong),
          const SizedBox(width: Sp.s + 2),
          Expanded(child: child),
        ]),
      ),
    );
  }
}
