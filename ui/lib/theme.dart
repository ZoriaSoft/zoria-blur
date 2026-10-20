import 'package:flutter/material.dart';

/// Neutral corporate desktop look, light and dark: greys, one restrained teal accent, hairline borders,
/// small radii, no gradients, no shadows. Every colour the app paints comes from [Pal].
@immutable
class Pal extends ThemeExtension<Pal> {
  const Pal({
    required this.chrome,
    required this.surface,
    required this.surfaceAlt,
    required this.hover,
    required this.border,
    required this.borderStrong,
    required this.text,
    required this.muted,
    required this.faint,
    required this.accent,
    required this.onAccent,
    required this.accentSoft,
    required this.accentText,
    required this.danger,
    required this.dangerSoft,
    required this.warn,
    required this.warnSoft,
    required this.canvas,
    required this.presence,
    required this.presenceBlur,
    required this.risk,
    required this.riskSevere,
  });

  final Color chrome; // window background, menu bar, status bar
  final Color surface; // panels
  final Color surfaceAlt; // section headers, timeline label column
  final Color hover;
  final Color border;
  final Color borderStrong;
  final Color text;
  final Color muted;
  final Color faint;
  final Color accent;
  final Color onAccent;
  final Color accentSoft;
  final Color accentText; // accent used as text on surface (contrast >= 4.5)
  final Color danger;
  final Color dangerSoft;
  final Color warn;
  final Color warnSoft;
  final Color canvas; // behind the video
  final Color presence; // timeline: person on screen, stays visible
  final Color presenceBlur; // timeline: person on screen, will be blurred
  final Color risk;
  final Color riskSevere;

  static const light = Pal(
    chrome: Color(0xFFF2F3F5),
    surface: Color(0xFFFFFFFF),
    surfaceAlt: Color(0xFFF7F8F9),
    hover: Color(0xFFEDEFF2),
    border: Color(0xFFDCE0E5),
    borderStrong: Color(0xFFC3C9D0),
    text: Color(0xFF16191D),
    muted: Color(0xFF555D68),
    faint: Color(0xFF858D98),
    accent: Color(0xFF0B6E5F),
    onAccent: Color(0xFFFFFFFF),
    accentSoft: Color(0xFFE1EFEB),
    accentText: Color(0xFF0A6457),
    danger: Color(0xFFC0333A),
    dangerSoft: Color(0xFFFBEBEC),
    warn: Color(0xFF8A5800),
    warnSoft: Color(0xFFFFF4DC),
    canvas: Color(0xFF15171A),
    presence: Color(0xFFB7BEC7),
    presenceBlur: Color(0xFF3E8F82),
    risk: Color(0xFFE0A030),
    riskSevere: Color(0xFFC0333A),
  );

  static const dark = Pal(
    chrome: Color(0xFF1A1C1F),
    surface: Color(0xFF222529),
    surfaceAlt: Color(0xFF1E2124),
    hover: Color(0xFF2B2F34),
    border: Color(0xFF34383E),
    borderStrong: Color(0xFF474C54),
    text: Color(0xFFE6E8EB),
    muted: Color(0xFFA3AAB3),
    faint: Color(0xFF79818B),
    accent: Color(0xFF4DB5A2),
    onAccent: Color(0xFF07201B),
    accentSoft: Color(0xFF1C3632),
    accentText: Color(0xFF63C4B1),
    danger: Color(0xFFF0767C),
    dangerSoft: Color(0xFF3A2225),
    warn: Color(0xFFE2B45A),
    warnSoft: Color(0xFF362E1E),
    canvas: Color(0xFF0E0F11),
    presence: Color(0xFF555C66),
    presenceBlur: Color(0xFF3F9C8C),
    risk: Color(0xFFD99A2B),
    riskSevere: Color(0xFFF0767C),
  );

  @override
  Pal copyWith() => this;

  @override
  Pal lerp(ThemeExtension<Pal>? other, double t) => t < 0.5 || other is! Pal ? this : other;
}

extension PalX on BuildContext {
  Pal get pal => Theme.of(this).extension<Pal>() ?? Pal.light;
}

/// Spacing scale (4 / 8 px grid).
class Sp {
  static const xs = 4.0;
  static const s = 8.0;
  static const m = 12.0;
  static const l = 16.0;
  static const xl = 24.0;
}

const _radius = 4.0;

ThemeData buildTheme([Brightness brightness = Brightness.light]) {
  final c = brightness == Brightness.dark ? Pal.dark : Pal.light;
  final scheme = ColorScheme.fromSeed(seedColor: c.accent, brightness: brightness).copyWith(
    primary: c.accent,
    onPrimary: c.onAccent,
    surface: c.surface,
    onSurface: c.text,
    onSurfaceVariant: c.muted,
    surfaceContainerHighest: c.hover,
    surfaceContainerHigh: c.surfaceAlt,
    surfaceContainer: c.surface,
    surfaceContainerLow: c.surface,
    surfaceContainerLowest: c.surface,
    error: c.danger,
    outline: c.borderStrong,
    outlineVariant: c.border,
    secondaryContainer: c.accentSoft,
    onSecondaryContainer: c.accentText,
    inverseSurface: c.text,
    onInverseSurface: c.surface,
  );
  final base = ThemeData(
    useMaterial3: true,
    brightness: brightness,
    colorScheme: scheme,
    fontFamily: 'Segoe UI',
    visualDensity: VisualDensity.compact,
    splashFactory: NoSplash.splashFactory,
  );
  final shape = RoundedRectangleBorder(borderRadius: BorderRadius.circular(_radius));
  const label = TextStyle(fontSize: 13, fontWeight: FontWeight.w600);
  return base.copyWith(
    extensions: [c],
    scaffoldBackgroundColor: c.chrome,
    canvasColor: c.surface,
    dividerColor: c.border,
    dividerTheme: DividerThemeData(color: c.border, thickness: 1, space: 1),
    hoverColor: c.hover,
    focusColor: c.accent.withValues(alpha: 0.14),
    highlightColor: Colors.transparent,
    textTheme: base.textTheme
        .apply(bodyColor: c.text, displayColor: c.text)
        .copyWith(
          bodyMedium: const TextStyle(fontSize: 13),
          bodySmall: TextStyle(fontSize: 12, color: c.muted),
          labelLarge: label,
        ),
    iconTheme: IconThemeData(color: c.muted, size: 18),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(0, 34),
        padding: const EdgeInsets.symmetric(horizontal: Sp.l),
        shape: shape,
        elevation: 0,
        textStyle: label,
        disabledBackgroundColor: c.hover,
        disabledForegroundColor: c.faint,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(0, 32),
        padding: const EdgeInsets.symmetric(horizontal: Sp.m),
        foregroundColor: c.text,
        side: BorderSide(color: c.borderStrong),
        shape: shape,
        textStyle: label,
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        minimumSize: const Size(0, 30),
        padding: const EdgeInsets.symmetric(horizontal: Sp.s),
        foregroundColor: c.accentText,
        shape: shape,
        textStyle: label,
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(
        foregroundColor: c.muted,
        shape: shape,
        minimumSize: const Size(32, 32),
        padding: const EdgeInsets.all(6),
        iconSize: 18,
      ),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: ButtonStyle(
        visualDensity: VisualDensity.compact,
        textStyle: const WidgetStatePropertyAll(TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600)),
        side: WidgetStatePropertyAll(BorderSide(color: c.borderStrong)),
        shape: WidgetStatePropertyAll(shape),
        backgroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected) ? c.accentSoft : Colors.transparent),
        foregroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected) ? c.accentText : c.text),
      ),
    ),
    sliderTheme: SliderThemeData(
      trackHeight: 3,
      activeTrackColor: c.accent,
      inactiveTrackColor: c.border,
      thumbColor: c.accent,
      overlayShape: const RoundSliderOverlayShape(overlayRadius: 12),
      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7, elevation: 0, pressedElevation: 0),
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: c.accent,
      linearTrackColor: c.border,
      linearMinHeight: 4,
    ),
    checkboxTheme: CheckboxThemeData(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(3)),
      side: BorderSide(color: c.borderStrong, width: 1.5),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: c.surface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(6), side: BorderSide(color: c.borderStrong)),
      titleTextStyle: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: c.text, fontFamily: 'Segoe UI'),
    ),
    inputDecorationTheme: InputDecorationTheme(
      isDense: true,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(_radius), borderSide: BorderSide(color: c.borderStrong)),
      enabledBorder:
          OutlineInputBorder(borderRadius: BorderRadius.circular(_radius), borderSide: BorderSide(color: c.borderStrong)),
      focusedBorder:
          OutlineInputBorder(borderRadius: BorderRadius.circular(_radius), borderSide: BorderSide(color: c.accent, width: 1.5)),
    ),
    tooltipTheme: TooltipThemeData(
      waitDuration: const Duration(milliseconds: 500),
      decoration: BoxDecoration(color: brightness == Brightness.dark ? const Color(0xFF3A3F46) : const Color(0xFF2A2E33),
          borderRadius: BorderRadius.circular(_radius)),
      textStyle: const TextStyle(color: Colors.white, fontSize: 12),
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: brightness == Brightness.dark ? const Color(0xFF3A3F46) : const Color(0xFF2A2E33),
      contentTextStyle: const TextStyle(color: Colors.white, fontSize: 13),
      shape: shape,
      elevation: 0,
    ),
    menuTheme: MenuThemeData(
      style: MenuStyle(
        backgroundColor: WidgetStatePropertyAll(c.surface),
        surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
        elevation: const WidgetStatePropertyAll(2),
        shadowColor: WidgetStatePropertyAll(Colors.black.withValues(alpha: 0.25)),
        side: WidgetStatePropertyAll(BorderSide(color: c.border)),
        shape: WidgetStatePropertyAll(shape),
        padding: const WidgetStatePropertyAll(EdgeInsets.symmetric(vertical: Sp.xs)),
      ),
    ),
    menuBarTheme: MenuBarThemeData(
      style: MenuStyle(
        backgroundColor: const WidgetStatePropertyAll(Colors.transparent),
        surfaceTintColor: const WidgetStatePropertyAll(Colors.transparent),
        elevation: const WidgetStatePropertyAll(0),
        shape: const WidgetStatePropertyAll(RoundedRectangleBorder()),
        padding: const WidgetStatePropertyAll(EdgeInsets.zero),
      ),
    ),
    menuButtonTheme: MenuButtonThemeData(
      style: ButtonStyle(
        minimumSize: const WidgetStatePropertyAll(Size(0, 30)),
        padding: const WidgetStatePropertyAll(EdgeInsets.symmetric(horizontal: Sp.m)),
        textStyle: const WidgetStatePropertyAll(TextStyle(fontSize: 13, fontFamily: 'Segoe UI')),
        foregroundColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.disabled) ? c.faint : c.text),
        iconColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.disabled) ? c.faint : c.muted),
        iconSize: const WidgetStatePropertyAll(16),
        overlayColor: WidgetStatePropertyAll(c.hover),
        shape: WidgetStatePropertyAll(shape),
      ),
    ),
    expansionTileTheme: ExpansionTileThemeData(
      shape: const Border(),
      collapsedShape: const Border(),
      iconColor: c.muted,
      collapsedIconColor: c.muted,
      textColor: c.text,
      collapsedTextColor: c.text,
    ),
  );
}

/// Panel with a hairline border; no shadow, small radius.
class Panel extends StatelessWidget {
  const Panel({super.key, required this.child, this.padding = const EdgeInsets.all(Sp.l)});
  final Widget child;
  final EdgeInsetsGeometry padding;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: c.border),
      ),
      child: child,
    );
  }
}

/// Small section heading used in the inspector and dialogs.
class SectionLabel extends StatelessWidget {
  const SectionLabel(this.text, {super.key, this.trailing});
  final String text;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    return SizedBox(
      height: 28,
      child: Row(children: [
        Expanded(
          child: Text(text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: c.muted, letterSpacing: 0.1)),
        ),
        ?trailing,
      ]),
    );
  }
}
