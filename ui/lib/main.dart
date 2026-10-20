import 'dart:async';
import 'dart:io';
import 'dart:ui' show AppExitResponse;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'app_state.dart';
import 'autopilot.dart';
import 'screens/home.dart';
import 'settings.dart';
import 'strings.dart';
import 'theme.dart';

/// Second launches (single instance, windows/runner/main.cpp) forward their video here.
const instanceChannel = MethodChannel('zoria_blur/instance');

/// App -> runner: caption colour follows the app theme.
const windowChannel = MethodChannel('zoria_blur/window');

Future<void> main(List<String> args) async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = AppSettings.load();
  // self-test screenshots in a chosen theme (not saved: the self-test uses its own settings folder anyway)
  switch (Platform.environment['ZORIA_AUTOPILOT_THEME']) {
    case 'dark':
      settings.themeMode = ThemeMode.dark;
    case 'light':
      settings.themeMode = ThemeMode.light;
  }
  final state = AppState(settings: settings)
    // first launch after install: welcome -> system check -> ready (never again once finished or skipped)
    ..wizardOpen = !settings.setupDone;
  instanceChannel.setMethodCallHandler((call) async {
    if (call.method == 'open' && call.arguments is String) {
      final path = call.arguments as String;
      _logInstance(path);
      await state.openExternal(path);
    }
    return null;
  });
  // closing the window: stop every engine process first
  AppLifecycleListener(onExitRequested: () async {
    await state.shutdown();
    return AppExitResponse.exit;
  });
  runApp(ZoriaBlurApp(state: state));
  final autopilot = Autopilot.fromEnvironment(state);
  final fromArgs = args.where((a) => !a.startsWith('-')).firstOrNull;
  if (autopilot == null && fromArgs != null) unawaited(state.openExternal(fromArgs));
  await state.init();
  autopilot?.run();
}

/// `ZORIA_INSTANCE_LOG=file`: append every forwarded path (self-test of the single-instance hand-over).
void _logInstance(String path) {
  final f = Platform.environment['ZORIA_INSTANCE_LOG'];
  if (f == null) return;
  try {
    File(f).writeAsStringSync('${DateTime.now().toIso8601String()} $path\n', mode: FileMode.append);
  } catch (_) {}
}

class ZoriaBlurApp extends StatelessWidget {
  const ZoriaBlurApp({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: state.settings,
      builder: (context, _) => RepaintBoundary(
        key: shotKey,
        child: MaterialApp(
          title: S.appName,
          debugShowCheckedModeBanner: false,
          theme: buildTheme(Brightness.light),
          darkTheme: buildTheme(Brightness.dark),
          themeMode: state.settings.themeMode,
          builder: (context, child) {
            _syncCaption(Theme.of(context).brightness == Brightness.dark);
            return child ?? const SizedBox.shrink();
          },
          home: HomeScreen(state: state),
        ),
      ),
    );
  }
}

bool? _captionDark;

void _syncCaption(bool dark) {
  if (_captionDark == dark) return;
  _captionDark = dark;
  windowChannel.invokeMethod<void>('setDark', dark).catchError((_) {});
}
