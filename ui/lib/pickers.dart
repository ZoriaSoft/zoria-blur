import 'package:file_selector/file_selector.dart';

import 'app_state.dart';
import 'strings.dart';

/// The open dialog shows video files only (one filter, no "all files").
const videoTypeGroup = XTypeGroup(label: S.videoFiles, extensions: videoExtensions);

Future<String?> pickVideoFile({String? initialDirectory}) async {
  final f = await openFile(acceptedTypeGroups: const [videoTypeGroup], initialDirectory: initialDirectory);
  return f?.path;
}
