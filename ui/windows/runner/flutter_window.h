#ifndef RUNNER_FLUTTER_WINDOW_H_
#define RUNNER_FLUTTER_WINDOW_H_

#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <flutter/method_channel.h>

#include <memory>

#include "win32_window.h"

// Single instance (main.cpp): window property of the first instance and the WM_COPYDATA tag a second
// launch uses to hand over a video path (UTF-16, NUL-terminated).
inline constexpr wchar_t kMainProp[] = L"ZoriaBlurMain";
inline constexpr ULONG_PTR kOpenFileCopyData = 0x5A424F50;  // 'ZBOP'

// A window that does nothing but host a Flutter view.
class FlutterWindow : public Win32Window {
 public:
  // Creates a new FlutterWindow hosting a Flutter view running |project|.
  explicit FlutterWindow(const flutter::DartProject& project);
  virtual ~FlutterWindow();

 protected:
  // Win32Window:
  bool OnCreate() override;
  void OnDestroy() override;
  LRESULT MessageHandler(HWND window, UINT const message, WPARAM const wparam,
                         LPARAM const lparam) noexcept override;

 private:
  // The project to run.
  flutter::DartProject project_;

  // The Flutter instance hosted by this window.
  std::unique_ptr<flutter::FlutterViewController> flutter_controller_;

  // zoria_blur/instance: "open" <path> when a second launch forwards a video.
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> instance_channel_;
  // Flutter -> runner: "setDark" (caption colour follows the app theme)
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> window_channel_;
};

#endif  // RUNNER_FLUTTER_WINDOW_H_
