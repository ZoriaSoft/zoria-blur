#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include <shellapi.h>

#include <cwchar>
#include <string>

#include "flutter_window.h"
#include "utils.h"

namespace {

// Single instance: the first Zoria Blur owns this mutex and marks its window with kMainProp. A second
// launch brings that window to the front (restored if minimized), hands over its video path (if any)
// with WM_COPYDATA and exits.
constexpr wchar_t kMutexName[] = L"Local\\ZoriaSoft.ZoriaBlur.SingleInstance";

BOOL CALLBACK FindMain(HWND hwnd, LPARAM out) {
  if (::GetPropW(hwnd, kMainProp) != nullptr) {
    *reinterpret_cast<HWND*>(out) = hwnd;
    return FALSE;
  }
  return TRUE;
}

std::wstring FirstFileArgument() {
  int argc = 0;
  wchar_t** argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
  std::wstring path;
  if (argv != nullptr) {
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != L'-') {
        path = argv[i];
        break;
      }
    }
    ::LocalFree(argv);
  }
  return path;
}

void ActivateFirstInstance() {
  HWND main = nullptr;
  // the first instance may still be starting: wait up to 5 s for its window
  for (int i = 0; i < 50 && main == nullptr; ++i) {
    ::EnumWindows(FindMain, reinterpret_cast<LPARAM>(&main));
    if (main == nullptr) ::Sleep(100);
  }
  if (main == nullptr) return;
  if (::IsIconic(main)) {
    ::ShowWindow(main, SW_RESTORE);
  } else {
    ::ShowWindow(main, SW_SHOW);
  }
  ::SetForegroundWindow(main);
  const std::wstring path = FirstFileArgument();
  if (!path.empty()) {
    COPYDATASTRUCT cds{};
    cds.dwData = kOpenFileCopyData;
    cds.cbData = static_cast<DWORD>((path.size() + 1) * sizeof(wchar_t));
    cds.lpData = const_cast<wchar_t*>(path.c_str());
    DWORD_PTR result = 0;
    ::SendMessageTimeoutW(main, WM_COPYDATA, 0, reinterpret_cast<LPARAM>(&cds), SMTO_ABORTIFHUNG, 3000, &result);
  }
}

}  // namespace

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  HANDLE instance_mutex = ::CreateMutexW(nullptr, TRUE, kMutexName);
  if (instance_mutex != nullptr && ::GetLastError() == ERROR_ALREADY_EXISTS) {
    ActivateFirstInstance();
    ::CloseHandle(instance_mutex);
    return EXIT_SUCCESS;
  }

  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(1360, 860);
  if (!window.Create(L"Zoria Blur", origin, size)) {
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);
  ::SetPropW(window.GetHandle(), kMainProp, reinterpret_cast<HANDLE>(1));
  // Self-test runs (ZORIA_AUTOPILOT_VIDEO) take their own screenshots: keep the window off screen and
  // unfocused so a build machine's user is not disturbed.
  if (::GetEnvironmentVariableW(L"ZORIA_AUTOPILOT_VIDEO", nullptr, 0) > 0) {
    // ZORIA_AUTOPILOT_SIZE=1440x900: logical client size of the self-test window (screenshots)
    int w = 0, h = 0;
    wchar_t sz[32] = {};
    if (::GetEnvironmentVariableW(L"ZORIA_AUTOPILOT_SIZE", sz, 32) > 0 && swscanf_s(sz, L"%dx%d", &w, &h) == 2 && w > 0 &&
        h > 0) {
      const double scale = ::GetDpiForWindow(window.GetHandle()) / 96.0;
      RECT rc{0, 0, static_cast<LONG>(w * scale), static_cast<LONG>(h * scale)};
      ::AdjustWindowRectEx(&rc, WS_OVERLAPPEDWINDOW, FALSE, 0);
      ::SetWindowPos(window.GetHandle(), nullptr, -6000, 40, rc.right - rc.left, rc.bottom - rc.top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
    } else {
      ::SetWindowPos(window.GetHandle(), nullptr, -6000, 40, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
  }

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  if (instance_mutex != nullptr) ::CloseHandle(instance_mutex);
  return EXIT_SUCCESS;
}
