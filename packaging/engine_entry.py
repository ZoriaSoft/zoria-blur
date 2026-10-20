"""PyInstaller entry point for zoria-blur-engine.exe (absolute import; __main__.py uses a relative one).

The exe is a windowed (GUI) build: an exception that escapes main() would make PyInstaller's bootloader show a modal
"Unhandled exception in script" box on the user's desktop. Nothing may escape: it goes to
%LOCALAPPDATA%\\ZoriaBlur\\logs\\engine_crash.txt and the process exits with code 1."""
import os
import sys


def _log_crash():
    try:
        import time
        import traceback
        d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ZoriaBlur", "logs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "engine_crash.txt"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}\n{traceback.format_exc()}\n")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    try:
        from zoria_blur.cli import main
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:  # noqa: BLE001
        _log_crash()
        code = 1
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        os._exit(code or 0)  # stdout already closed by the app: leave without the interpreter's final flush
    sys.exit(code or 0)
