# Builds dist\ZoriaBlur\ (ZoriaBlur.exe + engine\zoria-blur-engine.exe + models + ffmpeg).
# Run from the repo root on Windows:  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# Needs: .venv-dml (requirements with onnxruntime-directml + pyinstaller), bin\ffmpeg.exe + ffprobe.exe (+ DLLs
# of a shared build), models\*.onnx, Flutter on PATH (or $env:FLUTTER_ROOT\bin), Inno Setup 6 for the installer step
# ($env:ISCC or the default install folders).
# -SkipInstaller: stop after dist\ZoriaBlur.  -InstallerOnly: only (re)build dist\ZoriaBlur-Setup-<version>.exe
# from the existing dist\ZoriaBlur.
# Optional NVIDIA engine: if build\engine_cuda_dist\zoria-blur-engine (packaging\build_cuda_engine.ps1, or -CudaEngine
# DIR) exists it is copied to dist\ZoriaBlur\engine-cuda (also with -InstallerOnly); -NoCuda leaves it out.
param([switch]$SkipEngine, [switch]$SkipUi, [switch]$SkipInstaller, [switch]$InstallerOnly, [switch]$NoCuda,
      [string]$CudaEngine = '')
$ErrorActionPreference = 'Continue'   # native tools write progress to stderr; failures are checked via $LASTEXITCODE
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
$py = "$root\.venv-dml\Scripts\python.exe"
$dist = "$root\dist\ZoriaBlur"
if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) { if ($env:FLUTTER_ROOT) { $env:Path = "$env:FLUTTER_ROOT\bin;" + $env:Path } }

if ($InstallerOnly) { $SkipEngine = $true; $SkipUi = $true }

if (-not $SkipEngine) {
  # only the models the app uses (yolox_s is a CLI-only option)
  $stage = "$root\build\engine_stage"
  Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force "$stage\models" | Out-Null
  foreach ($m in 'det_10g', 'det_500m', 'w600k_mbf', 'yolox_tiny', 'yolox_nano') { Copy-Item "$root\models\$m.onnx" "$stage\models\" }
  & $py -m PyInstaller --noconfirm --clean --onedir --windowed -n zoria-blur-engine `
      --distpath "$root\build\engine_dist" --workpath "$root\build\pyi" --specpath "$root\build" `
      --paths $root --collect-binaries onnxruntime --exclude-module tkinter --exclude-module matplotlib --exclude-module PIL --exclude-module scipy `
      --add-data "$stage\models;models" `
      "$root\packaging\engine_entry.py"
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
  # ffmpeg goes in as plain files (PyInstaller's DLL analysis would copy its DLLs a second time);
  # OpenCV's own ffmpeg plugin is not used (video goes through ffmpeg pipes)
  $int = "$root\build\engine_dist\zoria-blur-engine\_internal"
  New-Item -ItemType Directory -Force "$int\bin" | Out-Null
  Copy-Item "$root\bin\ffmpeg.exe", "$root\bin\ffprobe.exe", "$root\bin\*.dll", "$root\bin\FFMPEG_LICENSE.txt" "$int\bin\" -ErrorAction SilentlyContinue
  Remove-Item "$int\cv2\opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue
}

if (-not $SkipUi) {
  Push-Location "$root\ui"
  flutter build windows --release
  if ($LASTEXITCODE -ne 0) { Pop-Location; throw "flutter build failed ($LASTEXITCODE)" }
  Pop-Location
}

if (-not $InstallerOnly) {
  Remove-Item -Recurse -Force $dist -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $dist | Out-Null
  Copy-Item -Recurse "$root\ui\build\windows\x64\runner\Release\*" $dist
  Copy-Item -Recurse "$root\build\engine_dist\zoria-blur-engine" "$dist\engine"
}
# optional CUDA engine (borrows models + ffmpeg from engine\ at run time, see zoria_blur\paths.py)
if (-not $CudaEngine) { $CudaEngine = "$root\build\engine_cuda_dist\zoria-blur-engine" }
Remove-Item -Recurse -Force "$dist\engine-cuda" -ErrorAction SilentlyContinue
if ((-not $NoCuda) -and (Test-Path "$CudaEngine\zoria-blur-engine.exe")) {
  Copy-Item -Recurse $CudaEngine "$dist\engine-cuda"
  "{0:N0} MB  {1}" -f ((Get-ChildItem -Recurse "$dist\engine-cuda" | Measure-Object Length -Sum).Sum / 1MB), "$dist\engine-cuda (CUDA engine)"
} else { "no CUDA engine in this build" }
# license notes next to ZoriaBlur.exe (also refreshed for -InstallerOnly)
Copy-Item "$root\packaging\THIRD_PARTY.txt" $dist -ErrorAction SilentlyContinue
Copy-Item "$root\bin\FFMPEG_LICENSE.txt" $dist -ErrorAction SilentlyContinue
"{0:N0} MB  {1}" -f ((Get-ChildItem -Recurse $dist | Measure-Object Length -Sum).Sum / 1MB), "$dist\ZoriaBlur.exe"

if (-not $SkipInstaller) {
  # two one-file installers (version = ui\pubspec.yaml, e.g. 0.1.0+1 -> 0.1.0):
  #   dist\ZoriaBlur-Setup-<version>.exe         light: DirectML (any card) + CPU, ~1/7 of the size
  #   dist\ZoriaBlur-Setup-<version>-NVIDIA.exe  also the CUDA engine (only when this build has one)
  # The app tells an NVIDIA user with the light one where the NVIDIA one is (system check advice).
  if (-not (Test-Path "$dist\ZoriaBlur.exe")) { throw "no $dist\ZoriaBlur.exe - build the app first" }
  $iscc = @($env:ISCC, "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") |
          Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
  if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found; set `$env:ISCC" }
  $ver = ((Select-String -Path "$root\ui\pubspec.yaml" -Pattern '^version:\s*(\S+)').Matches[0].Groups[1].Value)
  $short = $ver.Split('+')[0]
  if (Test-Path "$dist\engine-cuda\zoria-blur-engine.exe") {
    & $iscc /Q "/DAppVersion=$ver" "/DSourceDir=$dist" "/DFlavor=-NVIDIA" "$root\packaging\installer.iss"
    if ($LASTEXITCODE -ne 0) { throw "ISCC (NVIDIA) failed ($LASTEXITCODE)" }
  }
  # light: the same files without engine-cuda (moved aside for the compile, always moved back)
  $aside = "$root\build\engine-cuda-aside"
  Remove-Item -Recurse -Force $aside -ErrorAction SilentlyContinue
  if (Test-Path "$dist\engine-cuda") { Move-Item "$dist\engine-cuda" $aside }
  try {
    & $iscc /Q "/DAppVersion=$ver" "/DSourceDir=$dist" "$root\packaging\installer.iss"
    if ($LASTEXITCODE -ne 0) { throw "ISCC (light) failed ($LASTEXITCODE)" }
  } finally {
    if (Test-Path $aside) { Move-Item $aside "$dist\engine-cuda" }
  }
  Get-ChildItem "$root\dist\ZoriaBlur-Setup-$short*.exe" | ForEach-Object { "{0:N0} MB  {1}" -f ($_.Length / 1MB), $_.FullName }
}
