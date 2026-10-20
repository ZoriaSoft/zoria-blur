# Builds the optional NVIDIA engine: build\engine_cuda_dist\zoria-blur-engine\zoria-blur-engine.exe
# (PyInstaller --onedir, onnxruntime-gpu = CUDA + CPU). build_windows.ps1 then copies it to dist\ZoriaBlur\engine-cuda.
# Run from the repo root on Windows:  powershell -ExecutionPolicy Bypass -File packaging\build_cuda_engine.ps1
#   -Python <python 3.12 exe>  used only to create .venv-cuda when it does not exist yet
#
# No system CUDA toolkit is needed: the CUDA 12 / cuDNN 9 runtime DLLs come from NVIDIA's pip wheels and are copied
# next to onnxruntime_providers_cuda.dll (zoria_blur\backend.py points the DLL search there). Only the DLLs the
# engine really loads are shipped (packaging\cuda_dll_probe.py lists them). No models / ffmpeg in this folder:
# the CUDA engine borrows them from the main engine next to it (<app>\engine\_internal, zoria_blur\paths.py).
#
# cuDNN is pinned to 9.10.2.21: cuDNN 9.27 fails every Conv on Turing GPUs (sm_75) with
# "Failed to initialize CUDNN Frontend" (tested 2026-10-14 with onnxruntime-gpu 1.24.4, driver 616.56).
param([string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
$venv = "$root\.venv-cuda"
$py = "$venv\Scripts\python.exe"
$pins = @('numpy==2.5.4', 'opencv-python-headless==5.0.0.93', 'onnxruntime-gpu==1.24.4', 'pyinstaller==6.22.3',
          'nvidia-cuda-runtime-cu12==12.9.79', 'nvidia-cublas-cu12==12.9.2.10', 'nvidia-cufft-cu12==11.4.1.4',
          'nvidia-cudnn-cu12==9.10.2.21')
if (-not (Test-Path $py)) {
  & $Python -m venv $venv
  if ($LASTEXITCODE -ne 0) { throw "venv failed (Python 3.12 at $Python?)" }
}
& $py -m pip install --disable-pip-version-check -q @pins
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# DLLs to ship, by wheel folder (site-packages\nvidia\<pkg>\bin). onnxruntime_providers_cuda.dll imports cudart,
# cublas, cublasLt, cufft and cudnn64_9; cuDNN loads its sub-libraries on demand (graph/ops/heuristic/engines).
# Left out: cudnn_adv (RNN/attention, ~270 MB), cudnn_cnn (legacy conv API), curand, nvrtc, nvJitLink, cufftw, nvblas.
$need = [ordered]@{
  'cuda_runtime' = @('cudart64_12.dll')
  'cublas'       = @('cublas64_12.dll', 'cublasLt64_12.dll')
  'cufft'        = @('cufft64_11.dll')
  'cudnn'        = @('cudnn64_9.dll', 'cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_heuristic64_9.dll',
                     'cudnn_engines_precompiled64_9.dll', 'cudnn_engines_runtime_compiled64_9.dll')
}
$sp = "$venv\Lib\site-packages"

$out = "$root\build\engine_cuda_dist"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
# --collect-binaries onnxruntime brings onnxruntime_providers_cuda/_shared/_tensorrt.dll; the nvidia wheels are not
# imported by the engine, so PyInstaller does not pull them in on its own (copied below, only what is needed)
& $py -m PyInstaller --noconfirm --clean --onedir --windowed -n zoria-blur-engine `
    --distpath $out --workpath "$root\build\pyi-cuda" --specpath "$root\build\cuda-spec" `
    --paths $root --collect-binaries onnxruntime --exclude-module tkinter --exclude-module matplotlib `
    --exclude-module PIL --exclude-module scipy --exclude-module sympy --exclude-module nvidia `
    "$root\packaging\engine_entry.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
$capi = "$out\zoria-blur-engine\_internal\onnxruntime\capi"
if (-not (Test-Path "$capi\onnxruntime_providers_cuda.dll")) { throw "onnxruntime_providers_cuda.dll missing in $capi" }
# TensorRT EP needs nvinfer (not shipped): its provider DLL is dead weight; OpenCV's own ffmpeg plugin is not used
Remove-Item "$capi\onnxruntime_providers_tensorrt.dll" -ErrorAction SilentlyContinue
Remove-Item "$out\zoria-blur-engine\_internal\cv2\opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue
foreach ($pkg in $need.Keys) {
  foreach ($dll in $need[$pkg]) {
    $src = "$sp\nvidia\$pkg\bin\$dll"
    if (-not (Test-Path $src)) { throw "missing $src" }
    Copy-Item $src $capi
  }
}
# NVIDIA runtime license notes next to the DLLs
New-Item -ItemType Directory -Force "$out\zoria-blur-engine\_internal\licenses-nvidia" | Out-Null
foreach ($pkg in $need.Keys) {
  Get-ChildItem "$sp\nvidia_*$($pkg -replace '_','*')*.dist-info\License.txt", "$sp\nvidia_*$($pkg -replace '_','*')*.dist-info\licenses\*" -ErrorAction SilentlyContinue |
    Select-Object -First 1 | ForEach-Object { Copy-Item $_.FullName "$out\zoria-blur-engine\_internal\licenses-nvidia\$pkg-LICENSE.txt" }
}
"{0:N0} MB  {1}" -f ((Get-ChildItem -Recurse "$out\zoria-blur-engine" | Measure-Object Length -Sum).Sum / 1MB), "$out\zoria-blur-engine\zoria-blur-engine.exe"
