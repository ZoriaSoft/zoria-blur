# Backend benchmark without installing: devices + analyze + render (blur-except, all faces) + QA per backend, run
# from a built dist\ZoriaBlur (CUDA from engine-cuda\, DirectML / CPU from engine\). Results: work\bench\summary.json.
#   powershell -ExecutionPolicy Bypass -File scripts\bench_backends.ps1 -Clip work\clip4.mp4 [-Runs cuda,dml,cpu] [-Dist DIR]
param([string]$Clip = "work\clip4.mp4", [string[]]$Runs = @("cuda", "dml", "cpu"), [string]$Dist = "dist\ZoriaBlur")
Set-Location (Resolve-Path "$PSScriptRoot\..").Path
$d = $Dist
$Runs = @($Runs | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File passes "cuda,dml" as one string
$res = @()
function Eng([string]$exe, [string[]]$a, [string]$tag) {
  $o = "work\bench\$tag.json"; $e = "work\bench\$tag.err"
  $t = Get-Date
  $p = Start-Process $exe -ArgumentList $a -RedirectStandardOutput $o -RedirectStandardError $e -WindowStyle Hidden -Wait -PassThru
  $s = ((Get-Date) - $t).TotalSeconds
  $j = Get-Content $o | Where-Object { $_ -match '^\{' } | Select-Object -Last 1 | ConvertFrom-Json
  return @{ wall = [math]::Round($s, 1); rc = $p.ExitCode; j = $j }
}
New-Item -ItemType Directory -Force work\bench | Out-Null
foreach ($b in $Runs) {
  $exe = if ($b -eq 'cuda') { "$d\engine-cuda\zoria-blur-engine.exe" } else { "$d\engine\zoria-blur-engine.exe" }
  $proj = "work\bench\proj_$b"; Remove-Item -Recurse -Force $proj -ErrorAction SilentlyContinue
  $dv = Eng $exe @('devices') "devices_$b"
  $an = Eng $exe @('--backend', $b, 'analyze', $Clip, '-p', $proj) "analyze_$b"
  $rn = Eng $exe @('--backend', $b, 'render', $Clip, '-p', $proj, '--mode', 'blur-except', '-o', "work\bench\out_$b.mp4") "render_$b"
  $res += [pscustomobject]@{
    backend = $b; devices_chosen = $dv.j.chosen; device_name = $dv.j.device_name; devices_wall = $dv.wall
    probes = ($dv.j.probes | ForEach-Object { "$($_.backend):$($_.ok):$($_.ms)" }) -join ' '
    analyze_rc = $an.rc; analyze_wall = $an.wall; analyze_engine_s = $an.j.timing.seconds_total; analyze_fps = $an.j.timing.video_fps_processed
    persons = @($an.j.persons).Count; analyze_device = ($an.j.device | ConvertTo-Json -Compress -Depth 4)
    render_rc = $rn.rc; render_wall = $rn.wall; render_s = $rn.j.seconds; render_fps = $rn.j.fps
    qa_leaks = $rn.j.qa.leak_count; qa_faces = $rn.j.qa.faces_detected; render_device = ($rn.j.device | ConvertTo-Json -Compress -Depth 4)
    error = "$($an.j.error) $($rn.j.error)".Trim()
  }
}
$res | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 work\bench\summary.json
$res | Format-List
