# ============================================================
# build-engines.ps1 - builds TensorRT engines (fp16, static shapes)
# for Real-ESRGAN models on YOUR GPU (RTX 3080 / other NVIDIA).
# Engines are cached: existing .engine files are skipped.
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-engines.ps1
#   powershell -ExecutionPolicy Bypass -File build-engines.ps1 -Only RealESRGAN_x4plus
#   powershell -ExecutionPolicy Bypass -File build-engines.ps1 -Rebuild
# ============================================================
param(
    [string]$BaseDir = $(if ($env:AIVIDEOENHANCER_HOME) { $env:AIVIDEOENHANCER_HOME } else { 'C:\AIVideoEnhancer' }),
    [string[]]$Only = @(),
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
$EnginesDir = Join-Path $BaseDir 'engines'
$ModelsDir  = Join-Path $BaseDir 'models'
$JsonPath   = Join-Path $BaseDir 'engine\engines.json'
$CachePath  = Join-Path $EnginesDir 'timing.cache'

$trtexec = Get-ChildItem -Path (Join-Path $BaseDir 'vsmlrt') -Recurse -Filter 'trtexec.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $trtexec) {
    Write-Host "[ERROR] trtexec.exe not found under $BaseDir\vsmlrt - run install.ps1 first." -ForegroundColor Red
    exit 1
}
$trtexec = $trtexec.FullName

if (-not (Test-Path $JsonPath)) { Write-Host "[ERROR] engines.json not found: $JsonPath" -ForegroundColor Red; exit 1 }
$engines = Get-Content $JsonPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $EnginesDir | Out-Null

Write-Host '============================================='
Write-Host ' Building TensorRT engines (fp16, static shapes)'
Write-Host " GPU: $((Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1).Name)"
Write-Host '============================================='

$failed = @()
foreach ($e in $engines) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $e.model)) { continue }
    $enginePath = Join-Path $EnginesDir ('{0}_1x3x{1}x{2}_fp16.engine' -f $e.model, $e.h, $e.w)
    $onnxPath   = Join-Path $ModelsDir  ($e.model + '.onnx')

    if ((Test-Path $enginePath) -and (-not $Rebuild)) {
        Write-Host ('[SKIP] {0} (already built)' -f (Split-Path $enginePath -Leaf)) -ForegroundColor DarkGray
        continue
    }
    if (-not (Test-Path $onnxPath)) {
        Write-Host ('[ERROR] ONNX missing: {0}' -f $onnxPath) -ForegroundColor Red
        $failed += $e.model
        continue
    }

    $shape = 'input:1x3x{0}x{1}' -f $e.h, $e.w
    Write-Host ('[BUILD] {0}  shape {1}  (RRDB models may take 5-15 min...)' -f (Split-Path $enginePath -Leaf), $shape) -ForegroundColor Cyan
    $t0 = Get-Date
    & $trtexec --onnx="$onnxPath" --fp16 `
        "--minShapes=$shape" "--optShapes=$shape" "--maxShapes=$shape" `
        "--saveEngine=$enginePath" --skipInference "--timingCacheFile=$CachePath" 2>&1 |
        Select-String -Pattern 'Engine built|Error|error|failed|Supported' | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $enginePath)) {
        Write-Host ('[ERROR] build failed: {0}' -f (Split-Path $enginePath -Leaf)) -ForegroundColor Red
        $failed += $e.model
        continue
    }
    $dt = [math]::Round(((Get-Date) - $t0).TotalSeconds)
    $sz = [math]::Round((Get-Item $enginePath).Length / 1MB)
    Write-Host ('[OK] {0}  built in {1}s, {2} MB' -f (Split-Path $enginePath -Leaf), $dt, $sz) -ForegroundColor Green
}

Write-Host ''
if ($failed.Count -gt 0) {
    Write-Host ('Finished with ERRORS for: {0}' -f ($failed -join ', ')) -ForegroundColor Red
    Write-Host 'Check NVIDIA driver version (needs 525+ for CUDA 12.9) and free disk/VRAM.'
    exit 1
}
Write-Host 'All engines ready. You can play: AI processing is active by default in mpv.' -ForegroundColor Green
