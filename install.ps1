# ==================================================================
# install.ps1 - one-click installer for "AI Video Enhancer"
# (honest Real-ESRGAN real-time upscaling: mpv + VapourSynth + TensorRT)
#
# What it does:
#   1. Checks NVIDIA driver (CUDA 12.9 needs 525+)
#   2. Installs Python 3.12 (winget) if missing        [VapourSynth requirement]
#   3. Installs VapourSynth R81 (silent) if missing
#   4. Downloads vs-mlrt v15.13 (TensorRT 10.13 bundle) -> %BaseDir%\vsmlrt
#   5. Downloads Real-ESRGAN ONNX models               -> %BaseDir%\models
#   6. Builds 8 TensorRT fp16 engines for your GPU     -> %BaseDir%\engines
#   7. Downloads portable mpv (VapourSynth-enabled)    -> %BaseDir%\player\mpv
#   8. Writes mpv configs, sets AIVIDEOENHANCER_HOME, creates shortcut
#   9. Runs a quick benchmark
#
# Usage (from elevated PowerShell):
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -BaseDir D:\AIVideoEnhancer
# ==================================================================
param(
    [string]$BaseDir = $(if ($env:AIVIDEOENHANCER_HOME) { $env:AIVIDEOENHANCER_HOME } else { 'C:\AIVideoEnhancer' }),
    [switch]$NoMpv,
    [switch]$NoShortcut,
    [switch]$SkipEngines
)

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# ---------- constants ----------
$VS_VERSION    = 'R81'
$VS_URL        = 'https://github.com/vapoursynth/vapoursynth/releases/download/R81/VapourSynth-x64-R81.exe'
$MLRT_VERSION  = 'v15.13'
$MLRT_URL      = 'https://github.com/AmusementClub/vs-mlrt/releases/download/v15.13/vsmlrt-windows-x64-tensorrt.v15.13.7z'
$SEVENZ_URL    = 'https://www.7-zip.org/a/7zr.exe'
$MPV_REPO_API  = 'https://api.github.com/repos/zhongfly/mpv-winbuild/releases/latest'
$MPV_FALLBACK  = 'https://github.com/zhongfly/mpv-winbuild/releases/download/2026-10-20-e471496c56/mpv-x86_64-v3-20261020-git-e471496c56.7z'
$MODELS_TAG    = 'models-v1'
$MODELS_REPO   = 'https://github.com/ketaevweb/ai-video-enhancer/releases/download/' + $MODELS_TAG
$MODELS        = @('RealESRGAN_x4plus','RealESRGAN_x2plus','realesr-general-x4v3','realesr-general-wdn-x4v3','realesr-animevideov3')
$PY_URL        = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step($msg)   { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)     { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn2($msg)  { Write-Host "    [!]  $msg" -ForegroundColor Yellow }

function Invoke-Download($url, $out) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & curl.exe -L --fail --retry 3 -o $out $url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    }
}

# ---------- 0. sanity ----------
Write-Host '====================================================='
Write-Host ' AI Video Enhancer installer (Real-ESRGAN realtime)'
Write-Host '====================================================='

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Write-Warn2 'Not running as Administrator - VapourSynth/machine env may fail. Right-click -> Run as administrator.' }

New-Item -ItemType Directory -Force -Path $BaseDir, (Join-Path $BaseDir 'engine'), (Join-Path $BaseDir 'models'), (Join-Path $BaseDir 'engines'), (Join-Path $BaseDir 'logs') | Out-Null

# ---------- 1. GPU / driver ----------
Write-Step 'Checking NVIDIA GPU and driver'
try {
    $smi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    if ($smi) {
        $drv = ($smi -split ',')[1].Trim()
        $major = [int]($drv.Split('.')[0])
        Write-Ok ("GPU: {0}, driver {1}" -f $smi)
        if ($major -lt 525) { Write-Warn2 "Driver $drv is OLD. TensorRT/CUDA 12.9 needs driver 525+. UPDATE YOUR NVIDIA DRIVER or engines will fail to build/run." }
    } else { Write-Warn2 'nvidia-smi not found - is the NVIDIA driver installed?' }
} catch { Write-Warn2 'Could not query NVIDIA driver.' }

# ---------- 2. Python ----------
Write-Step 'Checking Python 3.12+ (required by VapourSynth)'
$pyOk = $false
foreach ($c in @('py', 'python')) {
    try {
        $v = & $c -3.12 -V 2>$null
        if (-not $v) { $v = & $c -V 2>$null }
        if ($v -match 'Python 3\.(12|13|14)') { $pyOk = $true; Write-Ok "$v found ($c)"; break }
    } catch {}
}
if (-not $pyOk) {
    Write-Host '    Installing Python 3.12 (this is silent, takes ~2 min)...'
    $haveWinget = Get-Command winget -ErrorAction SilentlyContinue
    if ($haveWinget) {
        & winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    } else {
        $pySetup = Join-Path $env:TEMP 'python312.exe'
        Invoke-Download $PY_URL $pySetup
        Start-Process -Wait -FilePath $pySetup -ArgumentList '/quiet','InstallAllUsers=1','PrependPath=1','Include_test=0'
    }
    $pyOk = $true
    Write-Ok 'Python installed'
}

# ---------- 3. VapourSynth ----------
Write-Step 'Checking VapourSynth'
function Find-VS {
    @(
        (Get-Command vspipe -ErrorAction SilentlyContinue).Source,
        (Join-Path $env:ProgramFiles 'VapourSynth\bin\vspipe.exe'),
        (Join-Path $env:ProgramFiles 'VapourSynth\vspipe.exe')
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
$vspipe = Find-VS
if (-not $vspipe) {
    Write-Host "    Installing VapourSynth $VS_VERSION (silent)..."
    $vsSetup = Join-Path $env:TEMP ("VapourSynth-x64-$VS_VERSION.exe")
    Invoke-Download $VS_URL $vsSetup
    Start-Process -Wait -FilePath $vsSetup -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART'
    $vspipe = Find-VS
    if (-not $vspipe) {
        Write-Warn2 'Silent install may have failed - running installer interactively, complete it in the GUI.'
        Start-Process -Wait -FilePath $vsSetup
        $vspipe = Find-VS
    }
    if (-not $vspipe) { Write-Host '[ERROR] VapourSynth still not found. Reboot and re-run install.ps1.' -ForegroundColor Red; exit 1 }
}
Write-Ok "VapourSynth: $vspipe"

# ---------- 4. vs-mlrt (TensorRT runtime) ----------
Write-Step "Downloading/extracting vs-mlrt $MLRT_VERSION (TensorRT bundle, ~1.8 GB)..."
$vstrtTarget = Join-Path $BaseDir 'vsmlrt\vstrt.dll'
if (Test-Path $vstrtTarget) {
    Write-Ok 'vs-mlrt already present'
} else {
    $arc = Join-Path $BaseDir 'vsmlrt.7z'
    if (-not (Test-Path $arc)) { Invoke-Download $MLRT_URL $arc }
    $sevenz = Join-Path $BaseDir 'tools\7zr.exe'
    New-Item -ItemType Directory -Force -Path (Join-Path $BaseDir 'tools') | Out-Null
    if (-not (Test-Path $sevenz)) { Invoke-Download $SEVENZ_URL $sevenz }
    Write-Host '    Extracting (this takes a few minutes)...'
    $src = Join-Path $BaseDir 'vsmlrt_src'
    & $sevenz x $arc ("-o$src") -y -aoa | Out-Null
    if ($LASTEXITCODE -ne 0) { throw '7z extraction failed (archive damaged? delete vsmlrt.7z and retry)' }
    New-Item -ItemType Directory -Force -Path (Join-Path $BaseDir 'vsmlrt') | Out-Null
    # flatten: all DLLs + vstrt.dll + trtexec.exe into $BaseDir\vsmlrt
    Get-ChildItem -Path $src -Recurse -Include *.dll | ForEach-Object { Copy-Item $_.FullName (Join-Path $BaseDir 'vsmlrt') -Force }
    $vstrtSrc  = Get-ChildItem -Path $src -Recurse -Filter 'vstrt.dll'    | Select-Object -First 1
    $trtSrc    = Get-ChildItem -Path $src -Recurse -Filter 'trtexec.exe' | Select-Object -First 1
    if (-not $vstrtSrc -or -not $trtSrc) { throw 'vstrt.dll/trtexec.exe not found in the vs-mlrt archive' }
    Copy-Item $vstrtSrc.FullName (Join-Path $BaseDir 'vsmlrt\vstrt.dll') -Force
    Copy-Item $trtSrc.FullName   (Join-Path $BaseDir 'vsmlrt\trtexec.exe') -Force
    Remove-Item -Recurse -Force $src, $arc -ErrorAction SilentlyContinue
    Write-Ok ('vs-mlrt ready: {0}' -f (Join-Path $BaseDir 'vsmlrt'))
}

# ---------- 5. engine code + models ----------
Write-Step 'Installing engine scripts and ONNX models'
Copy-Item (Join-Path $RepoDir 'engine\*') (Join-Path $BaseDir 'engine') -Recurse -Force
foreach ($m in $MODELS) {
    $dst = Join-Path $BaseDir "models\$m.onnx"
    if (Test-Path $dst) { Write-Ok "model $m exists"; continue }
    # prefer models shipped next to the repo (offline install)
    $local = Join-Path $RepoDir "models\$m.onnx"
    if (Test-Path $local) { Copy-Item $local $dst -Force; Write-Ok "model $m copied from repo"; continue }
    Invoke-Download "$MODELS_REPO/$m.onnx" $dst
    Write-Ok "model $m downloaded"
}

# ---------- 6. TensorRT engines ----------
if ($SkipEngines) {
    Write-Warn2 'Skipping engine build (-SkipEngines). Run build-engines.ps1 later!'
} else {
    Write-Step 'Building TensorRT engines (one-time, RRDB models take 5-15 min each)...'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoDir 'build-engines.ps1') -BaseDir $BaseDir
    if ($LASTEXITCODE -ne 0) { Write-Host '[ERROR] Engine build failed - see messages above.' -ForegroundColor Red; exit 1 }
}

# ---------- 7. portable mpv ----------
if ($NoMpv) {
    Write-Warn2 'Skipping mpv (-NoMpv)'
} else {
    Write-Step 'Installing portable mpv (VapourSynth-enabled build)...'
    $mpvDir = Join-Path $BaseDir 'player\mpv'
    if (Test-Path (Join-Path $mpvDir 'mpv.exe')) {
        Write-Ok 'mpv already present'
    } else {
        $mpvUrl = $MPV_FALLBACK
        try {
            $rel = Invoke-RestMethod -Uri $MPV_REPO_API -Headers @{ 'User-Agent' = 'ai-video-enhancer' }
            $asset = $rel.assets | Where-Object { $_.name -match '^mpv-x86_64-v3-\d+-git-.*\.7z$' } | Select-Object -First 1
            if ($asset) { $mpvUrl = $asset.browser_download_url }
        } catch { Write-Warn2 'GitHub API failed, using pinned mpv build' }
        $mArc = Join-Path $BaseDir 'mpv.7z'
        Invoke-Download $mpvUrl $mArc
        New-Item -ItemType Directory -Force -Path $mpvDir | Out-Null
        & (Join-Path $BaseDir 'tools\7zr.exe') x $mArc ("-o$mpvDir") -y -aoa | Out-Null
        Remove-Item $mArc -Force
        Write-Ok ('mpv: {0}' -f (Join-Path $mpvDir 'mpv.exe'))
    }
    # configs
    $cfg = Join-Path $mpvDir 'portable_config'
    New-Item -ItemType Directory -Force -Path $cfg | Out-Null
    $baseFwd = $BaseDir.Replace('\', '/')
    foreach ($f in @('mpv.conf', 'input.conf')) {
        (Get-Content (Join-Path $RepoDir "player\$f") -Raw -Encoding UTF8).Replace('%%BASE%%', $baseFwd) |
            Set-Content (Join-Path $cfg $f) -Encoding UTF8
    }
    Write-Ok 'mpv configs written (AI enabled by default)'
}

# ---------- 8. environment + shortcut ----------
Write-Step 'Setting environment and shortcuts'
[Environment]::SetEnvironmentVariable('AIVIDEOENHANCER_HOME', $BaseDir, 'Machine')
$env:AIVIDEOENHANCER_HOME = $BaseDir
Write-Ok ('AIVIDEOENHANCER_HOME = {0}' -f $BaseDir)

if (-not $NoShortcut -and (Test-Path (Join-Path $BaseDir 'player\mpv\mpv.exe'))) {
    try {
        $ws = New-Object -ComObject WScript.Shell
        $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'AI Video Player (Real-ESRGAN).lnk'))
        $lnk.TargetPath = Join-Path $BaseDir 'player\mpv\mpv.exe'
        $lnk.WorkingDirectory = Join-Path $BaseDir 'player\mpv'
        $lnk.IconLocation = Join-Path $BaseDir 'player\mpv\mpv.exe,0'
        $lnk.Save()
        Write-Ok 'Desktop shortcut created'
    } catch { Write-Warn2 'Shortcut creation failed (not critical)' }
}

# ---------- 9. smoke test ----------
Write-Step 'Smoke test: rendering 24 frames through the MAX preset (SD tier)'
$env:AIV_BENCH_TIER = 'sd'; $env:AIV_BENCH_PRESET = 'max'
try {
    $test = & $vspipe (Join-Path $BaseDir 'engine\bench.vpy') - --start 0 --end 23 2>&1 | Out-String
    $m = [regex]::Match($test, '\(([\d\.]+) fps\)')
    if ($m.Success) {
        Write-Ok ("Smoke test passed: {0} fps in MAX preset" -f $m.Groups[1].Value)
    } else {
        Write-Host $test
        Write-Warn2 'Smoke test did not report fps - check messages above.'
    }
} catch { Write-Warn2 ("Smoke test failed: {0}" -f $_.Exception.Message) }

# ---------- done ----------
Write-Host ''
Write-Host '=====================================================' -ForegroundColor Green
Write-Host ' INSTALLATION COMPLETE' -ForegroundColor Green
Write-Host '=====================================================' -ForegroundColor Green
Write-Host (' Player:  {0}' -f (Join-Path $BaseDir 'player\mpv\mpv.exe'))
Write-Host ' Hotkeys:  Ctrl+1 max | Ctrl+2 fast | Ctrl+3 denoise | Ctrl+4 anime | Ctrl+0 off'
Write-Host (' Bench:    powershell -File "{0}"' -f (Join-Path $RepoDir 'bench.ps1'))
Write-Host ' Just open any 480p/720p video - AI works automatically.'
