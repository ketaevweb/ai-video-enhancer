# ============================================================
# bench.ps1 - measures AI processing FPS via vspipe
# Usage:
#   powershell -ExecutionPolicy Bypass -File bench.ps1                    # all presets, SD tier
#   powershell -ExecutionPolicy Bypass -File bench.ps1 -Tier hd          # HD tier
#   powershell -ExecutionPolicy Bypass -File bench.ps1 -Preset max       # one preset
# ============================================================
param(
    [string]$BaseDir = $(if ($env:AIVIDEOENHANCER_HOME) { $env:AIVIDEOENHANCER_HOME } else { 'C:\AIVideoEnhancer' }),
    [ValidateSet('sd','hd')] [string]$Tier = 'sd',
    [ValidateSet('max','fast','denoise','anime','all')] [string]$Preset = 'all',
    [int]$Frames = 120
)

$ErrorActionPreference = 'Stop'
$vspipe = @(
    (Get-Command vspipe -ErrorAction SilentlyContinue).Source,
    (Join-Path $env:ProgramFiles 'VapourSynth\bin\vspipe.exe'),
    (Join-Path $env:ProgramFiles 'VapourSynth\vspipe.exe')
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $vspipe) { Write-Host '[ERROR] vspipe not found - install VapourSynth (run install.ps1).' -ForegroundColor Red; exit 1 }

$presets = if ($Preset -eq 'all') { @('max','fast','denoise','anime') } else { @($Preset) }
$modelNames = @{ max = 'RealESRGAN'; fast = 'general-x4v3'; denoise = 'general-wdn'; anime = 'animevideov3' }

Write-Host '====================================================='
Write-Host (' Bench: tier={0}, {1} frames' -f $Tier.ToUpper(), $Frames)
Write-Host '====================================================='

foreach ($p in $presets) {
    $env:AIV_BENCH_TIER = $Tier
    $env:AIV_BENCH_PRESET = $p
    $out = & $vspipe (Join-Path $BaseDir 'engine\bench.vpy') - --start 0 --end ($Frames - 1) 2>&1 | Out-String
    $m = [regex]::Match($out, 'Output (\d+) frames in ([\d\.]+) seconds \(([\d\.]+) fps\)')
    if ($m.Success) {
        $fps = [double]$m.Groups[3].Value
        $rt  = if ($fps -ge 48) { 'REAL-TIME (48+ fps, 2x margin)' }
               elseif ($fps -ge 24) { 'real-time OK for 24fps films' }
               elseif ($fps -ge 18) { 'borderline (may stutter on 24fps)' }
               else { 'NOT real-time - use a faster preset' }
        Write-Host (' {0,-8} ({1,-13}): {2,7:F2} fps  -> {3}' -f $p, $modelNames[$p], $fps, $rt) -ForegroundColor $(if ($fps -ge 24) { 'Green' } else { 'Yellow' })
    } else {
        Write-Host (' {0,-8}: FAILED' -f $p) -ForegroundColor Red
        Write-Host $out
    }
}
Write-Host ''
Write-Host 'Tip: 24 fps = cinema. If "max" is below 24 fps use "fast" (Ctrl+2 in mpv).'
