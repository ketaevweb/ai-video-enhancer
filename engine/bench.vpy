# bench.vpy - бенчмарк инференса (запускается через vspipe, см. bench.ps1)
# Переменные окружения:
#   AIV_BENCH_TIER   = sd | hd   (по умолчанию sd)
#   AIV_BENCH_PRESET = max | fast | denoise | anime (по умолчанию max)
import os
import sys

import vapoursynth as vs
from vapoursynth import core

_BASE = os.environ.get("AIVIDEOENHANCER_HOME", r"C:\AIVideoEnhancer")
sys.path.insert(0, os.path.join(_BASE, "engine"))

from enhance import enhance  # noqa: E402

tier = os.environ.get("AIV_BENCH_TIER", "sd").lower()
preset = os.environ.get("AIV_BENCH_PRESET", "max").lower()

if tier == "hd":
    src = core.std.BlankClip(width=1280, height=720, format=vs.YUV420P8,
                             fpsnum=24000, fpsden=1001, length=100000,
                             color=[114, 128, 122])
else:
    src = core.std.BlankClip(width=854, height=480, format=vs.YUV420P8,
                             fpsnum=24000, fpsden=1001, length=100000,
                             color=[114, 128, 122])

enhance(src, preset=preset).set_output()
