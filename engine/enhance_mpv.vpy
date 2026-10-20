# enhance_mpv.vpy - точка входа для фильтра mpv --vf=vapoursynth
# mpv определяет переменные: video_in, video_in_dw, video_in_dh,
# container_fps, display_fps и user_data (пресет из input.conf).
import os
import sys

try:
    _preset = user_data
except NameError:
    _preset = "auto"
if not _preset:
    _preset = "auto"

_BASE = os.environ.get("AIVIDEOENHANCER_HOME", r"C:\AIVideoEnhancer")
sys.path.insert(0, os.path.join(_BASE, "engine"))

from enhance import enhance_mpv  # noqa: E402

enhance_mpv(video_in, _preset, video_in_dw, video_in_dh).set_output()
