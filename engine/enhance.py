# -*- coding: utf-8 -*-
"""
AI Video Enhancer - "честный" Real-ESRGAN в реальном времени (VapourSynth + TensorRT).

Пайплайн:
  YUV (исходник) -> [масштаб в канонический кадр ИИ] -> RGBS -> паддинг до формы
  движка -> core.trt.Model (Real-ESRGAN, fp16, статический TensorRT-движок) ->
  кроп паддинга -> [масштаб в целевое разрешение] -> YUV-выход.

Канонические формы движка (engines.json):
  SD-тир: 960x540  (для источников с высотой <= 576)
  HD-тир: 1280x720 (для источников с высотой 577..720)

Соотношение сторон сохраняется через display-размер (SAR/DAR): кадр вписывается
(cover->contain) в канонический бокс, свободные поля добиваются чёрным и
вырезаются после инференса, затем кадр масштабируется в целевую высоту 1440
с точным DAR. Для 16:9 это ровно 2560x1440 (2K-монитор).

Публичные функции:
  enhance(clip, preset="auto", dw=0, dh=0, target_h=1440)
  enhance_mpv(video_in, preset, dw, dh)  - обёртка для фильтра mpv --vf=vapoursynth
"""

import json
import os
import sys

import vapoursynth as vs
from vapoursynth import core

# --------------------------------------------------------------------------
# Базовый каталог установки. Можно переопределить переменной окружения.
BASE = os.environ.get("AIVIDEOENHANCER_HOME", r"C:\AIVideoEnhancer")
ENGINE_DIR = os.path.join(BASE, "engine")
ENGINES_DIR = os.path.join(BASE, "engines")
VSMLRT_DLL = os.path.join(BASE, "vsmlrt", "vsmlrt-cuda", "vstrt.dll")

# Канонические входные формы движков (совпадают с engines.json)
CANON_SD = (960, 540)    # (w, h) вход ИИ для SD-тира
CANON_HD = (1280, 720)   # (w, h) вход ИИ для HD-тира

# Пресет -> модель. SD = источники <= 576, HD = 577..720.
PRESET_SD = {
    "max":     "RealESRGAN_x4plus",        # максимум детализации (RRDB, тяжёлый)
    "fast":    "realesr-general-x4v3",     # компактный, заметно быстрее
    "denoise": "realesr-general-wdn-x4v3", # с сильным шумоподавлением
    "anime":   "realesr-animevideov3",     # для аниме/мультипликации
}
PRESET_HD = {
    "max":     "RealESRGAN_x2plus",        # честный RRDB x2 -> ровно 1440p
    "fast":    "realesr-general-x4v3",
    "denoise": "realesr-general-wdn-x4v3",
    "anime":   "realesr-animevideov3",
}

TARGET_H = 1440          # целевая высота выхода (2K-монитор, 2560x1440)
NUM_STREAMS = 2          # параллельные CUDA-стримы в vstrt
USE_CUDA_GRAPH = True    # снижает CPU-оверхед инференса


def _load_engines_json():
    path = os.path.join(ENGINE_DIR, "engines.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {(e["model"], e["h"], e["w"]): e for e in json.load(f)}
    except Exception:
        # минимальный фолбэк, если json не читается
        return {
            ("RealESRGAN_x4plus", 540, 960): {"scale": 4},
            ("RealESRGAN_x2plus", 720, 1280): {"scale": 2},
            ("realesr-general-x4v3", 540, 960): {"scale": 4},
            ("realesr-general-x4v3", 720, 1280): {"scale": 4},
            ("realesr-general-wdn-x4v3", 540, 960): {"scale": 4},
            ("realesr-general-wdn-x4v3", 720, 1280): {"scale": 4},
            ("realesr-animevideov3", 540, 960): {"scale": 4},
            ("realesr-animevideov3", 720, 1280): {"scale": 4},
        }


_ENGINES = _load_engines_json()


def _round_even(x: float) -> int:
    """Округление до чётного целого (требование планарных YUV-форматов)."""
    return max(2, int(round(x / 2.0)) * 2)


def _get_prop(clip, name, default):
    try:
        f = clip.get_frame(0)
        if name in f.props and f.props[name] is not None:
            return f.props[name]
    except Exception:
        pass
    return default


def _ensure_trt():
    """Подгружает vstrt.dll, если он ещё не зарегистрирован в VapourSynth."""
    if hasattr(core, "trt"):
        return
    if not os.path.exists(VSMLRT_DLL):
        raise RuntimeError(
            "vstrt.dll не найден: {}\n"
            "Запустите install.ps1 (шаг vs-mlrt) ещё раз.".format(VSMLRT_DLL)
        )
    core.std.LoadPlugin(VSMLRT_DLL)


def _pick_preset(preset: str, h: int):
    """Возвращает (тир, имя_модели) либо (None, None) если ИИ не применяется."""
    preset = (preset or "auto").strip().lower()
    if preset in ("off", "none", ""):
        return None, None
    if h <= 576:
        tier = "sd"
    elif h <= 720:
        tier = "hd"
    else:
        return None, None  # источник уже >= 720p+: без ИИ по умолчанию
    table = PRESET_SD if tier == "sd" else PRESET_HD
    if preset == "auto":
        preset = "max"          # "auto" = максимум (пользователь просил максимум)
    if preset not in table:
        preset = "max" if preset == "max" else "fast"
    return tier, table[preset]


def _matrix_code(clip):
    """Определяет матрицу YUV->RGB: по проперти _Matrix, иначе эвристика."""
    m = _get_prop(clip, "_Matrix", None)
    if m in (1, 5, 6, 4, 7):
        return int(m)
    return 6 if clip.height <= 576 else 1   # SD -> BT.601, HD -> BT.709


def _ai_engine_call(rgb, model_name, cw, ch):
    """Прогон RGBS-кадра через TensorRT-движок Real-ESRGAN."""
    _ensure_trt()
    entry = _ENGINES.get((model_name, ch, cw))
    scale = int(entry["scale"]) if entry else 4
    engine = os.path.join(ENGINES_DIR,
                          "{}_1x3x{}x{}_fp16.engine".format(model_name, ch, cw))
    if not os.path.exists(engine):
        raise RuntimeError(
            "Движок не найден: {}\n"
            "Соберите его: PowerShell -> build-engines.ps1 "
            "(или запустите install.ps1 целиком).".format(engine)
        )
    out = core.trt.Model(rgb, engine_path=engine,
                         use_cuda_graph=USE_CUDA_GRAPH,
                         num_streams=NUM_STREAMS,
                         verbosity=1)
    return out, scale


def enhance(clip, preset="auto", dw=0, dh=0, target_h=TARGET_H):
    """
    Главный вход. clip - любой 8/10-битный YUV/RGB клип VapourSynth (SDR).
    dw/dh - display-размер (SAR-коррекция); 0 = взять из проперти SAR.
    """
    if clip.format.color_family not in (vs.YUV, vs.RGB, vs.GRAY):
        raise RuntimeError("Формат не поддерживается: " + str(clip.format.color_family))

    tier, model_name = _pick_preset(preset, clip.height)
    if tier is None:
        return clip  # пресет off или источник >= 720p - без изменений

    cw, ch = CANON_SD if tier == "sd" else CANON_HD

    # Display-размер (анаморфные DVD и т.п.)
    if dw <= 0 or dh <= 0:
        sar_n = int(_get_prop(clip, "_SARNum", 0) or 0)
        sar_d = int(_get_prop(clip, "_SARDen", 0) or 0)
        if sar_n > 0 and sar_d > 0:
            dw = clip.width * sar_n / float(sar_d)
            dh = float(clip.height)
        else:
            dw, dh = float(clip.width), float(clip.height)

    # 1) Вписываем display-кадр в канонический бокс (contain), чётные размеры
    scale = min(cw / dw, ch / dh)
    tw = _round_even(dw * scale)
    th = _round_even(dh * scale)
    if tw > cw:      # страховка от двойного округления
        tw = cw
    if th > ch:
        th = ch

    # 2) Масштаб YUV до входа ИИ (хрома апсемплится здесь же, корректно)
    pre = core.resize.Bicubic(clip, tw, th)

    # 3) YUV -> RGBS (float 0..1), с правильной матрицей и диапазоном
    matrix = _matrix_code(clip)
    rgb = core.resize.Bicubic(pre, format=vs.RGBS, matrix_in=matrix)

    # 4) Паддинг до канонической формы движка
    px, py = (cw - tw) // 2, (ch - th) // 2
    if px or py:
        rgb = core.std.AddBorders(rgb, left=px, right=px, top=py, bottom=py)

    # 5) Инференс
    up, n_scale = _ai_engine_call(rgb, model_name, cw, ch)

    # 6) Кроп паддинга (масштабированного) и возврат в YUV целевого размера
    if px or py:
        up = core.std.CropAbs(up, tw * n_scale, th * n_scale, px * n_scale, py * n_scale)

    out_w = _round_even(target_h * dw / dh)
    out_fmt = vs.YUV420P10 if clip.format.bits_per_sample > 8 else vs.YUV420P8
    out = core.resize.Bicubic(
        up, out_w, target_h,
        format=out_fmt, matrix=matrix,
        range=1, dither_type="error_diffusion",
    )
    return out


def enhance_mpv(video_in, preset="auto", dw=0, dh=0):
    """Обёртка для mpv: возвращает клип с сохранёнными таймингами."""
    return enhance(video_in, preset=preset, dw=dw or 0, dh=dh or 0)
