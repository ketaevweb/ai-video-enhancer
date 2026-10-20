# Third-party components

| Component | License | Source |
|---|---|---|
| mpv (portable build by zhongfly/mpv-winbuild) | GPLv2+ | https://github.com/zhongfly/mpv-winbuild |
| VapourSynth R81 | LGPL-2.1+ | https://github.com/vapoursynth/vapoursynth |
| vs-mlrt v15.13 (vstrt plugin + NVIDIA TensorRT 10.13 / CUDA 12.9 runtime) | MIT (plugin), NVIDIA SDK License (TensorRT) | https://github.com/AmusementClub/vs-mlrt |
| Real-ESRGAN models (x4plus, x2plus, general-x4v3, wdn-x4v3, animevideov3) | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| 7zr.exe (7-Zip standalone) | LGPL / unRAR restriction | https://www.7-zip.org |
| RRDBNet / SRVGGNet architectures in tools/convert_pth_to_onnx.py | BSD-3-Clause (basicsr / realesrgan) | https://github.com/xinntao/Real-ESRGAN |

Real-ESRGAN ONNX models in the release are converted from the official `.pth`
checkpoints without any weight modification.
