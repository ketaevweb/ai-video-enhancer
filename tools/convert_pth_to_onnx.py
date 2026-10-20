#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_pth_to_onnx.py - convert ANY official Real-ESRGAN .pth checkpoint
to ONNX so it can be used with this project (TensorRT engine building).

Based on architectures from the Real-ESRGAN project (BSD-3-Clause):
  https://github.com/xinntao/Real-ESRGAN

Usage:
  python convert_pth_to_onnx.py RealESRGAN_x4plus.pth RealESRGAN_x4plus.onnx

After conversion build an engine (add a row to engine/engines.json first):
  trtexec --onnx=<file>.onnx --fp16 --minShapes=input:1x3x540x960 \
      --optShapes=input:1x3x540x960 --maxShapes=input:1x3x540x960 \
      --saveEngine=<name>_1x3x540x960_fp16.engine --skipInference
"""
import collections
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- RRDBNet (basicsr) -----------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + num_grow_ch * 2, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + num_grow_ch * 3, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + num_grow_ch * 4, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# -------------------------- SRVGGNetCompact ----------------------------------
class SRVGGNetCompact(nn.Module):
    """Exact copy of realesrgan/archs/srvgg_arch.py (official layout)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(t, n):
        if t == "relu":
            return nn.ReLU(inplace=True)
        if t == "leakyrelu":
            return nn.LeakyReLU(0.1, inplace=True)
        return nn.PReLU(num_parameters=n)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out


def build_model(sd):
    if any(k.startswith("body.0.rdb1.") for k in sd):
        in_ch = sd["conv_first.weight"].shape[1]
        num_feat = sd["conv_first.weight"].shape[0]
        scale = {3: 4, 12: 2, 48: 1}[in_ch]
        num_block = len({k.split(".")[1] for k in sd if k.startswith("body.")})
        return RRDBNet(3, sd["conv_last.weight"].shape[0], scale, num_feat, num_block)
    conv4 = [k for k in sd if k.startswith("body.") and k.endswith(".weight") and sd[k].dim() == 4]
    final_idx = max(int(k.split(".")[1]) for k in conv4)
    num_feat = sd["body.0.weight"].shape[0]
    upscale = int(round((sd["body.%d.weight" % final_idx].shape[0] / 3) ** 0.5))
    act = "prelu" if any(sd[k].dim() == 1 for k in sd if k.startswith("body.") and k.endswith(".weight")) else "relu"
    return SRVGGNetCompact(3, 3, num_feat, (final_idx - 2) // 2, upscale, act)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    pth_path, onnx_path = sys.argv[1], sys.argv[2]

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    assert isinstance(sd, collections.abc.Mapping)
    model = build_model(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("loaded:", type(model).__name__,
          "| params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    print("saved:", onnx_path)


if __name__ == "__main__":
    main()
