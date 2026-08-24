"""Compatibility implementation for archived checkpoints that reference DySample."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("Dy_Sample",)


def _normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight, mean, std)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def _constant_init(module, value, bias=0.0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, value)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


class Dy_Sample(nn.Module):
    """Dynamic upsampler retained only for archived-checkpoint compatibility."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        if style not in {"lp", "pl"}:
            raise ValueError("style must be 'lp' or 'pl'")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2):
            raise ValueError("pixel-first mode requires channels divisible by scale squared")
        if in_channels < groups or in_channels % groups:
            raise ValueError("input channels must be divisible by groups")

        offset_in = in_channels // scale**2 if style == "pl" else in_channels
        offset_out = 2 * groups if style == "pl" else 2 * groups * scale**2
        self.offset = nn.Conv2d(offset_in, offset_out, 1)
        _normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(offset_in, offset_out, 1)
            _constant_init(self.scope, 0.0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        axis = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((yy, xx)).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coords_w, coords_h = torch.meshgrid(coords_w, coords_h, indexing="ij")
        coords = torch.stack((coords_w, coords_h)).transpose(1, 2).unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([width, height], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(batch, -1, height, width), self.scale)
        coords = coords.view(batch, 2, -1, self.scale * height, self.scale * width)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(batch, -1, self.scale * height, self.scale * width)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        shuffled = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = self.offset(shuffled) * self.scope(shuffled).sigmoid()
            offset = F.pixel_unshuffle(offset, self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(shuffled), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)
