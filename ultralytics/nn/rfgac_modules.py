"""RFGAC-YOLO layers used by the submitted model configuration.

This file preserves the executable implementation found in the authors'
archived Ultralytics 8.3.0 workspace.  Keep it synchronized with the model
configuration and manuscript before creating a public release.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = (
    "C3k2_DCSPConv",
    "C3k2_RFCAConv_GAM",
    "RFCAConv_GAM",
    "SDI",
    "ShuffledDSConv",
)


def autopad(k, p=None, d=1):
    """Return padding that preserves the spatial shape for stride 1."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Convolution followed by batch normalization and SiLU."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ChannelShuffle(nn.Module):
    """Shuffle channels across groups when the channel count is divisible."""

    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        groups = min(self.groups, channels)
        if channels % groups:
            groups = 1
        x = x.view(batch, groups, channels // groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)


class ShuffledDSConv(nn.Module):
    """Depthwise convolution, channel shuffle, and pointwise projection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=4):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
            padding_mode="replicate" if kernel_size == 7 else "zeros",
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(groups)
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.shuffle(x)
        return self.act(self.bn2(self.pwconv(x)))


class DSConvBlock(nn.Module):
    """Seven-by-seven depthwise-shuffle block used inside the archived L-GAM."""

    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
            bias=False,
            padding_mode="replicate",
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(groups)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.shuffle(x)
        return self.bn2(self.pw(x))


class AKConv(nn.Module):
    """Adaptive-kernel layer retained for the archived DSCP ablation path."""

    def __init__(self, inc, outc, num_param=2, stride=1, bias=None):
        super().__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU(),
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        """Compatibility hook retained for checkpoints serialized by the archive."""
        del module, grad_input, grad_output
        return None

    def forward(self, x):
        offset = self.p_conv(x)
        points = offset.size(1) // 2
        positions = self._get_positions(offset, points)
        lower_left = positions.detach().floor()
        upper_right = lower_left + 1
        lower_left = self._clamp_positions(lower_left, x, points).long()
        upper_right = self._clamp_positions(upper_right, x, points).long()
        lower_right = torch.cat((lower_left[..., :points], upper_right[..., points:]), dim=-1)
        upper_left = torch.cat((upper_right[..., :points], lower_left[..., points:]), dim=-1)
        positions = self._clamp_positions(positions, x, points)

        weight_ll = (1 + lower_left[..., :points] - positions[..., :points]) * (
            1 + lower_left[..., points:] - positions[..., points:]
        )
        weight_ur = (1 - upper_right[..., :points] + positions[..., :points]) * (
            1 - upper_right[..., points:] + positions[..., points:]
        )
        weight_lr = (1 + lower_right[..., :points] - positions[..., :points]) * (
            1 - lower_right[..., points:] + positions[..., points:]
        )
        weight_ul = (1 - upper_left[..., :points] + positions[..., :points]) * (
            1 + upper_left[..., points:] - positions[..., points:]
        )
        sampled = (
            weight_ll.unsqueeze(1) * self._gather(x, lower_left, points)
            + weight_ur.unsqueeze(1) * self._gather(x, upper_right, points)
            + weight_lr.unsqueeze(1) * self._gather(x, lower_right, points)
            + weight_ul.unsqueeze(1) * self._gather(x, upper_left, points)
        )
        sampled = rearrange(sampled, "b c h w n -> b c (h n) w")
        return self.conv(sampled)

    def _get_positions(self, offset, points):
        _, _, height, width = offset.shape
        base_size = round(math.sqrt(points))
        rows, remainder = divmod(points, base_size)
        grid_x, grid_y = torch.meshgrid(
            torch.arange(rows, device=offset.device),
            torch.arange(base_size, device=offset.device),
            indexing="ij",
        )
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
        if remainder:
            extra_x = torch.full((remainder,), rows, device=offset.device)
            extra_y = torch.arange(remainder, device=offset.device)
            grid_x, grid_y = torch.cat((grid_x, extra_x)), torch.cat((grid_y, extra_y))
        kernel_grid = torch.cat((grid_x, grid_y)).view(1, 2 * points, 1, 1).to(offset.dtype)
        base_x, base_y = torch.meshgrid(
            torch.arange(0, height * self.stride, self.stride, device=offset.device),
            torch.arange(0, width * self.stride, self.stride, device=offset.device),
            indexing="ij",
        )
        base = torch.cat(
            (
                base_x.reshape(1, 1, height, width).repeat(1, points, 1, 1),
                base_y.reshape(1, 1, height, width).repeat(1, points, 1, 1),
            ),
            dim=1,
        ).to(offset.dtype)
        return (base + kernel_grid + offset).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _clamp_positions(positions, x, points):
        return torch.cat(
            (
                positions[..., :points].clamp(0, x.size(2) - 1),
                positions[..., points:].clamp(0, x.size(3) - 1),
            ),
            dim=-1,
        )

    @staticmethod
    def _gather(x, positions, points):
        batch, height, width, _ = positions.shape
        channels = x.size(1)
        indices = positions[..., :points] * x.size(3) + positions[..., points:]
        indices = indices.unsqueeze(1).expand(-1, channels, -1, -1, -1).reshape(batch, channels, -1)
        return x.reshape(batch, channels, -1).gather(-1, indices).reshape(batch, channels, height, width, points)


class AKBottleneck(nn.Module):
    """Adaptive-kernel bottleneck used by the archived nested C3k branch."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        del g
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = AKConv(hidden, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class DCSPBottleneck(nn.Module):
    """Bottleneck used by the archived standalone DSCPConv ablation."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = DSConvBlock(hidden, c2, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class GAM(nn.Module):
    """Lightweight channel-spatial attention from the archived implementation."""

    def __init__(self, in_channels, rate=4):
        super().__init__()
        mid_channels = in_channels // rate
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dsconv1 = DSConvBlock(in_channels, mid_channels)
        self.dsconv2 = DSConvBlock(mid_channels, in_channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_perm = x.permute(0, 2, 3, 1).view(batch, -1, channels)
        channel_att = self.channel_att(x_perm).view(batch, height, width, channels).permute(0, 3, 1, 2)
        x = x * channel_att
        spatial_att = self.relu(self.dsconv1(x))
        spatial_att = self.sigmoid(self.dsconv2(spatial_att))
        return x * spatial_att


class RFCAConv_GAM(nn.Module):
    """Receptive-field expansion followed by lightweight global attention."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, reduction=4, norm_type="group"):
        super().__init__()
        del reduction, norm_type
        self.kernel_size = kernel_size
        self.generate = nn.Conv2d(
            in_channels,
            in_channels * kernel_size**2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        expanded_channels = in_channels * kernel_size**2
        groups = min(32, expanded_channels)
        while expanded_channels % groups:
            groups //= 2
        self.norm = nn.GroupNorm(groups, expanded_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gam = GAM(in_channels)
        # These archived layers are retained for checkpoint compatibility even
        # though the recorded forward path does not call them.
        self.conv_gate = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.w_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1, bias=True),
        )
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        feat = self.relu(self.norm(self.generate(x)))
        feat = rearrange(
            feat,
            "b (c k1 k2) h w -> b c (h k1) (w k2)",
            c=channels,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )
        return self.conv_out(self.gam(feat) + feat)


class RFGACBottleneck(nn.Module):
    """Bottleneck with YAML-controlled residual behavior."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        del g
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = RFCAConv_GAM(hidden, c2, 3, 1)
        self.add = shortcut and c1 == c2
        self.shortcut_conv = nn.Conv2d(c1, c2, 1, stride=1) if self.add else None

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="nearest")
        return out + self.shortcut_conv(x) if self.add else out


class C2f(nn.Module):
    """Minimal C2f implementation used by the custom block."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(RFGACBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """Minimal CSP block used by the archived C3k selector path."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(
            *(RFGACBottleneck(hidden, hidden, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 path with the archived two-block RFGAC bottleneck structure."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(
            *(RFGACBottleneck(hidden, hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_RFCAConv_GAM(C2f):
    """C3k2-compatible wrapper used by the final archived YAML."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else RFGACBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k2_DCSPConv(C2f):
    """C3k2 wrapper used for the standalone DSCPConv ablation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            DCSPC3k(self.c, self.c, 2, shortcut, g) if c3k else DCSPBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class DCSPC3(C3):
    """C3 branch retained exactly for the archived mixed DSCP/AK ablation."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(AKBottleneck(hidden, hidden, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class DCSPC3k(DCSPC3):
    """Nested C3k selector branch from the archived DSCP ablation code."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(AKBottleneck(hidden, hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class SDI(nn.Module):
    """Align multi-scale features and combine them by Hadamard product."""

    def __init__(self, channel):
        super().__init__()
        self.dsconvs = nn.ModuleList(ShuffledDSConv(c, channel[0], kernel_size=3) for c in channel)

    def forward(self, xs):
        answer = torch.ones_like(xs[0])
        target_size = xs[0].shape[-2:]
        for index, x in enumerate(xs):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
            answer = answer * self.dsconvs[index](x)
        return answer
