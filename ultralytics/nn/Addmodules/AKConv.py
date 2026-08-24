"""Compatibility exports for archived standalone DSCPConv checkpoints."""

from ultralytics.nn.rfgac_modules import (
    AKBottleneck,
    AKConv,
    C2f,
    C3k2_DCSPConv,
    ChannelShuffle,
    Conv,
    DCSPC3,
    DCSPC3k,
    DCSPBottleneck,
    DSConvBlock,
    autopad,
)

Bottleneck_DCSP = DCSPBottleneck
Bottleneck = AKBottleneck
C3 = DCSPC3
C3k = DCSPC3k

__all__ = (
    "autopad",
    "Conv",
    "ChannelShuffle",
    "DSConvBlock",
    "AKConv",
    "Bottleneck",
    "Bottleneck_DCSP",
    "C2f",
    "C3",
    "C3k",
    "C3k2_DCSPConv",
)
