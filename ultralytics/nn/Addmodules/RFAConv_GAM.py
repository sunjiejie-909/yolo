"""Compatibility exports for archived RFGAC-YOLO checkpoints."""

from ultralytics.nn.rfgac_modules import (
    C2f,
    C3,
    C3k,
    C3k2_RFCAConv_GAM,
    ChannelShuffle,
    Conv,
    DSConvBlock,
    GAM,
    RFCAConv_GAM,
    RFGACBottleneck,
    autopad,
)

Bottleneck = RFGACBottleneck

__all__ = (
    "autopad",
    "Conv",
    "ChannelShuffle",
    "DSConvBlock",
    "GAM",
    "RFCAConv_GAM",
    "Bottleneck",
    "C2f",
    "C3",
    "C3k",
    "C3k2_RFCAConv_GAM",
)
