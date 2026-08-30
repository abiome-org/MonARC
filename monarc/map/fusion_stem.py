"""Trainable fusion stem for DSM and vector rasters (not RGB)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monarc.map.dino_backbone import DINOV2_B_EMBED, DINOV2_B_PATCH

GEO_DIM = 128
FUSED_DIM = 256
DEFAULT_VECTOR_CHANNELS = 4


class FusionStem(nn.Module):
    """Encode elevation and vector masks onto the DINO patch grid.

    RGB never enters this module. The stem downsamples by the ViT patch size
    (14) so ``f_geo`` aligns with ``f_rgb``.
    """

    def __init__(
        self,
        vector_channels: int = DEFAULT_VECTOR_CHANNELS,
        geo_dim: int = GEO_DIM,
        patch_size: int = DINOV2_B_PATCH,
    ) -> None:
        super().__init__()
        if vector_channels < 1:
            raise ValueError("vector_channels must be >= 1")
        in_ch = 1 + vector_channels
        self.vector_channels = vector_channels
        self.geo_dim = geo_dim
        self.patch_size = patch_size
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, geo_dim, kernel_size=1),
        )

    def forward(self, dsm: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if dsm.ndim != 4 or dsm.shape[1] != 1:
            raise ValueError(f"DSM must be [B, 1, H, W], got {tuple(dsm.shape)}")
        if vectors.ndim != 4 or vectors.shape[1] != self.vector_channels:
            raise ValueError(
                f"vectors must be [B, {self.vector_channels}, H, W], got {tuple(vectors.shape)}"
            )
        if dsm.shape[-2:] != vectors.shape[-2:]:
            raise ValueError("DSM and vector rasters must share H, W")
        stacked = torch.cat([dsm, vectors], dim=1)
        geo = self.body(stacked)
        _, _, height, width = geo.shape
        out_h = height // self.patch_size
        out_w = width // self.patch_size
        if out_h < 1 or out_w < 1:
            raise ValueError("input smaller than one DINO patch")
        return F.adaptive_avg_pool2d(geo, (out_h, out_w))


class ChannelFusion(nn.Module):
    """Concatenate frozen RGB tokens with geo tokens, then mix to ``FUSED_DIM``."""

    def __init__(
        self,
        rgb_dim: int = DINOV2_B_EMBED,
        geo_dim: int = GEO_DIM,
        fused_dim: int = FUSED_DIM,
    ) -> None:
        super().__init__()
        self.rgb_dim = rgb_dim
        self.geo_dim = geo_dim
        self.fused_dim = fused_dim
        self.mix = nn.Sequential(
            nn.Conv2d(rgb_dim + geo_dim, fused_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fused_dim, fused_dim, kernel_size=1),
        )

    def forward(self, f_rgb: torch.Tensor, f_geo: torch.Tensor) -> torch.Tensor:
        if f_rgb.shape[0] != f_geo.shape[0] or f_rgb.shape[-2:] != f_geo.shape[-2:]:
            raise ValueError(
                f"RGB/geo grids must match batch and spatial size, got {tuple(f_rgb.shape)} vs {tuple(f_geo.shape)}"
            )
        if f_rgb.shape[1] != self.rgb_dim:
            raise ValueError(f"expected RGB dim {self.rgb_dim}, got {f_rgb.shape[1]}")
        if f_geo.shape[1] != self.geo_dim:
            raise ValueError(f"expected geo dim {self.geo_dim}, got {f_geo.shape[1]}")
        return self.mix(torch.cat([f_rgb, f_geo], dim=1))
