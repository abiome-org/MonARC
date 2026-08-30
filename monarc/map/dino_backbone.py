"""Frozen DINOv2-B/14 RGB backbone.

Official DINOv2 weights are never downloaded. Tests and the dry-run CLI use a
frozen patch-14 768-d stub that matches the DINOv2-B tensor contract
``[B, 3, H, W] -> [B, 768, H/14, W/14]``. A local weights file may be supplied
later; this module does not call ``torch.hub`` or Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_B_EMBED = 768
DINOV2_B_PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradients and set eval mode on every parameter."""
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()
    return module


class FrozenDinoBackbone(nn.Module):
    """Frozen RGB encoder with the DINOv2-B/14 spatial contract.

    ``mode="stub"`` is a single stride-14 convolution. It exists so CPU tests and
    the dry-run CLI can extract patch tokens without network weight downloads.
    ``mode="vitb14"`` is reserved for a local official checkpoint; constructing
    it without ``weights_path`` raises.
    """

    def __init__(
        self,
        mode: str = "stub",
        embed_dim: int = DINOV2_B_EMBED,
        patch_size: int = DINOV2_B_PATCH,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"stub", "vitb14"}:
            raise ValueError(f"unsupported backbone mode: {mode}")
        if mode == "vitb14" and weights_path is None:
            raise ValueError(
                "vitb14 requires a local DINOv2-B weights file; "
                "this package does not download checkpoints"
            )
        self.mode = mode
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        if weights_path is not None:
            self._load_local_patch(Path(weights_path))
        freeze_module(self)

    def _load_local_patch(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "patch_embed.proj.weight" in payload:
            self.patch.weight.copy_(payload["patch_embed.proj.weight"])
            if "patch_embed.proj.bias" in payload and self.patch.bias is not None:
                self.patch.bias.copy_(payload["patch_embed.proj.bias"])
        elif isinstance(payload, dict) and "weight" in payload:
            self.patch.load_state_dict(payload)
        else:
            raise ValueError(f"unrecognized local DINO weight format: {path}")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB tensor must be [B, 3, H, W], got {tuple(rgb.shape)}")
        _, _, height, width = rgb.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"spatial size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        x = (rgb - self.mean) / self.std
        return self.patch(x)


def interpolate_pos_grid(feat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear resize of a feature grid to ``(height, width)``."""
    return F.interpolate(feat, size=(height, width), mode="bilinear", align_corners=False)
