"""Perspective encoder: frozen RGB backbone plus a lightweight map-space adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from monarc.map.dino_backbone import DINOV2_B_EMBED, FrozenDinoBackbone
from monarc.map.quantizer import FSQHead
from monarc.perspective.calibrator import TemperatureCalibrator
from monarc.perspective.slot_head import LandmarkSlots, emit_slots


class PerspectiveAdapter(nn.Module):
    """Project perspective DINO tokens into the fused/FSQ map space."""

    def __init__(self, in_dim: int = DINOV2_B_EMBED, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, f_rgb: torch.Tensor) -> torch.Tensor:
        return self.net(f_rgb)


class PerspectiveEncoder(nn.Module):
    """Onboard pixel path: frozen DINO -> adapter -> FSQ -> calibrated slots.

    Images stop here. Downstream localization consumes only slot tuples.
    """

    def __init__(
        self,
        backbone: FrozenDinoBackbone,
        adapter: PerspectiveAdapter,
        fsq_head: FSQHead,
        calibrator: TemperatureCalibrator | None = None,
        top_k: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.fsq_head = fsq_head
        self.calibrator = calibrator or TemperatureCalibrator()
        self.top_k = top_k

    def forward(self, rgb: torch.Tensor) -> LandmarkSlots:
        f_rgb = self.backbone(rgb)
        aligned = self.adapter(f_rgb)
        z_hat, codes = self.fsq_head(aligned)
        conf = self.calibrator(z_hat)
        return emit_slots(codes, conf, patch_size=self.backbone.patch_size, top_k=self.top_k)
