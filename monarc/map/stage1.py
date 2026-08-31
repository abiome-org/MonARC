"""Stage-1 tiny codebook training on fused tokens (FSQ projection + fusion)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monarc.map.dino_backbone import FrozenDinoBackbone
from monarc.map.fusion_stem import ChannelFusion, FusionStem
from monarc.map.quantizer import FSQHead


class RgbReconHead(nn.Module):
    """Reconstruct frozen DINO tokens from quantized scalars (no class heads)."""

    def __init__(self, fsq_dim: int, rgb_dim: int) -> None:
        super().__init__()
        self.net = nn.Conv2d(fsq_dim, rgb_dim, kernel_size=1)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


def spatial_smoothness(z_hat: torch.Tensor) -> torch.Tensor:
    dh = z_hat[:, :, 1:, :] - z_hat[:, :, :-1, :]
    dw = z_hat[:, :, :, 1:] - z_hat[:, :, :, :-1]
    return dh.pow(2).mean() + dw.pow(2).mean()


def train_tiny_codebook(
    backbone: FrozenDinoBackbone,
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    rgb: torch.Tensor,
    dsm: torch.Tensor,
    vectors: torch.Tensor,
    *,
    steps: int = 8,
    lr: float = 1e-3,
    smooth_weight: float = 0.1,
    device: torch.device | str = "cpu",
) -> dict:
    """Few-step Stage-1 update: fusion + FSQ projection, frozen RGB encoder."""
    device = torch.device(device)
    backbone = backbone.to(device)
    fusion_stem = fusion_stem.to(device)
    channel_fusion = channel_fusion.to(device)
    fsq_head = fsq_head.to(device)
    recon = RgbReconHead(fsq_head.fsq.num_dimensions, backbone.embed_dim).to(device)
    rgb = rgb.to(device)
    dsm = dsm.to(device)
    vectors = vectors.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    fsq_params = list(fsq_head.parameters())
    other = (
        list(fusion_stem.parameters())
        + list(channel_fusion.parameters())
        + list(recon.parameters())
    )
    opt = torch.optim.Adam(
        [
            {"params": other, "lr": lr},
            {"params": fsq_params, "lr": lr * 0.1},
        ]
    )
    losses: list[float] = []
    backbone.eval()
    with torch.no_grad():
        f_rgb = backbone(rgb)
    for _ in range(int(steps)):
        fusion_stem.train()
        channel_fusion.train()
        fsq_head.train()
        recon.train()
        opt.zero_grad(set_to_none=True)
        f_geo = fusion_stem(dsm, vectors)
        fused = channel_fusion(f_rgb, f_geo)
        z_hat, codes = fsq_head(fused)
        recon_rgb = recon(z_hat)
        loss_recon = F.mse_loss(recon_rgb, f_rgb)
        loss_smooth = spatial_smoothness(z_hat)
        loss = loss_recon + smooth_weight * loss_smooth
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    fusion_stem.eval()
    channel_fusion.eval()
    fsq_head.eval()
    with torch.no_grad():
        f_geo = fusion_stem(dsm, vectors)
        fused = channel_fusion(f_rgb, f_geo)
        z_hat, codes = fsq_head(fused)
    used = int(torch.unique(codes).numel())
    return {
        "losses": losses,
        "codes": codes.detach().cpu(),
        "fused": fused.detach().cpu(),
        "z_hat": z_hat.detach().cpu(),
        "f_rgb": f_rgb.detach().cpu(),
        "codebook_size": int(fsq_head.fsq.codebook_size),
        "unique_codes": used,
        "levels": list(fsq_head.fsq.levels),
    }


def encode_fused(
    backbone: FrozenDinoBackbone,
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    rgb: torch.Tensor,
    dsm: torch.Tensor,
    vectors: torch.Tensor,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen-eval encode: DINO RGB + fusion + FSQ codes."""
    device = torch.device(device)
    backbone.eval()
    fusion_stem.eval()
    channel_fusion.eval()
    fsq_head.eval()
    with torch.no_grad():
        f_rgb = backbone(rgb.to(device))
        f_geo = fusion_stem(dsm.to(device), vectors.to(device))
        fused = channel_fusion(f_rgb, f_geo)
        z_hat, codes = fsq_head(fused)
    return f_rgb.cpu(), fused.cpu(), codes.cpu()


def default_stage1_modules(
    vector_channels: int = 4,
    levels: Sequence[int] = (5, 5, 5),
) -> tuple[FrozenDinoBackbone, FusionStem, ChannelFusion, FSQHead]:
    backbone = FrozenDinoBackbone(mode="stub")
    stem = FusionStem(vector_channels=vector_channels)
    mix = ChannelFusion()
    head = FSQHead(levels=levels)
    return backbone, stem, mix, head
