"""Stage-1 tiny codebook training on fused tokens (FSQ projection + fusion)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monarc.map.dino_backbone import FrozenDinoBackbone
from monarc.map.fusion_stem import ChannelFusion, FusionStem
from monarc.map.quantizer import DEFAULT_FSQ_LEVELS, FSQHead


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


def stage1_loss(
    z: torch.Tensor,
    z_hat_nchw: torch.Tensor,
    target: torch.Tensor,
    recon: RgbReconHead,
    fsq_head: FSQHead,
    *,
    smooth_weight: float,
    usage_weight: float,
) -> torch.Tensor:
    loss = F.mse_loss(recon(z_hat_nchw), target)
    if smooth_weight:
        loss = loss + float(smooth_weight) * spatial_smoothness(z_hat_nchw)
    if usage_weight:
        loss = loss + float(usage_weight) * fsq_head.fsq.usage_loss(z)
    return loss


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
    smooth_weight: float = 0.05,
    usage_weight: float = 1.0,
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
    opt = torch.optim.Adam(
        list(fusion_stem.parameters())
        + list(channel_fusion.parameters())
        + list(fsq_head.parameters())
        + list(recon.parameters()),
        lr=lr,
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
        z = fsq_head.project(fused)
        z_hat_hw, codes = fsq_head.fsq(z)
        z_hat = z_hat_hw.permute(0, 3, 1, 2).contiguous()
        loss = stage1_loss(
            z,
            z_hat,
            f_rgb,
            recon,
            fsq_head,
            smooth_weight=smooth_weight,
            usage_weight=usage_weight,
        )
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


def _geo_from_optional(
    fusion_stem: FusionStem,
    f_rgb: torch.Tensor,
    dsm: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> torch.Tensor:
    if dsm is None:
        return torch.zeros(
            f_rgb.shape[0],
            fusion_stem.geo_dim,
            f_rgb.shape[2],
            f_rgb.shape[3],
            device=f_rgb.device,
            dtype=f_rgb.dtype,
        )
    if vectors is None:
        vectors = torch.zeros(
            dsm.shape[0],
            fusion_stem.vector_channels,
            dsm.shape[2],
            dsm.shape[3],
            device=dsm.device,
            dtype=dsm.dtype,
        )
    return fusion_stem(dsm, vectors)


def encode_from_features(
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    f_rgb: torch.Tensor,
    dsm: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """FSQ codes from cached DINO grids. DSM/vectors optional."""
    device = torch.device(device)
    fusion_stem.eval()
    channel_fusion.eval()
    fsq_head.eval()
    with torch.no_grad():
        f_rgb = f_rgb.to(device)
        dsm_t = dsm.to(device) if dsm is not None else None
        vec_t = vectors.to(device) if vectors is not None else None
        f_geo = _geo_from_optional(fusion_stem, f_rgb, dsm_t, vec_t)
        fused = channel_fusion(f_rgb, f_geo)
        _z_hat, codes = fsq_head(fused)
    return fused.cpu(), codes.cpu()


def train_from_features(
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    f_rgb: torch.Tensor,
    dsm: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
    *,
    steps: int = 8,
    lr: float = 1e-3,
    smooth_weight: float = 0.05,
    usage_weight: float = 1.0,
    batch_size: int = 4,
    device: torch.device | str = "cpu",
    recon: RgbReconHead | None = None,
    start_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[..., None] | None = None,
) -> dict:
    """Stage-1 FSQ/fusion updates on cached frozen DINO tokens (no RGB re-encode)."""
    device = torch.device(device)
    fusion_stem = fusion_stem.to(device)
    channel_fusion = channel_fusion.to(device)
    fsq_head = fsq_head.to(device)
    if recon is None:
        recon = RgbReconHead(fsq_head.fsq.num_dimensions, int(f_rgb.shape[1]))
    recon = recon.to(device)
    n = int(f_rgb.shape[0])
    if n < 1:
        raise ValueError("feature cache is empty")
    batch_size = max(1, min(int(batch_size), n))
    if optimizer is None:
        optimizer = torch.optim.Adam(
            list(fusion_stem.parameters())
            + list(channel_fusion.parameters())
            + list(fsq_head.parameters())
            + list(recon.parameters()),
            lr=lr,
        )
    losses: list[float] = []
    last_codes: torch.Tensor | None = None
    last_fused: torch.Tensor | None = None
    step = int(start_step)
    target = step + int(steps)
    while step < target:
        fusion_stem.train()
        channel_fusion.train()
        fsq_head.train()
        recon.train()
        perm = torch.randperm(n)[:batch_size]
        rgb_b = f_rgb[perm].to(device=device, dtype=torch.float32)
        dsm_b = dsm[perm].to(device=device, dtype=torch.float32) if dsm is not None else None
        vec_b = (
            vectors[perm].to(device=device, dtype=torch.float32) if vectors is not None else None
        )
        optimizer.zero_grad(set_to_none=True)
        f_geo = _geo_from_optional(fusion_stem, rgb_b, dsm_b, vec_b)
        fused = channel_fusion(rgb_b, f_geo)
        z = fsq_head.project(fused)
        z_hat_hw, codes = fsq_head.fsq(z)
        z_hat = z_hat_hw.permute(0, 3, 1, 2).contiguous()
        loss = stage1_loss(
            z,
            z_hat,
            rgb_b,
            recon,
            fsq_head,
            smooth_weight=smooth_weight,
            usage_weight=usage_weight,
        )
        loss.backward()
        optimizer.step()
        loss_v = float(loss.detach().cpu())
        losses.append(loss_v)
        last_codes = codes.detach()
        last_fused = fused.detach()
        step += 1
        if on_step is not None:
            on_step(step, loss_v, optimizer, recon)
    fusion_stem.eval()
    channel_fusion.eval()
    fsq_head.eval()
    code_chunks: list[torch.Tensor] = []
    fused_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            rgb_b = f_rgb[sl].to(device=device, dtype=torch.float32)
            dsm_b = dsm[sl].to(device=device, dtype=torch.float32) if dsm is not None else None
            vec_b = (
                vectors[sl].to(device=device, dtype=torch.float32) if vectors is not None else None
            )
            f_geo = _geo_from_optional(fusion_stem, rgb_b, dsm_b, vec_b)
            fused_b = channel_fusion(rgb_b, f_geo)
            _z_hat, codes_b = fsq_head(fused_b)
            code_chunks.append(codes_b.detach().cpu())
            fused_chunks.append(fused_b.detach().cpu())
    codes = torch.cat(code_chunks, dim=0)
    fused = torch.cat(fused_chunks, dim=0)
    last_codes = codes
    last_fused = fused
    used = int(torch.unique(codes).numel())
    n_tokens = int(codes.numel())
    return {
        "losses": losses,
        "codes": last_codes.detach().cpu() if last_codes is not None else codes.detach().cpu(),
        "fused": last_fused.detach().cpu() if last_fused is not None else fused.detach().cpu(),
        "codebook_size": int(fsq_head.fsq.codebook_size),
        "unique_codes": used,
        "n_tokens": n_tokens,
        "n_chips": n,
        "levels": list(fsq_head.fsq.levels),
        "step": step,
        "recon": recon,
        "optimizer": optimizer,
    }


def stage1_checkpoint(
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    *,
    step: int,
    recon: RgbReconHead | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "step": int(step),
        "fusion_stem": fusion_stem.state_dict(),
        "channel_fusion": channel_fusion.state_dict(),
        "fsq_head": fsq_head.state_dict(),
        "levels": list(fsq_head.fsq.levels),
    }
    if recon is not None:
        payload["recon"] = recon.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    return payload


def load_stage1_checkpoint(
    path: str,
    fusion_stem: FusionStem,
    channel_fusion: ChannelFusion,
    fsq_head: FSQHead,
    *,
    recon: RgbReconHead | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    fusion_stem.load_state_dict(payload["fusion_stem"])
    channel_fusion.load_state_dict(payload["channel_fusion"])
    fsq_head.load_state_dict(payload["fsq_head"])
    if recon is not None and "recon" in payload:
        recon.load_state_dict(payload["recon"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def default_stage1_modules(
    vector_channels: int = 4,
    levels: Sequence[int] = DEFAULT_FSQ_LEVELS,
    backbone: FrozenDinoBackbone | None = None,
) -> tuple[FrozenDinoBackbone, FusionStem, ChannelFusion, FSQHead]:
    if backbone is None:
        backbone = FrozenDinoBackbone(mode="stub")
    stem = FusionStem(vector_channels=vector_channels)
    mix = ChannelFusion()
    head = FSQHead(levels=levels)
    return backbone, stem, mix, head
