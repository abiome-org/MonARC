"""Sparse landmark slot emission from an FSQ code grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LandmarkSlots:
    """Sparse ``(code, u, v, confidence)`` observations from one frame."""

    codes: torch.Tensor
    uv: torch.Tensor
    confidence: torch.Tensor
    batch_index: torch.Tensor

    def to_numpy(self) -> tuple:
        return (
            self.codes.detach().cpu().numpy(),
            self.uv.detach().cpu().numpy(),
            self.confidence.detach().cpu().numpy(),
            self.batch_index.detach().cpu().numpy(),
        )


def patch_center_grid(height: int, width: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """Pixel centers of a ``height x width`` patch grid, shaped [H, W, 2]."""
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * patch_size
    cols = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * patch_size
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([uu, vv], dim=-1)


def emit_slots(
    codes: torch.Tensor,
    confidence: torch.Tensor,
    patch_size: int,
    top_k: int = 32,
) -> LandmarkSlots:
    """Keep the top-k patches per batch item by calibrated confidence."""
    if codes.ndim != 3:
        raise ValueError(f"codes must be [B, H, W], got {tuple(codes.shape)}")
    if confidence.shape != codes.shape:
        raise ValueError("confidence must match codes shape")
    batch, height, width = codes.shape
    k = min(int(top_k), height * width)
    uv_grid = patch_center_grid(height, width, patch_size, codes.device)
    uv_flat = uv_grid.reshape(1, height * width, 2).expand(batch, -1, -1)
    conf_flat = confidence.reshape(batch, -1)
    code_flat = codes.reshape(batch, -1)
    values, indices = torch.topk(conf_flat, k=k, dim=1)
    gathered_codes = torch.gather(code_flat, 1, indices)
    gathered_uv = torch.gather(uv_flat, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    batch_index = torch.arange(batch, device=codes.device).unsqueeze(1).expand(-1, k)
    return LandmarkSlots(
        codes=gathered_codes.reshape(-1).long(),
        uv=gathered_uv.reshape(-1, 2),
        confidence=values.reshape(-1),
        batch_index=batch_index.reshape(-1),
    )
