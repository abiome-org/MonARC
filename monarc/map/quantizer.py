"""Finite Scalar Quantization (Mentzer et al., 2023). No VQ-VAE codebook."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monarc.map.fusion_stem import FUSED_DIM


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator."""
    return x + (x.round() - x).detach()


class FSQ(nn.Module):
    """Finite scalar quantizer with fixed per-dimension levels ``L``.

    Codebook size is ``prod(L)``. There are no learned codebook vectors.
    """

    def __init__(self, levels: Sequence[int] = (5, 5, 5)) -> None:
        super().__init__()
        levels_t = torch.tensor([int(v) for v in levels], dtype=torch.float32)
        if torch.any(levels_t < 2):
            raise ValueError("each FSQ level must be >= 2")
        self.levels: tuple[int, ...] = tuple(int(v) for v in levels)
        self.num_dimensions = len(self.levels)
        self.codebook_size = int(torch.prod(levels_t).item())
        self.register_buffer("levels_buf", levels_t)
        self.register_buffer("half_l", (levels_t - 1.0) / 2.0)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z) * self.half_l

    def integers(self, z: torch.Tensor) -> torch.Tensor:
        """Integer coordinates in ``{0, ..., L_i-1}`` per dimension."""
        z_q = torch.clamp(round_ste(self.bound(z)), min=-self.half_l, max=self.half_l)
        return torch.minimum(torch.clamp(z_q + self.half_l, min=0), self.levels_buf - 1)

    def pack(self, digits: torch.Tensor) -> torch.Tensor:
        """Mixed-radix pack of per-dimension digits into a scalar code."""
        stride = 1
        codes = torch.zeros(digits.shape[:-1], dtype=torch.long, device=digits.device)
        for dim, level in enumerate(self.levels):
            codes = codes + digits[..., dim].long() * stride
            stride *= level
        return codes

    def unpack(self, codes: torch.Tensor) -> torch.Tensor:
        """Mixed-radix unpack to per-dimension digits."""
        codes = codes.long()
        digits = []
        residual = codes
        for level in self.levels:
            digits.append(residual % level)
            residual = residual // level
        return torch.stack(digits, dim=-1).to(dtype=self.half_l.dtype)

    def reconstruct(self, digits: torch.Tensor) -> torch.Tensor:
        """Map integer digits back to the bounded scalar domain ``[-1, 1]``."""
        centered = digits.to(self.half_l.dtype) - self.half_l
        return torch.where(self.half_l > 0, centered / self.half_l, centered)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize last-dim features. Returns ``(z_hat, codes)``."""
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(f"expected last dim {self.num_dimensions}, got {z.shape[-1]}")
        z_q = torch.clamp(round_ste(self.bound(z)), min=-self.half_l, max=self.half_l)
        z_hat = torch.where(self.half_l > 0, z_q / self.half_l, z_q)
        digits = torch.minimum(torch.clamp(z_q.detach() + self.half_l, min=0), self.levels_buf - 1)
        codes = self.pack(digits)
        return z_hat, codes


class FSQHead(nn.Module):
    """Project fused tokens to FSQ dimension and emit integer codes."""

    def __init__(self, in_dim: int = FUSED_DIM, levels: Sequence[int] = (5, 5, 5)) -> None:
        super().__init__()
        self.fsq = FSQ(levels)
        self.proj = nn.Conv2d(in_dim, self.fsq.num_dimensions, kernel_size=1)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.45)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(fused).permute(0, 2, 3, 1).contiguous()
        z_hat, codes = self.fsq(z)
        z_hat_nchw = z_hat.permute(0, 3, 1, 2).contiguous()
        return z_hat_nchw, codes
