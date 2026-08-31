"""Finite Scalar Quantization (Mentzer et al., 2023). No VQ-VAE codebook."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from monarc.map.fusion_stem import FUSED_DIM

DEFAULT_FSQ_LEVELS: tuple[int, ...] = (8, 5, 5, 5)
DRY_RUN_FSQ_LEVELS: tuple[int, ...] = (5, 5, 5)
USAGE_FLOOR_FRAC = 0.25


def parse_fsq_levels(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        parts = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
    else:
        parts = tuple(int(v) for v in raw)
    if not parts:
        raise ValueError("FSQ levels must be a non-empty sequence")
    if any(v < 2 for v in parts):
        raise ValueError("each FSQ level must be >= 2")
    return parts


def fsq_usage_floor(
    n_chips: int,
    codebook_size: int,
    n_tokens: int | None = None,
) -> int:
    """Minimum unique codes before a run is treated as collapsed.

    Floor is ``ceil(USAGE_FLOOR_FRAC * min(n_chips, K, n_tokens))``. Four unique
    codes on 128 Golden-Morrison chips fails this for both K=125 and K=1000.
    """
    if n_chips < 1 or codebook_size < 1:
        raise ValueError("n_chips and codebook_size must be >= 1")
    scale = min(int(n_chips), int(codebook_size))
    if n_tokens is not None:
        if n_tokens < 1:
            raise ValueError("n_tokens must be >= 1")
        scale = min(scale, int(n_tokens))
    return max(1, math.ceil(USAGE_FLOOR_FRAC * scale))


def assert_fsq_not_collapsed(
    unique_codes: int,
    n_chips: int,
    codebook_size: int,
    n_tokens: int | None = None,
) -> None:
    floor = fsq_usage_floor(n_chips, codebook_size, n_tokens)
    if int(unique_codes) < floor:
        raise AssertionError(
            f"FSQ codebook collapse: unique_codes={int(unique_codes)} "
            f"< floor={floor} (n_chips={int(n_chips)}, K={int(codebook_size)}"
            + (f", n_tokens={int(n_tokens)}" if n_tokens is not None else "")
            + ")"
        )


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator."""
    return x + (x.round() - x).detach()


class FSQ(nn.Module):
    """Finite scalar quantizer with fixed per-dimension levels ``L``.

    Codebook size is ``prod(L)``. There are no learned codebook vectors.
    Even levels use the Mentzer et al. half-bin offset so all ``L_i`` bins
    are reachable. ``usage_loss`` is a differentiable occupancy/covariance
    term on the pre-quant projection (not a VQ-VAE commitment loss).
    """

    def __init__(self, levels: Sequence[int] = DEFAULT_FSQ_LEVELS) -> None:
        super().__init__()
        parsed = parse_fsq_levels(levels)
        levels_t = torch.tensor(list(parsed), dtype=torch.float32)
        self.levels: tuple[int, ...] = parsed
        self.num_dimensions = len(self.levels)
        self.codebook_size = int(torch.prod(levels_t).item())
        self.register_buffer("levels_buf", levels_t)
        self.register_buffer("half_l", (levels_t - 1.0) / 2.0)
        offset = torch.where(levels_t % 2 == 1, torch.zeros_like(levels_t), torch.full_like(levels_t, 0.5))
        self.register_buffer("offset", offset)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = self.half_l * (1.0 - eps)
        shift = torch.tan(self.offset / half_l.clamp_min(eps))
        return torch.tanh(z + shift) * half_l - self.offset

    def _bin_centers(self, dim: int) -> torch.Tensor:
        level = self.levels[dim]
        return (
            torch.arange(level, device=self.half_l.device, dtype=self.half_l.dtype)
            - self.half_l[dim]
            - self.offset[dim]
        )

    def integers(self, z: torch.Tensor) -> torch.Tensor:
        """Integer coordinates in ``{0, ..., L_i-1}`` per dimension."""
        z_q = round_ste(self.bound(z))
        digits = z_q + self.half_l + self.offset
        return torch.minimum(torch.clamp(digits, min=0), self.levels_buf - 1)

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
        centered = digits.to(self.half_l.dtype) - self.half_l - self.offset
        return torch.where(self.half_l > 0, centered / self.half_l, centered)

    def usage_loss(self, z: torch.Tensor, tau: float = 0.5, cov_weight: float = 0.5) -> torch.Tensor:
        """``1 - H(soft occupancy)/log(K)`` plus off-diagonal FSQ-dim covariance."""
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(f"expected last dim {self.num_dimensions}, got {z.shape[-1]}")
        occupancy = self._occupancy_loss(z, tau=tau)
        return occupancy + float(cov_weight) * self._dim_covariance(z)

    def _occupancy_loss(self, z: torch.Tensor, tau: float) -> torch.Tensor:
        zb = self.bound(z)
        flat = zb.reshape(-1, self.num_dimensions)
        if self.codebook_size <= 65536:
            joint: torch.Tensor | None = None
            for dim, _level in enumerate(self.levels):
                centers = self._bin_centers(dim)
                dist = (flat[:, dim].unsqueeze(1) - centers.unsqueeze(0)).abs()
                p = torch.softmax(-dist / max(float(tau), 1e-4), dim=-1)
                if joint is None:
                    joint = p
                else:
                    joint = (joint.unsqueeze(-1) * p.unsqueeze(1)).reshape(joint.shape[0], -1)
            mean_p = joint.mean(0).clamp_min(1e-8)
            entropy = -(mean_p * mean_p.log()).sum()
            return 1.0 - entropy / math.log(self.codebook_size)
        terms: list[torch.Tensor] = []
        for dim, level in enumerate(self.levels):
            centers = self._bin_centers(dim)
            dist = (flat[:, dim].unsqueeze(1) - centers.unsqueeze(0)).abs()
            p = torch.softmax(-dist / max(float(tau), 1e-4), dim=-1)
            mean_p = p.mean(0).clamp_min(1e-8)
            entropy = -(mean_p * mean_p.log()).sum()
            terms.append(1.0 - entropy / math.log(level))
        return torch.stack(terms).mean()

    def _dim_covariance(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, self.num_dimensions)
        if x.shape[0] < 2 or self.num_dimensions < 2:
            return z.new_zeros(())
        x = x - x.mean(0)
        std = x.std(0).clamp_min(1e-4)
        x = x / std
        cov = (x.transpose(0, 1) @ x) / max(x.shape[0] - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).mean()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize last-dim features. Returns ``(z_hat, codes)``."""
        if z.shape[-1] != self.num_dimensions:
            raise ValueError(f"expected last dim {self.num_dimensions}, got {z.shape[-1]}")
        z_q = round_ste(self.bound(z))
        z_hat = torch.where(self.half_l > 0, z_q / self.half_l, z_q)
        digits = torch.minimum(
            torch.clamp(z_q.detach() + self.half_l + self.offset, min=0),
            self.levels_buf - 1,
        )
        codes = self.pack(digits)
        return z_hat, codes


class FSQHead(nn.Module):
    """Project fused tokens to FSQ dimension and emit integer codes."""

    def __init__(self, in_dim: int = FUSED_DIM, levels: Sequence[int] = DEFAULT_FSQ_LEVELS) -> None:
        super().__init__()
        self.fsq = FSQ(levels)
        self.proj = nn.Conv2d(in_dim, self.fsq.num_dimensions, kernel_size=1)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.45)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def project(self, fused: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` fused tokens to pre-quant FSQ features ``[B, H, W, d]``."""
        return self.proj(fused).permute(0, 2, 3, 1).contiguous()

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.project(fused)
        z_hat, codes = self.fsq(z)
        z_hat_nchw = z_hat.permute(0, 3, 1, 2).contiguous()
        return z_hat_nchw, codes
