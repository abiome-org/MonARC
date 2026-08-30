"""Temperature-scaled confidence from quantized token magnitude."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureCalibrator(nn.Module):
    """Map per-patch logits to ``(0, 1)`` with a positive temperature."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.score = nn.Conv2d(1, 1, kernel_size=1)
        self.temperature = float(temperature)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        energy = z_hat.pow(2).mean(dim=1, keepdim=True)
        logits = self.score(energy) / self.temperature
        return torch.sigmoid(logits).squeeze(1)
