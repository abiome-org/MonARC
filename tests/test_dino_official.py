"""Official DINOv2-B load path. Hub is mocked; default stays stub."""

import torch
import torch.nn as nn

from monarc.map.dino_backbone import (
    FrozenDinoBackbone,
    OfficialDinov2Grid,
    load_frozen_dino,
    resolve_backbone_mode,
    tokens_to_nchw,
)


class _FakeHubDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward_features(self, x: torch.Tensor) -> dict:
        batch, _, height, width = x.shape
        n = (height // 14) * (width // 14)
        tokens = torch.zeros(batch, n, 768, device=x.device, dtype=x.dtype)
        tokens = tokens + self.scale.to(dtype=x.dtype, device=x.device)
        return {"x_norm_patchtokens": tokens}


def test_auto_is_stub_on_cpu():
    assert resolve_backbone_mode("auto", device="cpu") == "stub"
    net = load_frozen_dino(mode="auto", device="cpu")
    assert net.mode == "stub"
    out = net(torch.rand(1, 3, 28, 28))
    assert out.shape == (1, 768, 2, 2)


def test_tokens_to_nchw_shape():
    tokens = torch.arange(2 * 4 * 768, dtype=torch.float32).reshape(2, 4, 768)
    grid = tokens_to_nchw(tokens, 28, 28)
    assert grid.shape == (2, 768, 2, 2)


def test_vitb14_mocked_hub_is_frozen(monkeypatch):
    def fake_load(**_k):
        return _FakeHubDino(), "torch-hub:dinov2_vitb14"

    monkeypatch.setattr("monarc.map.dino_backbone.load_official_vitb14", fake_load)
    net = FrozenDinoBackbone(mode="vitb14", allow_download=True)
    assert net.mode == "vitb14"
    assert net.source == "torch-hub:dinov2_vitb14"
    assert all(not p.requires_grad for p in net.parameters())
    out = net(torch.rand(2, 3, 28, 28))
    assert out.shape == (2, 768, 2, 2)
    assert not any(p.is_cuda for p in net.parameters())


def test_official_grid_wraps_patch_tokens():
    inner = _FakeHubDino()
    grid = OfficialDinov2Grid(inner)
    x = torch.rand(1, 3, 56, 56)
    out = grid(x)
    assert out.shape == (1, 768, 4, 4)
