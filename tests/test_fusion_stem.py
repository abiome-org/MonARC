"""Channel fusion dimensionality, freeze of DINO, and gradient flow on the stem."""

import torch

from monarc.map.dino_backbone import FrozenDinoBackbone
from monarc.map.fusion_stem import ChannelFusion, FusionStem
from monarc.map.stage1 import default_stage1_modules, train_tiny_codebook


def test_dino_rejects_non_rgb():
    net = FrozenDinoBackbone(mode="stub")
    bad = torch.zeros(1, 6, 56, 56)
    try:
        net(bad)
        assert False, "6-channel RGB must be rejected"
    except ValueError as exc:
        assert "RGB" in str(exc)


def test_dino_is_frozen_and_cpu():
    net = FrozenDinoBackbone(mode="stub")
    rgb = torch.rand(2, 3, 56, 56)
    out = net(rgb)
    assert out.shape == (2, 768, 4, 4)
    assert all(not p.requires_grad for p in net.parameters())
    assert not any(p.is_cuda for p in net.parameters())


def test_vitb14_requires_local_weights():
    try:
        FrozenDinoBackbone(mode="vitb14")
        assert False, "vitb14 must not download weights"
    except ValueError as exc:
        assert "download" in str(exc).lower() or "local" in str(exc).lower()


def test_fusion_stem_shapes_and_grad():
    stem = FusionStem(vector_channels=4)
    mix = ChannelFusion()
    dsm = torch.rand(1, 1, 56, 56, requires_grad=True)
    vec = torch.rand(1, 4, 56, 56)
    f_rgb = torch.rand(1, 768, 4, 4)
    f_geo = stem(dsm, vec)
    fused = mix(f_rgb, f_geo)
    assert f_geo.shape == (1, 128, 4, 4)
    assert fused.shape == (1, 256, 4, 4)
    fused.sum().backward()
    stem_grad = [p.grad is not None and p.grad.abs().sum() > 0 for p in stem.parameters()]
    assert any(stem_grad)


def test_stage1_does_not_unfreeze_dino():
    backbone, stem, mix, head = default_stage1_modules()
    rgb = torch.rand(2, 3, 56, 56)
    dsm = torch.rand(2, 1, 56, 56)
    vec = torch.rand(2, 4, 56, 56)
    train_tiny_codebook(backbone, stem, mix, head, rgb, dsm, vec, steps=2)
    assert all(not p.requires_grad for p in backbone.parameters())
