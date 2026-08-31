"""Backbone must not hit torch.hub or HTTP during CPU tests."""

import torch

from monarc.map.dino_backbone import FrozenDinoBackbone, load_frozen_dino


def test_stub_forward_does_not_call_hub(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("torch.hub must not be called")

    monkeypatch.setattr(torch.hub, "load", boom)
    net = FrozenDinoBackbone(mode="stub")
    out = net(torch.rand(1, 3, 28, 28))
    assert out.shape == (1, 768, 2, 2)


def test_default_and_auto_do_not_call_hub(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("torch.hub must not be called")

    monkeypatch.setattr(torch.hub, "load", boom)
    default = FrozenDinoBackbone()
    auto = load_frozen_dino(mode="auto", device="cpu")
    assert default.mode == "stub"
    assert auto.mode == "stub"
    default(torch.rand(1, 3, 28, 28))
    auto(torch.rand(1, 3, 28, 28))
