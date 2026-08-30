"""Sparse slot emission and temperature calibration."""

import torch

from monarc.map.dino_backbone import FrozenDinoBackbone
from monarc.map.quantizer import FSQHead
from monarc.perspective.calibrator import TemperatureCalibrator
from monarc.perspective.encoder import PerspectiveAdapter, PerspectiveEncoder
from monarc.perspective.slot_head import emit_slots


def test_emit_slots_top_k():
    codes = torch.arange(8, dtype=torch.long).reshape(1, 2, 4)
    conf = torch.linspace(0.1, 0.8, 8).reshape(1, 2, 4)
    slots = emit_slots(codes, conf, patch_size=14, top_k=3)
    assert slots.codes.numel() == 3
    assert slots.uv.shape == (3, 2)
    assert torch.all(slots.confidence[:-1] >= slots.confidence[1:])


def test_perspective_encoder_emits_slots():
    backbone = FrozenDinoBackbone(mode="stub")
    adapter = PerspectiveAdapter()
    head = FSQHead(in_dim=256, levels=(5, 5, 5))
    enc = PerspectiveEncoder(backbone, adapter, head, TemperatureCalibrator(), top_k=5)
    rgb = torch.rand(2, 3, 56, 56)
    slots = enc(rgb)
    assert slots.codes.numel() == 10
    assert slots.uv.shape[0] == 10
