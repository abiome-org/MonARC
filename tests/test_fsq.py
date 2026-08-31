"""FSQ determinism, codebook size, usage floor, and absence of a learned VQ dictionary."""

import pytest
import torch

from monarc.map.quantizer import (
    DEFAULT_FSQ_LEVELS,
    FSQ,
    FSQHead,
    assert_fsq_not_collapsed,
    fsq_usage_floor,
    parse_fsq_levels,
)
from monarc.map.stage1 import default_stage1_modules, train_from_features


def test_codebook_size_is_product_of_levels():
    fsq = FSQ(levels=(5, 5, 5))
    assert fsq.codebook_size == 125
    assert not hasattr(fsq, "embedding")
    assert not any("codebook" in n.lower() and isinstance(m, torch.nn.Embedding) for n, m in fsq.named_modules())


def test_default_levels_are_mentzer_10bit():
    assert DEFAULT_FSQ_LEVELS == (8, 5, 5, 5)
    fsq = FSQ()
    assert fsq.codebook_size == 1000
    assert fsq.num_dimensions == 4


def test_parse_fsq_levels():
    assert parse_fsq_levels("8,5,5,5") == (8, 5, 5, 5)
    assert parse_fsq_levels((5, 5, 5)) == (5, 5, 5)
    with pytest.raises(ValueError):
        parse_fsq_levels("1,5")


def test_fsq_is_deterministic():
    fsq = FSQ(levels=(8, 5, 5))
    z = torch.randn(4, 7, 3)
    z_hat_a, codes_a = fsq(z)
    z_hat_b, codes_b = fsq(z)
    assert torch.equal(codes_a, codes_b)
    assert torch.allclose(z_hat_a, z_hat_b)
    assert codes_a.min() >= 0
    assert codes_a.max() < fsq.codebook_size


def test_fsq_pack_unpack():
    fsq = FSQ(levels=(4, 3, 5))
    digits = torch.tensor([[0, 0, 0], [3, 2, 4], [1, 1, 2]], dtype=torch.float32)
    codes = fsq.pack(digits)
    back = fsq.unpack(codes)
    assert torch.equal(back, digits)


def test_even_levels_use_all_bins():
    fsq = FSQ(levels=(8, 5, 5, 5))
    z = torch.zeros(4001, 4)
    z[:, 0] = torch.linspace(-16, 16, 4001)
    digits = fsq.integers(z)
    used = {int(v) for v in digits[:, 0].tolist()}
    assert used == set(range(8))


def test_fsq_head_spatial_codes():
    head = FSQHead(in_dim=256, levels=(5, 5, 5))
    fused = torch.randn(2, 256, 4, 4)
    z_hat, codes = head(fused)
    assert z_hat.shape == (2, 3, 4, 4)
    assert codes.shape == (2, 4, 4)


def test_usage_floor_flags_four_of_128():
    assert fsq_usage_floor(128, 125) > 4
    assert fsq_usage_floor(128, 1000) > 4
    with pytest.raises(AssertionError, match="collapse"):
        assert_fsq_not_collapsed(4, n_chips=128, codebook_size=125)
    with pytest.raises(AssertionError, match="collapse"):
        assert_fsq_not_collapsed(4, n_chips=128, codebook_size=1000)
    with pytest.raises(AssertionError, match="collapse"):
        assert_fsq_not_collapsed(1, n_chips=128, codebook_size=125)


def test_usage_loss_has_gradient():
    fsq = FSQ()
    z = torch.randn(16, 8, 4, requires_grad=True)
    loss = fsq.usage_loss(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_nonconstant_features_yield_many_unique_codes():
    torch.manual_seed(0)
    n_chips, height, width = 8, 4, 4
    _, stem, mix, head = default_stage1_modules()
    f_rgb = torch.randn(n_chips, 768, height, width)
    out = train_from_features(
        stem,
        mix,
        head,
        f_rgb,
        steps=8,
        batch_size=8,
        device="cpu",
        lr=1e-3,
    )
    n_tokens = n_chips * height * width
    floor = fsq_usage_floor(n_chips, out["codebook_size"], n_tokens)
    assert_fsq_not_collapsed(out["unique_codes"], n_chips, out["codebook_size"], n_tokens)
    assert out["unique_codes"] > 4
    assert out["unique_codes"] >= min(32, n_tokens // 4, out["codebook_size"])
    assert out["unique_codes"] >= floor
    assert out["codebook_size"] == 1000
    assert not any(isinstance(m, torch.nn.Embedding) for m in head.modules())
