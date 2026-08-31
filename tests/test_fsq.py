"""FSQ determinism, codebook size, and absence of a learned VQ dictionary."""

import torch

from monarc.map.quantizer import FSQ, FSQHead


def test_codebook_size_is_product_of_levels():
    fsq = FSQ(levels=(5, 5, 5))
    assert fsq.codebook_size == 125
    assert not hasattr(fsq, "embedding")
    assert not any("codebook" in n.lower() and isinstance(m, torch.nn.Embedding) for n, m in fsq.named_modules())


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


def test_fsq_head_spatial_codes():
    head = FSQHead(in_dim=256, levels=(5, 5, 5))
    fused = torch.randn(2, 256, 4, 4)
    z_hat, codes = head(fused)
    assert z_hat.shape == (2, 3, 4, 4)
    assert codes.shape == (2, 4, 4)
