"""Guard: no hardcoded NAIP quarter-quad IDs; no pixel-in policy; no VQ-VAE."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONARC = ROOT / "monarc"


def _python_sources():
    return list(MONARC.rglob("*.py"))


def test_no_hardcoded_naip_quarter_quads():
    banned = ("m_3910522", "m_3910523", "m_3910506")
    hits = []
    for path in _python_sources():
        text = path.read_text()
        for token in banned:
            if token in text:
                hits.append(f"{path}:{token}")
    assert hits == []


def test_no_hunter_policy_module():
    assert not (MONARC / "hunter").exists()


def test_ingest_does_not_use_boto3_or_dot_aws():
    text = (MONARC / "data" / "aflora_ingest.py").read_text()
    assert "boto3" not in text
    assert ".aws" not in text
    assert "AWS_ACCESS_KEY" not in text
    assert "naip-visualization" in text


def test_eval_retrieve_is_offline_numpy():
    text = (MONARC / "localization" / "eval_retrieve.py").read_text()
    assert "import torch" not in text
    assert "urllib" not in text
    assert "boto3" not in text
    assert "888" not in text


def test_no_vqvae_embedding_codebook():
    text = (MONARC / "map" / "quantizer.py").read_text()
    assert "nn.Embedding" not in text
    assert "VQVAE" not in text
    assert "VectorQuantizer" not in text


def test_perspective_is_only_pixel_consumer_contract():
    from monarc.perspective.encoder import PerspectiveEncoder
    from monarc.localization.dpnp import solve_pnp_lm
    from monarc.localization.matcher import match_codes

    assert "rgb" in PerspectiveEncoder.forward.__code__.co_varnames
    assert "rgb" not in solve_pnp_lm.__code__.co_varnames
    assert "rgb" not in match_codes.__code__.co_varnames
