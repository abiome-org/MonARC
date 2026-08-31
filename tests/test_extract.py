"""Extract and train-fsq on local chips. No network, stub backbone."""

import json
from pathlib import Path

import numpy as np
import torch

from monarc.cli import main
from monarc.map.extract import extract_chips, write_chip_fixture
from monarc.map.train_fsq import train_fsq_from_cache


def test_extract_writes_features_not_rasters(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("torch.hub must not be called")

    monkeypatch.setattr(torch.hub, "load", boom)
    chips = write_chip_fixture(tmp_path / "chips", n=4, size=28)
    out = tmp_path / "feat"
    meta = extract_chips(
        chips,
        out,
        size=28,
        backbone_mode="stub",
        device="cpu",
        batch_size=2,
    )
    assert meta["backbone_mode"] == "stub"
    assert meta["n_chips"] == 4
    assert meta["has_dsm"] is True
    assert meta["rasters_copied"] is False
    features = np.load(out / "features.npy")
    xyz = np.load(out / "xyz.npy")
    assert features.shape == (4, 768, 2, 2)
    assert xyz.shape == (4, 3)
    assert np.isfinite(xyz).all()
    assert (out / "dsm.npy").exists()
    assert not list(out.glob("**/*.tif"))
    assert not list(out.glob("**/*.png"))


def test_train_fsq_from_extract_checkpoints(tmp_path):
    chips = write_chip_fixture(tmp_path / "chips", n=4, size=28)
    feat_dir = tmp_path / "feat"
    extract_chips(chips, feat_dir, size=28, backbone_mode="stub", device="cpu")
    out = tmp_path / "fsq"
    report = train_fsq_from_cache(
        feat_dir,
        out,
        steps=6,
        batch_size=2,
        device="cpu",
        ckpt_every=3,
        keep_last=2,
        seed=0,
    )
    assert report["n_chips"] == 4
    assert report["has_dsm"] is True
    assert report["collapsed"] is False
    assert report["codebook_size"] == 1000
    assert (out / "codes.npy").exists()
    assert (out / "xyz.npy").exists()
    assert (out / "stage1_last.pt").exists()
    ckpts = list(out.glob("ckpt_step_*.pt"))
    assert len(ckpts) <= 2
    assert (out / "metric_index" / "codes.npy").exists()
    payload = torch.load(out / "stage1_last.pt", map_location="cpu", weights_only=True)
    assert payload["step"] == 6
    assert "fsq_head" in payload


def test_cli_extract_and_train_fsq(tmp_path, capsys):
    chips = write_chip_fixture(tmp_path / "chips", n=3, size=28)
    feat = tmp_path / "feat"
    code = main(
        [
            "extract",
            "--chips",
            str(chips),
            "--out",
            str(feat),
            "--size",
            "28",
            "--backbone",
            "stub",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["backbone_mode"] == "stub"
    fsq = tmp_path / "fsq"
    code = main(
        [
            "train-fsq",
            "--features",
            str(feat),
            "--out",
            str(fsq),
            "--steps",
            "4",
            "--ckpt-every",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["step"] == 4
    assert Path(report["out_dir"]).joinpath("codes.npy").exists()


def test_extract_without_dsm(tmp_path):
    chips = write_chip_fixture(tmp_path / "chips", n=2, size=28, with_dsm=False)
    out = tmp_path / "feat"
    meta = extract_chips(chips, out, size=28, backbone_mode="stub", device="cpu")
    assert meta["has_dsm"] is False
    assert not (out / "dsm.npy").exists()
    report = train_fsq_from_cache(out, tmp_path / "fsq", steps=2, device="cpu", ckpt_every=0)
    assert report["has_dsm"] is False
