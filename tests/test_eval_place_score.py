"""Fixture-only CPU tests for same-place verification."""

import json

import numpy as np
import torch
from PIL import Image

from monarc.cli import main
from monarc.localization.eval_place_score import (
    _reencode_crop_queries,
    evaluate_place_score_dirs,
    write_place_score_fixture,
)
from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
)
from monarc.map.extract import extract_chips
from monarc.map.dino_backbone import FrozenDinoBackbone
from monarc.map.stage1 import default_stage1_modules, stage1_checkpoint


def _write_reencoded_fixture(tmp_path, *, copy_far=False):
    chips, extract = tmp_path / "chips", tmp_path / "extract"
    chips.mkdir()
    size = 84
    yy, xx = np.mgrid[:size, :size]
    images = []
    for i in range(8):
        base = np.array([25 + 27 * i, 225 - 23 * i, 35 + (43 * i) % 180])
        pattern = (((xx // 7 + yy // 7) % 2) * 24 - 12)[..., None]
        image = np.clip(base + pattern + ((xx - yy) % 9)[..., None], 0, 255).astype(np.uint8)
        images.append(image)
    if copy_far:
        images[3] = images[0].copy()
        images[7] = images[4].copy()
    for i, image in enumerate(images):
        name = f"chip_{i:04d}.png"
        Image.fromarray(image, "RGB").save(chips / name)
        (chips / f"chip_{i:04d}.xyz.json").write_text(
            json.dumps({"xyz": [float(i % 4 * 100), float(i // 4 * 100), 80.0]}) + "\n"
        )
    torch.manual_seed(17)
    _, stem, mix, head = default_stage1_modules()
    checkpoint = tmp_path / "stage1_last.pt"
    weights = tmp_path / "stub_weights.pt"
    torch.save(stage1_checkpoint(stem, mix, head, step=0), checkpoint)
    backbone = FrozenDinoBackbone(mode="stub")
    torch.save(backbone.encoder.state_dict(), weights)
    extract_chips(chips, extract, size=size, backbone=backbone, device="cpu",
                  fsq_ckpt=checkpoint)
    # The fixture extract emitted codes with the exact checkpoint used by live-query encoding.
    (extract / "stage1_last.pt").write_bytes(checkpoint.read_bytes())
    (extract / "meta.json").write_text(json.dumps({
        **json.loads((extract / "meta.json").read_text()),
        "codebook_size": int(np.max(np.load(extract / "codes.npy"))) + 1,
    }) + "\n")
    return chips, extract, checkpoint, weights


def test_crop_place_verification_scores_same_place(tmp_path):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_place_score_fixture(extract, fsq, spacing_m=100.0)
    report = evaluate_place_score_dirs(extract, fsq, axis="east")
    assert report["track"] == "colorado-place-verification"
    assert report["protocol"] == "same-place overlap / crop-jitter"
    assert report["network"] is False
    assert report["n_crop_queries"] == report["n_gallery"]
    assert report["n_overlap_queries"] == 0
    assert report["not"] == ["university1652", "ortholoc", "colorado-flight-ate", "hunter", "vla"]
    for mode in (BAG_OF_CODES_DESCRIPTOR, DINO_POOLED_DESCRIPTOR, DINO_GRID_DESCRIPTOR):
        assert report["modes"][mode]["recall_at_1_same_place"] >= 0.75
        assert report["modes"][mode]["auroc"] >= 0.75
    assert report["n_inliers_same_place"] > report["n_inliers_far"] / report["n_far_queries"]
    assert report["modes"][DINO_GRID_DESCRIPTOR]["xy_when_true_in_top_k"]["median_m"] == 0.0


def test_tight_fixture_reports_real_overlap_queries(tmp_path):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_place_score_fixture(extract, fsq, spacing_m=10.0)
    report = evaluate_place_score_dirs(extract, fsq, axis="east")
    assert report["n_overlap_queries"] > 0
    assert report["overlap_radius_m"] == report["chip_size_m"]


def test_far_copies_do_not_invent_perfect_auc(tmp_path):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_place_score_fixture(extract, fsq, spacing_m=100.0, scramble_far=True)
    report = evaluate_place_score_dirs(extract, fsq, axis="east")
    assert report["modes"][BAG_OF_CODES_DESCRIPTOR]["auroc"] < 1.0
    assert report["modes"][DINO_GRID_DESCRIPTOR]["auroc"] < 1.0


def test_cli_writes_place_json(tmp_path, capsys):
    extract, fsq, out = tmp_path / "extract", tmp_path / "fsq", tmp_path / "place.json"
    write_place_score_fixture(extract, fsq)
    assert main(["eval-place-score", "--extract", str(extract), "--fsq", str(fsq),
                 "--out", str(out), "--axis", "east"]) == 0
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text())
    assert stdout["track"] == saved["track"] == "colorado-place-verification"
    assert saved["network"] is False


def test_reencoded_crop_cli_uses_real_png_views_and_limits_queries(tmp_path, capsys):
    chips, extract, checkpoint, weights = _write_reencoded_fixture(tmp_path)
    out = tmp_path / "place-reencoded.json"
    assert main(["eval-place-score", "--extract", str(extract), "--fsq", str(extract),
                 "--out", str(out), "--axis", "east", "--query-kind", "reencoded-crop",
                 "--chips", str(chips), "--fsq-ckpt", str(checkpoint),
                 "--backbone", "stub", "--weights", str(weights),
                 "--max-crop-queries", "2"]) == 0
    capsys.readouterr()
    saved = json.loads(out.read_text())
    assert saved["query_kind"] == "reencoded-crop"
    assert saved["not"] == ["university1652", "ortholoc", "colorado-flight-ate", "hunter", "vla"]
    assert saved["n_crop_queries"] == saved["n_reencoded"] == 2
    assert saved["crop_margin_px"] == 28
    assert saved["resize_px"] == saved["size_px"] == 84
    assert saved["network"] is False
    assert saved["modes"][DINO_POOLED_DESCRIPTOR]["recall_at_1_same_place"] >= 0.5
    assert saved["modes"][DINO_GRID_DESCRIPTOR]["auroc"] >= 0.5

    payload = json.loads((extract / "ids.json").read_text())
    queries, _ = _reencode_crop_queries(
        chips, payload, np.array([0]), checkpoint, size_px=84, patch_size=14,
        crop_margin=2, backbone_mode="stub", weights_path=weights, device="cpu",
        allow_download=False, max_crop_queries=None,
    )
    stored = np.load(extract / "features.npy")[0, :, 2:-2, 2:-2]
    assert queries[0]["features"].shape != stored.shape


def test_reencoded_far_copies_do_not_invent_perfect_auc(tmp_path):
    chips, extract, checkpoint, weights = _write_reencoded_fixture(tmp_path, copy_far=True)
    report = evaluate_place_score_dirs(
        extract, extract, axis="east", query_kind="reencoded-crop", chips_dir=chips,
        fsq_ckpt=checkpoint, backbone_mode="stub", weights_path=weights, device="cpu",
    )
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["auroc"] < 1.0
    assert report["modes"][DINO_GRID_DESCRIPTOR]["auroc"] < 1.0


def test_no_rehearsal_run_numbers_are_embedded():
    paths = ["tests/test_eval_place_score.py", "docs/evaluation.md", "README.md"]
    forbidden = ("0." + "015625", "49" + "40", "8" + "88", "0." + "997", "67" + ".2")
    for path in paths:
        text = open(path, encoding="utf-8").read()
        assert not any(value in text for value in forbidden)
