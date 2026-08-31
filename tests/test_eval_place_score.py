"""Fixture-only CPU tests for same-place verification."""

import json

import numpy as np
import torch
from PIL import Image

from monarc.cli import main
from monarc.localization.eval_place_score import (
    _reencode_crop_queries,
    _reencode_overlap_queries,
    _nearest_overlap_pairs,
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


def _write_overlap_fixture(tmp_path, *, spacing_m=21.0, copy_far=False):
    chips, extract = tmp_path / "chips", tmp_path / "extract"
    chips.mkdir()
    size, stride = 84, 42
    rng = np.random.default_rng(29)
    tile = rng.integers(0, 256, size=(size + stride, size + 3 * stride, 3), dtype=np.uint8)
    # A repeated strip makes adjacent gallery crops independently read the same
    # patterned place, while the held-out final strip remains different.
    repeated = rng.integers(0, 256, size=(size + stride, stride, 3), dtype=np.uint8)
    tile[:, :4 * stride] = np.tile(repeated, (1, 4, 1))
    for i in range(8):
        row, col = divmod(i, 4)
        image = tile[row * stride:row * stride + size, col * stride:col * stride + size].copy()
        if copy_far and i in (3, 7):
            image = tile[:size, :size].copy()
        name = f"chip_{i:04d}.png"
        Image.fromarray(image, "RGB").save(chips / name)
        (chips / f"chip_{i:04d}.xyz.json").write_text(
            json.dumps({"xyz": [float(col * spacing_m), float(row * 35.0), 80.0]}) + "\n")
    (chips / "chips_meta.json").write_text(json.dumps({"overlap_frac": 0.5}) + "\n")
    torch.manual_seed(17)
    _, stem, mix, head = default_stage1_modules()
    checkpoint, weights = tmp_path / "stage1_last.pt", tmp_path / "stub_weights.pt"
    torch.save(stage1_checkpoint(stem, mix, head, step=0), checkpoint)
    backbone = FrozenDinoBackbone(mode="stub")
    torch.save(backbone.encoder.state_dict(), weights)
    extract_chips(chips, extract, size=size, backbone=backbone, device="cpu", fsq_ckpt=checkpoint)
    (extract / "stage1_last.pt").write_bytes(checkpoint.read_bytes())
    return chips, extract, checkpoint, weights


def _write_mixed_fixture(tmp_path, *, hole=False):
    query_root = tmp_path / "query"
    query_root.mkdir()
    chips, query_extract, checkpoint, weights = _write_overlap_fixture(query_root)
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    query_features = np.load(query_extract / "features.npy")
    query_codes = np.load(query_extract / "codes.npy")
    if hole:
        xy = [(-2000.0, -2000.0), (2000.0, -2000.0), (-2000.0, 2000.0), (2000.0, 2000.0)]
        features = query_features[4:8].copy()
        codes = query_codes[4:8].copy()
    else:
        xy = [(0.0, 0.0), (2000.0, 0.0), (-2000.0, 2000.0)]
        features = np.stack([query_features[0], query_features[6], query_features[7]])
        codes = np.stack([query_codes[0], query_codes[6], query_codes[7]])
    np.save(gallery / "features.npy", features)
    np.save(gallery / "codes.npy", codes)
    np.save(gallery / "xyz.npy", np.asarray([[x, y, 80.0] for x, y in xy]))
    (gallery / "ids.json").write_text(json.dumps([f"gallery-{i}.png" for i in range(len(xy))]) + "\n")
    meta = {"size": 84, "backbone_mode": "stub", "codebook_size": int(np.max(codes)) + 1}
    (gallery / "meta.json").write_text(json.dumps(meta) + "\n")
    (gallery / "stage1_last.pt").write_bytes(checkpoint.read_bytes())
    return chips, query_extract, gallery, checkpoint, weights


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


def test_reencoded_overlap_cli_uses_neighbor_pngs_and_limit(tmp_path, capsys):
    chips, extract, checkpoint, weights = _write_overlap_fixture(tmp_path)
    out = tmp_path / "overlap.json"
    assert main(["eval-place-score", "--extract", str(extract), "--fsq", str(extract),
                 "--out", str(out), "--axis", "east", "--gsd-m", "0.5",
                 "--query-kind", "reencoded-overlap", "--chips", str(chips),
                 "--fsq-ckpt", str(checkpoint), "--backbone", "stub",
                 "--weights", str(weights), "--max-overlap-queries", "1"]) == 0
    capsys.readouterr()
    saved = json.loads(out.read_text())
    assert saved["query_kind"] == "reencoded-overlap"
    assert saved["protocol"] == "same-place overlap (reencoded neighbor)"
    assert saved["n_overlap_queries"] == saved["n_reencoded"] == 1
    assert saved["n_crop_queries"] == 0
    assert saved["n_overlap_pairs"] >= saved["n_overlap_queries"] > 0
    assert saved["median_neighbor_spacing_m"] < saved["chip_size_m"]
    assert saved["overlap_frac"] == 0.5
    assert saved["network"] is False
    assert saved["backbone_mode"] == "stub"
    assert saved["backbone_source"]
    assert saved["fsq_ckpt"] == str(checkpoint)
    assert saved["not"] == ["university1652", "ortholoc", "colorado-flight-ate", "hunter", "vla"]
    assert saved["modes"][DINO_GRID_DESCRIPTOR]["auroc"] >= 0.5
    assert saved["modes"][DINO_GRID_DESCRIPTOR]["recall_at_1_same_place"] >= 0.5

    ids = json.loads((extract / "ids.json").read_text())
    xyz = np.load(extract / "xyz.npy")
    split_gallery = np.array([0, 1, 2, 4, 5, 6])
    pairs, _ = _nearest_overlap_pairs(split_gallery, xyz, 42.0)
    queries, _ = _reencode_overlap_queries(
        chips, ids, pairs, checkpoint, size_px=84, backbone_mode="stub",
        weights_path=weights, device="cpu", allow_download=False, max_overlap_queries=1)
    assert not np.shares_memory(queries[0]["features"], np.load(extract / "features.npy"))
    assert queries[0]["true"] != queries[0]["source"]


def test_reencoded_overlap_honestly_reports_zero_on_coarse_grid(tmp_path):
    chips, extract, checkpoint, weights = _write_reencoded_fixture(tmp_path)
    report = evaluate_place_score_dirs(
        extract, extract, axis="east", gsd_m=0.3, query_kind="reencoded-overlap",
        chips_dir=chips, fsq_ckpt=checkpoint, backbone_mode="stub",
        weights_path=weights, device="cpu")
    assert report["n_overlap_queries"] == report["n_overlap_pairs"] == 0
    assert report["n_crop_queries"] == 0
    assert report["auroc"] is None


def test_reencoded_overlap_far_copies_do_not_invent_perfect_auc(tmp_path):
    chips, extract, checkpoint, weights = _write_overlap_fixture(tmp_path, copy_far=True)
    report = evaluate_place_score_dirs(
        extract, extract, axis="east", gsd_m=0.5, query_kind="reencoded-overlap",
        chips_dir=chips, fsq_ckpt=checkpoint, backbone_mode="stub",
        weights_path=weights, device="cpu")
    assert report["n_overlap_queries"] > 0
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["auroc"] < 1.0


def test_mixed_overlap_uses_full_km_gallery_and_limit(tmp_path, capsys):
    chips, query_extract, gallery, checkpoint, weights = _write_mixed_fixture(tmp_path)
    out = tmp_path / "mixed.json"
    assert main(["eval-place-score", "--query-kind", "reencoded-overlap",
                 "--query-extract", str(query_extract), "--extract", str(gallery),
                 "--fsq", str(gallery), "--chips", str(chips), "--out", str(out),
                 "--gsd-m", "0.5", "--backbone", "stub", "--weights", str(weights),
                 "--fsq-ckpt", str(checkpoint), "--max-overlap-queries", "1"]) == 0
    capsys.readouterr()
    report = json.loads(out.read_text())
    assert report["mixed_gallery"] is True
    assert report["query_kind"] == "reencoded-overlap"
    assert report["n_crop_queries"] == 0
    assert report["n_overlap_queries"] == report["n_reencoded"] == 1
    assert report["n_gallery"] == 3
    assert report["network"] is False
    assert report["far_distance"]["true_km_far"] is True
    assert report["far_distance"]["scale"] == "km"
    assert max(report["far_distance"]["gallery_span_east_m"],
               report["far_distance"]["gallery_span_north_m"]) >= 1000.0
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["auroc"] >= 0.5


def test_mixed_overlap_hole_does_not_assign_nearest_as_truth(tmp_path):
    chips, query_extract, gallery, checkpoint, weights = _write_mixed_fixture(tmp_path, hole=True)
    report = evaluate_place_score_dirs(
        gallery, gallery, query_extract_dir=query_extract, query_kind="reencoded-overlap",
        chips_dir=chips, gsd_m=0.5, fsq_ckpt=checkpoint, backbone_mode="stub",
        weights_path=weights, max_overlap_queries=2)
    assert report["query_bbox_inside_gallery_bbox"] is True
    assert report["n_overlap_queries"] == report["n_overlap_pairs"] == 0
    assert report["auroc"] is None
    assert report["recall_at_1_same_place"] is None
    assert report["far_distance"]["true_km_far"] is True
    for row in report["modes"][BAG_OF_CODES_DESCRIPTOR]["queries"]:
        assert row["true_id"] is None


def test_nonoverlapping_query_extract_is_honest_zero_in_both_modes(tmp_path):
    chips, query_extract, gallery, checkpoint, weights = _write_mixed_fixture(tmp_path)
    xyz = np.load(query_extract / "xyz.npy")
    xyz[:, 0] = np.arange(len(xyz)) * 1000.0
    np.save(query_extract / "xyz.npy", xyz)
    mixed = evaluate_place_score_dirs(
        gallery, gallery, query_extract_dir=query_extract, query_kind="reencoded-overlap",
        chips_dir=chips, gsd_m=0.5, fsq_ckpt=checkpoint, backbone_mode="stub", weights_path=weights)
    same = evaluate_place_score_dirs(
        query_extract, query_extract, query_kind="reencoded-overlap", chips_dir=chips,
        gsd_m=0.5, fsq_ckpt=checkpoint, backbone_mode="stub", weights_path=weights)
    assert mixed["n_overlap_queries"] == same["n_overlap_queries"] == 0


def test_no_rehearsal_run_numbers_are_embedded():
    paths = [
        "docs/evaluation.md", "README.md", "monarc/map/cog_chips.py",
        "monarc/data/aflora_ingest.py", "monarc/localization/eval_place_score.py",
    ]
    forbidden = (
        "0." + "0148", "0." + "014765625", "2" + "28", "67" + ".2",
        "2" + "56", "9" + "380", "4" + "019", "0." + "015625",
        "49" + "40", "8" + "88", "0." + "997",
    )
    for path in paths:
        text = open(path, encoding="utf-8").read()
        assert not any(value in text for value in forbidden)
