"""Fixture-only CPU tests for same-place verification."""

import json

from monarc.cli import main
from monarc.localization.eval_place_score import (
    evaluate_place_score_dirs,
    write_place_score_fixture,
)
from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
)


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


def test_no_rehearsal_run_numbers_are_embedded():
    paths = ["tests/test_eval_place_score.py", "docs/evaluation.md"]
    forbidden = ("0." + "015625", "49" + "40", "8" + "88")
    for path in paths:
        text = open(path, encoding="utf-8").read()
        assert not any(value in text for value in forbidden)
