"""Colorado-track retrieve on tiny spatial-holdout fixtures (bag-of-codes + DINO)."""

import json

import numpy as np

from monarc.cli import main
from monarc.localization.eval_retrieve import (
    evaluate_chip_retrieve,
    evaluate_retrieve_dirs,
    spatial_holdout_indices,
    write_retrieve_fixture,
)
from monarc.localization.global_retrieve import (
    BAG_OF_CODES_DESCRIPTOR,
    DINO_GRID_DESCRIPTOR,
    DINO_POOLED_DESCRIPTOR,
)


def test_spatial_holdout_is_a_box_not_random():
    xyz = np.array(
        [
            [0.0, 0.0, 80.0],
            [10.0, 0.0, 80.0],
            [20.0, 0.0, 80.0],
            [30.0, 0.0, 80.0],
            [0.0, 10.0, 80.0],
            [10.0, 10.0, 80.0],
            [20.0, 10.0, 80.0],
            [30.0, 10.0, 80.0],
        ]
    )
    split = spatial_holdout_indices(xyz, query_fraction=0.25, axis="east")
    query_e = xyz[split["query_idx"], 0]
    gallery_e = xyz[split["gallery_idx"], 0]
    assert split["kind"] == "spatial-box"
    assert split["disjoint_box"] is True
    assert np.min(query_e) >= np.max(gallery_e)
    assert set(query_e.tolist()) == {30.0}
    assert set(split["query_idx"].tolist()) == {3, 7}


def test_recall_and_xyz_error_on_matched_nearest(tmp_path):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    write_retrieve_fixture(extract, fsq, match_nearest=True)
    report = evaluate_retrieve_dirs(extract, fsq, query_fraction=0.25, axis="east")
    assert report["track"] == "colorado-retrieval"
    assert report["network"] is False
    assert report["features_used"] is True
    assert report["descriptor"] == BAG_OF_CODES_DESCRIPTOR
    assert report["descriptors"] == [
        BAG_OF_CODES_DESCRIPTOR,
        DINO_POOLED_DESCRIPTOR,
        DINO_GRID_DESCRIPTOR,
    ]
    assert BAG_OF_CODES_DESCRIPTOR in report["modes"]
    assert DINO_POOLED_DESCRIPTOR in report["modes"]
    assert report["modes"][BAG_OF_CODES_DESCRIPTOR]["features_used"] is False
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["features_used"] is True
    assert report["n_chips"] == 8
    assert report["n_query"] == 2
    assert report["n_gallery"] == 6
    assert report["split"]["tiny"] is True
    assert report["split"]["tiny_reason"]
    assert report["recall_at_1"] == 1.0
    assert report["recall_at_5"] == 1.0
    assert report["modes"][BAG_OF_CODES_DESCRIPTOR]["recall_at_1"] == 1.0
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["recall_at_1"] == 1.0
    assert abs(report["median_xyz_error_m"] - 10.0) < 1e-6
    assert abs(report["p90_xyz_error_m"] - 10.0) < 1e-6
    assert abs(report["median_oracle_xyz_m"] - 10.0) < 1e-6
    assert "888" not in json.dumps(report)
    assert "university1652" in report["not"]
    assert "Frozen DINO" in report["note"]


def test_mismatched_bags_do_not_invent_perfect_recall(tmp_path):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    write_retrieve_fixture(extract, fsq, match_nearest=False)
    report = evaluate_retrieve_dirs(extract, fsq, query_fraction=0.25, axis="east")
    assert report["recall_at_1"] == 0.0
    assert report["median_xyz_error_m"] > report["median_oracle_xyz_m"]
    assert report["split"]["tiny"] is True
    dino = report["modes"][DINO_POOLED_DESCRIPTOR]
    assert dino["recall_at_1"] == 1.0
    assert dino["recall_at_1"] > report["recall_at_1"]
    assert abs(dino["median_xyz_error_m"] - report["median_oracle_xyz_m"]) < 1e-6


def test_scrambled_dino_features_do_not_invent_perfect_recall(tmp_path):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    write_retrieve_fixture(extract, fsq, match_nearest=True, match_features_nearest=False)
    report = evaluate_retrieve_dirs(extract, fsq, query_fraction=0.25, axis="east")
    assert report["recall_at_1"] == 1.0
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["recall_at_1"] == 0.0
    assert report["modes"][DINO_POOLED_DESCRIPTOR]["median_xyz_error_m"] > report["median_oracle_xyz_m"]


def test_metric_code_on_inline_arrays():
    codes = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.int64,
    )
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [0.0, 8.0, 1.0],
        ]
    )
    report = evaluate_chip_retrieve(
        codes,
        xyz,
        codebook_size=8,
        ids=["a", "b", "c", "d"],
        query_fraction=0.34,
        axis="east",
    )
    assert report["n_query"] == 1
    assert report["queries"][0]["query_id"] == "c"
    assert report["recall_at_1"] == 1.0
    assert abs(report["median_xyz_error_m"] - 5.0) < 1e-6
    assert report["unique_codes_in_eval"] == 3
    assert report["features_used"] is False
    assert report["descriptors"] == [BAG_OF_CODES_DESCRIPTOR]
    assert DINO_POOLED_DESCRIPTOR not in report["modes"]


def test_inline_features_score_pooled_dino_on_same_split():
    codes = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        dtype=np.int64,
    )
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [0.0, 8.0, 1.0],
        ]
    )
    features = np.zeros((4, 3, 2, 2), dtype=np.float32)
    for i, east in enumerate((0.0, 5.0, 10.0, 0.0)):
        north = xyz[i, 1]
        features[i, 0, :, :] = east + 1.0
        features[i, 1, :, :] = north + 1.0
        features[i, 2, :, :] = 1.0
    report = evaluate_chip_retrieve(
        codes,
        xyz,
        codebook_size=16,
        ids=["a", "b", "c", "d"],
        query_fraction=0.34,
        axis="east",
        features=features,
    )
    assert report["features_used"] is True
    assert report["queries"][0]["query_id"] == "c"
    assert report["recall_at_1"] == 0.0
    dino = report["modes"][DINO_POOLED_DESCRIPTOR]
    assert dino["recall_at_1"] == 1.0
    assert dino["recall_at_1"] > report["recall_at_1"]
    assert abs(dino["median_xyz_error_m"] - 5.0) < 1e-6
    assert abs(dino["median_xyz_error_m"] - report["median_oracle_xyz_m"]) < 1e-6


def test_cli_eval_retrieve_writes_json(tmp_path, capsys):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    write_retrieve_fixture(extract, fsq)
    out = tmp_path / "report.json"
    code = main(
        [
            "eval-retrieve",
            "--extract",
            str(extract),
            "--fsq",
            str(fsq),
            "--out",
            str(out),
            "--axis",
            "east",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text())
    assert payload["track"] == "colorado-retrieval"
    assert saved["recall_at_1"] == payload["recall_at_1"]
    assert payload["features_used"] is True
    assert DINO_POOLED_DESCRIPTOR in payload["modes"]
    assert payload["modes"][DINO_POOLED_DESCRIPTOR]["recall_at_1"] == saved["modes"][DINO_POOLED_DESCRIPTOR]["recall_at_1"]
    assert payload["split"]["tiny"] is True
    assert "Colorado GPS-denied" in payload["note"] or "Colorado GPS-denied" in saved["note"]
    assert payload["unique_codes_in_eval"] != 888


def test_horizontal_error_when_z_is_nan(tmp_path):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    write_retrieve_fixture(extract, fsq)
    xyz = np.load(fsq / "xyz.npy")
    xyz[:, 2] = np.nan
    np.save(fsq / "xyz.npy", xyz)
    np.save(extract / "xyz.npy", xyz)
    report = evaluate_retrieve_dirs(extract, fsq, query_fraction=0.25, axis="east")
    assert report["xyz_error_kind"] == "horizontal-xy"
    assert np.isfinite(report["median_xyz_error_m"])
    assert np.isfinite(report["modes"][DINO_POOLED_DESCRIPTOR]["median_xyz_error_m"])


def test_missing_arrays_fail(tmp_path):
    extract = tmp_path / "extract"
    fsq = tmp_path / "fsq"
    extract.mkdir()
    fsq.mkdir()
    try:
        evaluate_retrieve_dirs(extract, fsq)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass

