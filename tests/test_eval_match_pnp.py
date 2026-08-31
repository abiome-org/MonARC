"""CPU-only local DINO matching on retrieved chip fixtures."""

import json

import numpy as np

from monarc.cli import main
from monarc.localization.eval_match_pnp import evaluate_match_pnp_dirs, match_dino_grids


def write_match_fixture(extract, fsq):
    extract.mkdir()
    fsq.mkdir()
    n_east, n_north, grid = 4, 2, 4
    n = n_east * n_north
    xyz = np.array(
        [[c * 10.0, r * 10.0, 80.0] for r in range(n_north) for c in range(n_east)],
        dtype=np.float64,
    )
    xyz[2, 2] = np.nan
    ids = [f"chip-{r}-{c}" for r in range(n_north) for c in range(n_east)]
    codes = np.zeros((n, grid, grid), dtype=np.int64)
    features = np.zeros((n, n * grid * grid, grid, grid), dtype=np.float32)
    for i in range(n):
        codes[i] = i + 1
        for token in range(grid * grid):
            row, col = divmod(token, grid)
            features[i, i * grid * grid + token, row, col] = 1.0
    # Each east holdout query ties bag similarity between a wrong earlier chip
    # and its true west neighbor. Stable retrieval ranks the wrong chip first,
    # while the true neighbor remains in top-K and has the matching local grid.
    for query, wrong, correct in ((3, 0, 2), (7, 4, 6)):
        codes[query, :, :2] = codes[wrong, :, :2]
        codes[query, :, 2:] = codes[correct, :, 2:]
        features[query] = features[correct]
    np.save(extract / "features.npy", features)
    np.save(extract / "xyz.npy", xyz)
    np.save(fsq / "codes.npy", codes)
    np.save(fsq / "xyz.npy", xyz)
    (extract / "ids.json").write_text(json.dumps(ids) + "\n")
    (extract / "meta.json").write_text(
        json.dumps({"backbone_mode": "stub", "patch_size": 14, "rasters_copied": False}) + "\n"
    )
    (fsq / "meta.json").write_text(json.dumps({"codebook_size": 16}) + "\n")


def test_mutual_grid_matches_are_patch_level():
    grid = np.eye(4, dtype=np.float32).reshape(4, 2, 2)
    query_idx, scores = match_dino_grids(grid, grid, min_cosine=0.9)
    assert query_idx.tolist() == [0, 1, 2, 3]
    assert np.allclose(scores, 1.0)


def test_matcher_refines_wrong_rank1_and_handles_nan_z(tmp_path):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_match_fixture(extract, fsq)
    report = evaluate_match_pnp_dirs(extract, fsq, axis="east", top_k=5)
    assert report["track"] == "colorado-match-pnp"
    assert report["network"] is False
    assert report["is_university1652"] is False
    assert report["is_gps_denied_flight_ate"] is False
    assert report["xyz_kind"] == "coarse-chip-center"
    assert report["xyz_is_chip_center"] is True
    assert report["dsm_z_may_be_nan"] is True
    assert report["split"]["tiny"] is True
    assert report["k"] == 5
    assert report["retrieve"]["recall_at_5"] == report["retrieve_recall_at_5"]
    assert all(row["rank1_id"] != row["top_k_ids"][1] for row in report["queries"])
    assert all(row["match_inlier_count"] == 16 for row in report["queries"])
    assert all(row["pnp_success"] is False for row in report["queries"])
    assert all(np.isfinite(row["xy_error_m"]) for row in report["queries"])
    assert report["aggregate"]["matcher_median_xy_error_m"] < report["aggregate"]["rank1_median_xy_error_m"]
    assert "university1652" in report["not"]
    assert "gps-denied-flight-ate" in report["not"]


def test_cli_eval_match_pnp_writes_json(tmp_path, capsys):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_match_fixture(extract, fsq)
    out = tmp_path / "match.json"
    assert main([
        "eval-match-pnp", "--extract", str(extract), "--fsq", str(fsq),
        "--axis", "east", "--top-k", "5", "--out", str(out),
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text())
    assert printed["track"] == saved["track"] == "colorado-match-pnp"
    assert printed["retrieve_recall_at_1"] == saved["retrieve_recall_at_1"]
