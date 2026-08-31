"""CPU-only local DINO matching on retrieved chip fixtures."""

import json

import numpy as np

from monarc.cli import main
from monarc.common.se3 import invert_se3
from monarc.localization.dpnp import PnPResult
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
    assert all(
        row["xy_estimate_kind"] == "matched-chip-center-horizontal-fallback"
        for row in report["queries"]
    )
    assert all(row["refined_xy_m"] == row["matcher_xy_m"] for row in report["queries"])
    assert all(row["pnp_xy_m"] is None for row in report["queries"])
    assert all(np.isfinite(row["xy_error_m"]) for row in report["queries"])
    assert report["aggregate"]["matcher_median_xy_error_m"] < report["aggregate"]["rank1_median_xy_error_m"]
    assert report["aggregate"]["n_pnp_success"] == 0
    assert report["aggregate"]["pnp_median_xy_error_m"] is None
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


def test_matcher_reports_when_all_chip_center_z_is_finite(tmp_path):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_match_fixture(extract, fsq)
    for directory in (extract, fsq):
        xyz = np.load(directory / "xyz.npy")
        xyz[:, 2] = 80.0
        np.save(directory / "xyz.npy", xyz)
    report = evaluate_match_pnp_dirs(extract, fsq, axis="east", top_k=5)
    assert report["xyz_kind"] == "coarse-chip-center"
    assert report["xyz_is_chip_center"] is True
    assert report["dsm_z_may_be_nan"] is False


def test_successful_pnp_reports_camera_in_world_xy(tmp_path, monkeypatch):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_match_fixture(extract, fsq)
    for directory in (extract, fsq):
        xyz = np.load(directory / "xyz.npy")
        xyz[:, 2] = np.array([70.0, 73.0, 79.0, 82.0, 71.0, 77.0, 80.0, 86.0])
        np.save(directory / "xyz.npy", xyz)
    features = np.load(extract / "features.npy")
    features[:] = features[0]
    np.save(extract / "features.npy", features)

    T_cw = np.eye(4)
    T_cw[:3, 3] = [-4.0, -6.0, -9.0]
    calls = 0

    def successful_pnp(corr, _K):
        nonlocal calls
        unique_xyz = np.unique(corr.xyz, axis=0)
        assert unique_xyz.shape[0] > 1
        assert np.linalg.matrix_rank(unique_xyz - unique_xyz.mean(axis=0)) == 3
        pose = T_cw.copy()
        if calls:
            pose[0, 3] = np.nan
        calls += 1
        return PnPResult(pose, np.arange(len(corr)), 0.0, True, len(corr))

    monkeypatch.setattr("monarc.localization.eval_match_pnp.solve_pnp_lm", successful_pnp)

    report = evaluate_match_pnp_dirs(extract, fsq, axis="east", top_k=5)
    successful = [row for row in report["queries"] if row["pnp_success"]]
    assert successful
    row = successful[0]
    assert row["xy_estimate_kind"] == "pnp-horizontal"
    expected_xy = invert_se3(np.asarray(row["pose_T_cw"]))[:2, 3]
    assert np.allclose(row["refined_xy_m"], expected_xy, rtol=0.0, atol=1e-9)
    assert np.allclose(row["pnp_xy_m"], expected_xy, rtol=0.0, atol=1e-9)
    failed = [row for row in report["queries"] if not row["pnp_success"]]
    assert failed[0]["xy_estimate_kind"] == "matched-chip-center-horizontal-fallback"
    assert failed[0]["refined_xy_m"] == failed[0]["matcher_xy_m"]
    assert failed[0]["pnp_xy_m"] is None
    assert report["aggregate"]["n_pnp_success"] == len(successful)
    assert report["aggregate"]["pnp_median_xy_error_m"] is not None


def test_per_patch_xyz_uses_only_best_candidate_local_ties(tmp_path, monkeypatch):
    extract, fsq = tmp_path / "extract", tmp_path / "fsq"
    write_match_fixture(extract, fsq)
    features = np.load(extract / "features.npy")
    n, _channels, height, width = features.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    patch_xyz = np.empty((n, height, width, 3), dtype=np.float64)
    for chip in range(n):
        patch_xyz[chip, ..., 0] = chip * 100.0 + cols
        patch_xyz[chip, ..., 1] = rows
        patch_xyz[chip, ..., 2] = rows * cols + cols
    np.save(extract / "patch_xyz.npy", patch_xyz)
    seen = []

    def inspect_pnp(corr, _K):
        seen.append(corr.xyz.copy())
        # A query's PnP set must come from one candidate, not top-K chips.
        assert np.ptp(corr.xyz[:, 0]) < 10.0
        assert np.unique(corr.xyz, axis=0).shape[0] > 1
        return PnPResult(np.eye(4), np.arange(len(corr)), 0.0, True, len(corr))

    monkeypatch.setattr("monarc.localization.eval_match_pnp.solve_pnp_lm", inspect_pnp)
    report = evaluate_match_pnp_dirs(extract, fsq, axis="east", top_k=5)
    assert seen
    assert report["xyz_kind"] == "per-patch-3dep"
    assert report["xyz_is_chip_center"] is False
    assert "per-patch-metric-xyz" not in report["not"]
