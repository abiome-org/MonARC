"""End-to-end CPU dry-run CLI on synthetic chips."""

import json
import os
from pathlib import Path

from monarc.cli import main
from monarc.dryrun import run_dry_run


def test_dry_run_pipeline(tmp_path):
    report = run_dry_run(tmp_path / "out", seed=1, steps=4, device="cpu")
    assert report["backbone_mode"] == "stub"
    assert report["n_landmarks"] == 64
    assert report["retrieve"]["rank1_identity"] is True
    assert report["pnp"]["success"] is True
    assert report["pnp"]["translation_error_m"] < 2.0
    assert (tmp_path / "out" / "metric_index" / "codes.npy").exists()
    assert (tmp_path / "out" / "dry_run_report.json").exists()
    assert not list(Path(tmp_path).glob("**/*.tif"))


def test_cli_dry_run(tmp_path, capsys):
    code = main(["dry-run", "--out", str(tmp_path / "cli"), "--steps", "2", "--seed", "0"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["codebook_size"] == 125


def test_cli_ingest_offline(tmp_path, capsys):
    fixtures = Path(__file__).parent / "fixtures" / "inventory"
    out = tmp_path / "m.json"
    code = main(["ingest-aoi", "--out", str(out), "--offline", str(fixtures)])
    assert code == 0
    assert out.exists()


def test_cli_bench_fixture(tmp_path, capsys):
    from monarc.data.uav_benchmarks import write_university1652_fixture

    root = write_university1652_fixture(tmp_path / "u")
    code = main(["bench-uav", "--root", str(root), "--list-only"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "university1652"


def test_cuda_env_is_blank():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
