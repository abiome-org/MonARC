"""University-1652 fixture loader. No dataset download."""

import shutil
from pathlib import Path

from monarc.data.uav_benchmarks import University1652, list_public_uav_benches, write_university1652_fixture


def test_university1652_is_first_public_bench():
    benches = list_public_uav_benches()
    implemented = [b for b in benches if b["status"] == "implemented"]
    assert implemented[0]["name"] == "university1652"
    ortho = [b for b in benches if b["name"] == "ortholoc"][0]
    assert ortho["status"] == "registered"
    assert "287" in ortho["notes"] or "awkward" in ortho["notes"].lower()


def test_fixture_pairs(tmp_path):
    root = write_university1652_fixture(tmp_path / "u1652", n_buildings=2, images_per=2)
    bench = University1652(root)
    pairs = bench.pairs("train")
    assert len(pairs) == 2
    assert {p.building_id for p in pairs} == {"0001", "0002"}
    train = bench.records("train")
    views = {r.view for r in train}
    assert views == {"drone", "satellite"}
    summary = bench.summary()
    assert summary["n_train_pairs"] == 2
    assert summary["report_track"] == "public-uav-adapter"
    query = bench.records("query_drone")
    assert len(query) == 4


def test_missing_root_does_not_download(tmp_path):
    missing = tmp_path / "nope"
    try:
        University1652(missing)
        assert False, "missing root must raise"
    except FileNotFoundError as exc:
        assert "optional" in str(exc).lower()


def test_download_without_url_does_not_fetch(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("network")

    monkeypatch.setattr("monarc.data.uav_benchmarks._fetch_url", boom)
    try:
        University1652(tmp_path / "missing", download=True)
        assert False, "download without URL must raise"
    except FileNotFoundError as exc:
        assert "MONARC_U1652_URL" in str(exc)


def test_download_from_local_zip(tmp_path):
    src = write_university1652_fixture(tmp_path / "src", n_buildings=2, images_per=1)
    archive = tmp_path / "u1652"
    zip_path = Path(shutil.make_archive(str(archive), "zip", root_dir=src))
    dest = tmp_path / "dest"
    bench = University1652(dest, download=True, download_url=zip_path.as_uri())
    assert len(bench.pairs("train")) == 2
    assert (dest / "train" / "drone").is_dir()
