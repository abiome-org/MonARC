"""Public UAV benchmark loaders. Download is optional; tests use fixtures.

University-1652 is the first implemented bench: a documented ImageFolder of
drone/satellite building IDs. OrthoLoC is registered but not the default path
(287 GB TUM dump, npz+GeoTIFF pairs, CC BY-NC-SA). DenseUAV is registered for
the same reason as a follow-on retrieval set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

UNIVERSITY1652_TRAIN_VIEWS = ("drone", "satellite")
UNIVERSITY1652_TEST_SPLITS = {
    "query_drone": ("test", "query_drone"),
    "gallery_drone": ("test", "gallery_drone"),
    "query_satellite": ("test", "query_satellite"),
    "gallery_satellite": ("test", "gallery_satellite"),
}

# Street views exist on disk for University-1652 but are out of MonARC's aerial
# codebook (docs/non-goals.md §2.5). The loader exposes them only if requested.
UNIVERSITY1652_STREET_VIEWS = ("street", "google", "query_street", "gallery_street")


@dataclass(frozen=True)
class BenchRecord:
    dataset: str
    split: str
    view: str
    building_id: str
    path: Path
    role: str


@dataclass(frozen=True)
class CrossViewPair:
    building_id: str
    drone: Path
    satellite: Path
    split: str


class DatasetLayoutError(FileNotFoundError):
    """Raised when a root exists but does not match the documented layout."""


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def _building_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


class University1652:
    """Parse the layumi University-1652 directory tree without downloading it.

    Expected layout (subset used by MonARC)::

        root/train/drone/<id>/*.jpg
        root/train/satellite/<id>/*.jpg
        root/test/query_drone/<id>/*.jpg
        root/test/gallery_satellite/<id>/*.jpg
        ...
    """

    name = "university1652"
    license_class = "academic-research"
    source_url = "https://github.com/layumi/University1652-Baseline"
    report_track = "public-uav-adapter"

    def __init__(self, root: str | Path, include_street: bool = False) -> None:
        self.root = Path(root)
        self.include_street = include_street
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"University-1652 root not found: {self.root}. "
                "Download is optional; pass a local tree or use the test fixture."
            )

    def _split_dir(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def records(self, split: str = "train") -> list[BenchRecord]:
        if split == "train":
            views = list(UNIVERSITY1652_TRAIN_VIEWS)
            if self.include_street:
                views.extend(("street", "google"))
            out: list[BenchRecord] = []
            for view in views:
                base = self._split_dir("train", view)
                for building in _building_dirs(base):
                    for image in _images_in(building):
                        out.append(
                            BenchRecord(
                                dataset=self.name,
                                split="train",
                                view=view,
                                building_id=building.name,
                                path=image,
                                role="train",
                            )
                        )
            return out
        if split not in UNIVERSITY1652_TEST_SPLITS:
            raise KeyError(f"unknown University-1652 split {split!r}")
        parent, leaf = UNIVERSITY1652_TEST_SPLITS[split]
        base = self._split_dir(parent, leaf)
        view = "drone" if "drone" in split else "satellite"
        role = "query" if split.startswith("query") else "gallery"
        out = []
        for building in _building_dirs(base):
            for image in _images_in(building):
                out.append(
                    BenchRecord(
                        dataset=self.name,
                        split=split,
                        view=view,
                        building_id=building.name,
                        path=image,
                        role=role,
                    )
                )
        return out

    def pairs(self, split: str = "train") -> list[CrossViewPair]:
        """One drone image and one satellite image per building when both exist."""
        if split == "train":
            drone_root = self._split_dir("train", "drone")
            sat_root = self._split_dir("train", "satellite")
        elif split == "test":
            drone_root = self._split_dir("test", "query_drone")
            sat_root = self._split_dir("test", "query_satellite")
        else:
            raise KeyError(f"pairs() expects split 'train' or 'test', got {split!r}")
        pairs: list[CrossViewPair] = []
        drone_ids = {p.name: p for p in _building_dirs(drone_root)}
        sat_ids = {p.name: p for p in _building_dirs(sat_root)}
        for building_id in sorted(set(drone_ids) & set(sat_ids)):
            drones = _images_in(drone_ids[building_id])
            sats = _images_in(sat_ids[building_id])
            if not drones or not sats:
                continue
            pairs.append(
                CrossViewPair(
                    building_id=building_id,
                    drone=drones[0],
                    satellite=sats[0],
                    split=split,
                )
            )
        return pairs

    def building_ids(self, split: str = "train") -> list[str]:
        return sorted({r.building_id for r in self.records(split)})

    def summary(self) -> dict:
        train = self.records("train")
        return {
            "dataset": self.name,
            "root": str(self.root),
            "license_class": self.license_class,
            "source_url": self.source_url,
            "report_track": self.report_track,
            "n_train_images": len(train),
            "n_train_buildings": len({r.building_id for r in train}),
            "n_train_pairs": len(self.pairs("train")),
            "include_street": self.include_street,
        }


def write_university1652_fixture(root: str | Path, n_buildings: int = 2, images_per: int = 2) -> Path:
    """Write a tiny University-1652 tree of solid-color JPEGs (no network)."""
    from PIL import Image

    root = Path(root)
    ids = [f"{i:04d}" for i in range(1, n_buildings + 1)]
    layouts = [
        ("train", "drone"),
        ("train", "satellite"),
        ("test", "query_drone"),
        ("test", "gallery_drone"),
        ("test", "query_satellite"),
        ("test", "gallery_satellite"),
    ]
    for parent, view in layouts:
        for i, building_id in enumerate(ids):
            folder = root / parent / view / building_id
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(images_per):
                color = (
                    40 + 40 * i,
                    80 + 20 * k,
                    160 if "sat" in view else 90,
                )
                img = Image.new("RGB", (32, 32), color=color)
                img.save(folder / f"image-{k:02d}.jpeg", format="JPEG")
    (root / "readme.txt").write_text("University-1652 fixture for MonARC unit tests.\n")
    return root


def list_public_uav_benches() -> list[dict]:
    """Registry of public UAV benches. Only University-1652 is a first-path loader."""
    return [
        {
            "name": "university1652",
            "loader": "monarc.data.uav_benchmarks.University1652",
            "status": "implemented",
            "report_track": "public-uav-adapter",
            "notes": "Least-friction first bench: ImageFolder building IDs, drone/satellite splits, no 6-DoF rasters.",
            "source_url": "https://github.com/layumi/University1652-Baseline",
            "license_class": "academic-research",
        },
        {
            "name": "denseuav",
            "loader": None,
            "status": "registered",
            "report_track": "public-uav-adapter",
            "notes": "Multi-altitude retrieval set. Deferred; layout is less standardized than University-1652.",
            "source_url": "https://github.com/Zgt-d/DenseUAV",
            "license_class": "research",
        },
        {
            "name": "ortholoc",
            "loader": None,
            "status": "registered",
            "report_track": "public-uav-adapter",
            "notes": (
                "Awkward as the first bench: ~287 GB TUM dump, npz+DOP+DSM samples, "
                "CC BY-NC-SA 4.0, mixed DE/US geodata. Better as a later 6-DoF adapter."
            ),
            "source_url": "https://deepscenario.github.io/OrthoLoC/",
            "license_class": "CC BY-NC-SA 4.0",
        },
    ]
