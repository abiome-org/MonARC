"""Extract frozen DINO features from a directory of RGB chips.

Writes feature grids, optional FSQ codes, and an xyz sidecar. RGB/DSM rasters
are not copied into the output (they are not an R2 product).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from monarc.map.dino_backbone import (
    DINOV2_B_PATCH,
    FrozenDinoBackbone,
    load_frozen_dino,
)
from monarc.map.fusion_stem import DEFAULT_VECTOR_CHANNELS
from monarc.map.stage1 import default_stage1_modules, encode_from_features

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def write_chip_fixture(
    root: str | Path,
    n: int = 4,
    size: int = 28,
    with_dsm: bool = True,
) -> Path:
    """Write solid-color RGB chips plus xyz (and optional DSM) sidecars."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        color = (40 + 50 * (i % 4), 90 + 20 * i, 140)
        Image.new("RGB", (size, size), color=color).save(root / f"chip_{i:04d}.png")
        (root / f"chip_{i:04d}.xyz.json").write_text(
            json.dumps({"xyz": [float(i * 12.0), 0.0, 80.0 + i]}) + "\n"
        )
        if with_dsm:
            dsm = np.full((size, size), 80.0 + i, dtype=np.float32)
            np.save(root / f"chip_{i:04d}.dsm.npy", dsm)
    return root


def _is_rgb_chip(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    stem = path.stem.lower()
    if stem.endswith(".dsm") or stem.endswith("_dsm") or stem.endswith("-dsm"):
        return False
    return True


def list_rgb_chips(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and _is_rgb_chip(p)]
    return sorted(files)


def _load_xyz_sidecar(chip: Path) -> np.ndarray:
    json_path = chip.with_suffix(".xyz.json")
    if not json_path.exists() and chip.suffix.lower() != ".json":
        json_path = chip.parent / f"{chip.stem}.xyz.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        if isinstance(payload, dict) and "xyz" in payload:
            return np.asarray(payload["xyz"], dtype=np.float64).reshape(3)
        return np.asarray(
            [payload["x"], payload["y"], payload["z"]], dtype=np.float64
        ).reshape(3)
    npy_path = chip.parent / f"{chip.stem}.xyz.npy"
    if npy_path.is_file():
        return np.asarray(np.load(npy_path), dtype=np.float64).reshape(3)
    return np.full((3,), np.nan, dtype=np.float64)


def load_xyz_table(path: Path, chips: Sequence[Path]) -> np.ndarray:
    """Load xyz aligned to ``chips`` from ``.npy`` or ``filename,x,y,z`` CSV."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        xyz = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
        if xyz.shape[0] != len(chips):
            raise ValueError(f"xyz rows {xyz.shape[0]} != n chips {len(chips)}")
        return xyz
    by_name: dict[str, np.ndarray] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = Path(row["filename"]).name
            by_name[key] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
            )
    rows = []
    for chip in chips:
        if chip.name not in by_name:
            raise KeyError(f"no xyz row for {chip.name} in {path}")
        rows.append(by_name[chip.name])
    return np.stack(rows, axis=0)


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _find_dsm(chip: Path, dsm_dir: Path | None) -> Path | None:
    candidates = [
        chip.parent / f"{chip.stem}.dsm.npy",
        chip.parent / f"{chip.stem}_dsm.npy",
        chip.parent / f"{chip.stem}.dsm.tif",
        chip.parent / f"{chip.stem}.dsm.tiff",
    ]
    if dsm_dir is not None:
        candidates.extend(
            [
                dsm_dir / f"{chip.stem}.npy",
                dsm_dir / f"{chip.stem}.tif",
                dsm_dir / f"{chip.stem}.tiff",
                dsm_dir / f"{chip.stem}.png",
                dsm_dir / chip.name,
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_dsm(path: Path, size: int) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
    else:
        img = Image.open(path)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.max() > 1.0 and img.mode in {"L", "RGB", "P"}:
            pass
    if arr.shape[-2:] != (size, size):
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        return tensor[0]
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def extract_chips(
    chips_dir: str | Path,
    out_dir: str | Path,
    *,
    size: int = 224,
    dsm_dir: str | Path | None = None,
    xyz_path: str | Path | None = None,
    backbone: FrozenDinoBackbone | None = None,
    backbone_mode: str = "auto",
    weights_path: str | Path | None = None,
    allow_download: bool = False,
    device: str = "cpu",
    batch_size: int = 8,
    fsq_ckpt: str | Path | None = None,
    vector_channels: int = DEFAULT_VECTOR_CHANNELS,
) -> dict:
    """Run frozen DINO on RGB chips and write ``features.npy`` + xyz sidecar."""
    chips_dir = Path(chips_dir)
    out_dir = Path(out_dir)
    if size % DINOV2_B_PATCH != 0:
        raise ValueError(f"size {size} must be divisible by {DINOV2_B_PATCH}")
    chips = list_rgb_chips(chips_dir)
    if not chips:
        raise FileNotFoundError(f"no RGB chips in {chips_dir}")
    dsm_dir_p = Path(dsm_dir) if dsm_dir is not None else None
    if backbone is None:
        backbone = load_frozen_dino(
            mode=backbone_mode,
            weights_path=weights_path,
            allow_download=allow_download,
            device=device,
        )
    else:
        backbone = backbone.to(device)
    backbone.eval()
    if xyz_path is not None:
        xyz = load_xyz_table(Path(xyz_path), chips)
    else:
        xyz = np.stack([_load_xyz_sidecar(p) for p in chips], axis=0)

    n = len(chips)
    grid = size // backbone.patch_size
    features = np.zeros((n, backbone.embed_dim, grid, grid), dtype=np.float16)
    dsm_stack = None
    dsm_paths = [_find_dsm(p, dsm_dir_p) for p in chips]
    has_dsm = all(p is not None for p in dsm_paths)
    if has_dsm:
        dsm_stack = np.zeros((n, 1, size, size), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = chips[start : start + batch_size]
            rgb = torch.stack([_load_rgb(p, size) for p in chunk], dim=0).to(device)
            feat = backbone(rgb).detach().cpu().numpy().astype(np.float16)
            features[start : start + len(chunk)] = feat
            if has_dsm:
                for i, path in enumerate(dsm_paths[start : start + len(chunk)]):
                    assert path is not None
                    dsm_stack[start + i, 0] = _load_dsm(path, size).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "xyz.npy", xyz)
    if has_dsm:
        np.save(out_dir / "dsm.npy", dsm_stack)
        vectors = np.zeros((n, vector_channels, size, size), dtype=np.float32)
        np.save(out_dir / "vectors.npy", vectors)

    codes = None
    if fsq_ckpt is not None:
        payload = torch.load(fsq_ckpt, map_location="cpu", weights_only=True)
        _, stem, mix, head = default_stage1_modules(
            vector_channels=vector_channels,
            levels=payload.get("levels", (5, 5, 5)),
        )
        if "fusion_stem" in payload:
            stem.load_state_dict(payload["fusion_stem"])
        if "channel_fusion" in payload:
            mix.load_state_dict(payload["channel_fusion"])
        if "fsq_head" in payload:
            head.load_state_dict(payload["fsq_head"])
        f_rgb = torch.from_numpy(features.astype(np.float32))
        dsm_t = (
            torch.from_numpy(dsm_stack)
            if dsm_stack is not None
            else None
        )
        vec_t = (
            torch.from_numpy(vectors)
            if dsm_stack is not None
            else None
        )
        _fused, codes_t = encode_from_features(
            stem, mix, head, f_rgb, dsm_t, vec_t, device=device
        )
        codes = codes_t.numpy()
        np.save(out_dir / "codes.npy", codes)

    ids = [p.name for p in chips]
    (out_dir / "ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    meta = {
        "n_chips": n,
        "size": size,
        "patch_size": backbone.patch_size,
        "embed_dim": backbone.embed_dim,
        "backbone_mode": backbone.mode,
        "backbone_source": backbone.source,
        "device": str(device),
        "has_dsm": bool(has_dsm),
        "xyz_finite": bool(np.isfinite(xyz).all()),
        "persist": ["features", "xyz", "compact_metadata"],
        "rasters_copied": False,
        "r2_objects": ["codes", "xyz", "meta"] if codes is not None else ["features", "xyz", "meta"],
        "note": "Features and FSQ codes only. Do not copy NAIP/3DEP rasters to R2.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def load_feature_cache(directory: str | Path) -> dict:
    """Load an extract directory written by ``extract_chips``."""
    directory = Path(directory)
    features = np.load(directory / "features.npy")
    xyz = np.load(directory / "xyz.npy")
    meta = json.loads((directory / "meta.json").read_text())
    dsm = np.load(directory / "dsm.npy") if (directory / "dsm.npy").exists() else None
    vectors = np.load(directory / "vectors.npy") if (directory / "vectors.npy").exists() else None
    codes = np.load(directory / "codes.npy") if (directory / "codes.npy").exists() else None
    ids = json.loads((directory / "ids.json").read_text()) if (directory / "ids.json").exists() else []
    return {
        "features": features,
        "xyz": xyz,
        "dsm": dsm,
        "vectors": vectors,
        "codes": codes,
        "ids": ids,
        "meta": meta,
    }
