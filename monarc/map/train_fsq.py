"""Train fusion + FSQ on cached DINO features (GPU train plane)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from monarc.map.extract import load_feature_cache
from monarc.map.metric_index import index_from_tokens
from monarc.map.quantizer import DEFAULT_FSQ_LEVELS, fsq_usage_floor, parse_fsq_levels
from monarc.map.stage1 import (
    RgbReconHead,
    default_stage1_modules,
    load_stage1_checkpoint,
    stage1_checkpoint,
    train_from_features,
)


def _prune_ckpts(out_dir: Path, keep_last: int) -> None:
    if keep_last < 0:
        return
    ckpts = sorted(out_dir.glob("ckpt_step_*.pt"))
    extra = ckpts[: max(0, len(ckpts) - keep_last)]
    for path in extra:
        path.unlink()


def train_fsq_from_cache(
    features_dir: str | Path,
    out_dir: str | Path,
    *,
    steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    ckpt_every: int = 50,
    keep_last: int = 3,
    resume: str | Path | None = None,
    levels: Sequence[int] | str = DEFAULT_FSQ_LEVELS,
    seed: int = 0,
    smooth_weight: float = 0.05,
    usage_weight: float = 1.0,
) -> dict:
    """Consume ``extract`` output: DINO grids (+ optional DSM) -> FSQ codes + xyz."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_feature_cache(features_dir)
    f_rgb = torch.from_numpy(np.asarray(cache["features"], dtype=np.float32))
    dsm = (
        torch.from_numpy(np.asarray(cache["dsm"], dtype=np.float32))
        if cache["dsm"] is not None
        else None
    )
    vectors = (
        torch.from_numpy(np.asarray(cache["vectors"], dtype=np.float32))
        if cache["vectors"] is not None
        else None
    )
    xyz = np.asarray(cache["xyz"], dtype=np.float64)
    _, stem, mix, head = default_stage1_modules(levels=parse_fsq_levels(levels))
    recon = RgbReconHead(head.fsq.num_dimensions, int(f_rgb.shape[1]))
    start_step = 0
    if resume is not None:
        payload = load_stage1_checkpoint(
            str(resume), stem, mix, head, recon=recon, map_location="cpu"
        )
        start_step = int(payload.get("step", 0))

    def _write(step: int, rec: RgbReconHead, tagged: bool) -> None:
        payload = stage1_checkpoint(
            stem,
            mix,
            head,
            step=step,
            recon=rec,
            extra={"device": str(device), "features_dir": str(features_dir)},
        )
        torch.save(payload, out_dir / "stage1_last.pt")
        if tagged:
            torch.save(payload, out_dir / f"ckpt_step_{step:06d}.pt")
            _prune_ckpts(out_dir, keep_last)

    def on_step(
        step: int,
        _loss: float,
        opt: torch.optim.Optimizer,
        rec: RgbReconHead,
    ) -> None:
        if ckpt_every > 0 and step % int(ckpt_every) == 0:
            _write(step, rec, tagged=True)

    try:
        stats = train_from_features(
            stem,
            mix,
            head,
            f_rgb,
            dsm,
            vectors,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            device=device,
            recon=recon,
            start_step=start_step,
            optimizer=None,
            on_step=on_step,
            smooth_weight=smooth_weight,
            usage_weight=usage_weight,
        )
    except KeyboardInterrupt:
        last = out_dir / "stage1_last.pt"
        if not last.exists():
            _write(start_step, recon, tagged=False)
        raise

    _write(int(stats["step"]), stats["recon"], tagged=True)
    codes = stats["codes"].numpy()
    np.save(out_dir / "codes.npy", codes)
    np.save(out_dir / "xyz.npy", xyz)
    meta = {
        "fsq_levels": list(head.fsq.levels),
        "codebook_size": int(head.fsq.codebook_size),
        "unique_codes": int(stats["unique_codes"]),
        "steps": int(stats["step"]),
        "device": str(device),
        "features_dir": str(features_dir),
        "has_dsm": cache["dsm"] is not None,
        "backbone_mode": cache["meta"].get("backbone_mode"),
        "persist": ["fsq_codes", "xyz", "compact_metadata"],
        "rasters_copied": False,
        "note": (
            "FSQ codes + xyz sidecar. Do not copy NAIP/3DEP rasters to R2. "
            "Numbers in this run are not Colorado flight metrics."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if np.isfinite(xyz).all() and xyz.shape[0] == codes.reshape(-1, *codes.shape[1:]).shape[0]:
        code_flat = codes.reshape(codes.shape[0], -1)
        # One code per chip: take the center patch so xyz stays [N, 3].
        center = code_flat[:, code_flat.shape[1] // 2]
        index_from_tokens(center, xyz, meta).save(out_dir / "metric_index")
    n_chips = int(f_rgb.shape[0])
    floor = fsq_usage_floor(
        n_chips,
        int(stats["codebook_size"]),
        int(stats["n_tokens"]),
    )
    report = {
        "step": int(stats["step"]),
        "losses": [float(x) for x in stats["losses"]],
        "unique_codes": int(stats["unique_codes"]),
        "codebook_size": int(stats["codebook_size"]),
        "n_tokens": int(stats["n_tokens"]),
        "out_dir": str(out_dir),
        "has_dsm": cache["dsm"] is not None,
        "n_chips": n_chips,
        "fsq_levels": list(head.fsq.levels),
        "usage_floor": int(floor),
        "collapsed": int(stats["unique_codes"]) < int(floor),
    }
    (out_dir / "train_fsq_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
