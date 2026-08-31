"""Command-line entry for local MonARC ingest, training, and evaluation paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def default_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _allow_download_flag(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "allow_download", False)):
        return True
    raw = os.environ.get("MONARC_DINO_ALLOW_DOWNLOAD", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cmd_dry_run(args: argparse.Namespace) -> int:
    from monarc.dryrun import run_dry_run

    report = run_dry_run(args.out, seed=args.seed, steps=args.steps, device="cpu")
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from monarc.map.extract import extract_chips

    meta = extract_chips(
        args.chips,
        args.out,
        size=args.size,
        dsm_dir=args.dsm_dir,
        xyz_path=args.xyz,
        backbone_mode=args.backbone,
        weights_path=args.weights,
        allow_download=_allow_download_flag(args),
        device=args.device,
        batch_size=args.batch_size,
        fsq_ckpt=args.fsq_ckpt,
    )
    json.dump(meta, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_train_fsq(args: argparse.Namespace) -> int:
    from monarc.map.train_fsq import train_fsq_from_cache

    report = train_fsq_from_cache(
        args.features,
        args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        ckpt_every=args.ckpt_every,
        keep_last=args.keep_last,
        resume=args.resume,
        seed=args.seed,
        levels=args.levels,
        smooth_weight=args.smooth_weight,
        usage_weight=args.usage_weight,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_ingest_aoi(args: argparse.Namespace) -> int:
    from monarc.data.aflora_ingest import ingest_aoi_to_path

    path = ingest_aoi_to_path(
        args.out,
        center=args.center,
        size_km=args.size_km,
        offline=args.offline,
        source=args.source,
        chips_dir=args.chips,
        chip_size=args.chip_size,
        chip_grid=args.chip_grid,
        max_chips=args.max_chips,
        materialize_only=args.materialize_only,
    )
    sys.stdout.write(str(path) + "\n")
    return 0


def _cmd_eval_retrieve(args: argparse.Namespace) -> int:
    from monarc.localization.eval_retrieve import evaluate_retrieve_dirs

    report = evaluate_retrieve_dirs(
        args.extract,
        args.fsq,
        query_fraction=args.query_fraction,
        axis=args.axis,
        out=args.out,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_eval_place_score(args: argparse.Namespace) -> int:
    from monarc.localization.eval_place_score import evaluate_place_score_dirs

    report = evaluate_place_score_dirs(
        args.extract, args.fsq, gsd_m=args.gsd_m, query_fraction=args.query_fraction,
        axis=args.axis, crop_margin=args.crop_margin, top_k=args.top_k, out=args.out,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_eval_match_pnp(args: argparse.Namespace) -> int:
    from monarc.localization.eval_match_pnp import evaluate_match_pnp_dirs

    report = evaluate_match_pnp_dirs(
        args.extract,
        args.fsq,
        query_fraction=args.query_fraction,
        axis=args.axis,
        top_k=args.top_k,
        retrieve_mode=args.retrieve_mode,
        min_cosine=args.min_cosine,
        out=args.out,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_fill_dsm_z(args: argparse.Namespace) -> int:
    from monarc.map.dsm_z import fill_xyz_dirs

    report = fill_xyz_dirs(
        args.extract,
        manifest_path=args.manifest,
        fsq_dir=args.fsq,
        chips_dir=args.chips,
        href=args.href,
        offline=args.offline,
        patches=args.patches,
        gsd_m=args.gsd_m,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_bench_uav(args: argparse.Namespace) -> int:
    from monarc.data.uav_benchmarks import University1652, list_public_uav_benches

    if args.list_benches:
        json.dump(list_public_uav_benches(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.dataset != "university1652":
        sys.stderr.write(
            f"dataset {args.dataset!r} has no first-path loader; "
            "use --list-benches\n"
        )
        return 2
    if args.root is None:
        sys.stderr.write("--root is required unless --list-benches is set\n")
        return 2
    bench = University1652(
        args.root,
        download=args.download,
        download_url=args.download_url,
    )
    payload = bench.summary()
    if not args.list_only:
        payload["pairs"] = [
            {"building_id": p.building_id, "drone": str(p.drone), "satellite": str(p.satellite)}
            for p in bench.pairs("train")
        ]
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monarc",
        description=(
            "MonARC: frozen DINOv2-B (stub by default; official vitb14 on GPU), "
            "FSQ train, code-xyz index, retrieve, matcher+PnP/LM. No Hunter."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    dry = sub.add_parser(
        "dry-run",
        help="Synthetic chips: DINO stub features, train FSQ, index, retrieve, PnP/LM",
    )
    dry.add_argument("--out", type=Path, default=Path("artifacts/dry-run"))
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--steps", type=int, default=8)
    dry.set_defaults(func=_cmd_dry_run)

    extract = sub.add_parser(
        "extract",
        help="RGB chips -> frozen DINO features + xyz sidecar (optional DSM, optional FSQ codes)",
    )
    extract.add_argument("--chips", type=Path, required=True, help="Directory of RGB chips")
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--dsm-dir", type=Path, default=None)
    extract.add_argument("--xyz", type=Path, default=None, help="Optional xyz.npy or filename,x,y,z CSV")
    extract.add_argument("--size", type=int, default=224, help="Square resize; must be divisible by 14")
    extract.add_argument(
        "--backbone",
        default="auto",
        choices=["auto", "stub", "vitb14"],
        help="auto: stub on CPU / when weights absent; vitb14 when CUDA cache or --allow-download",
    )
    extract.add_argument("--weights", type=Path, default=None, help="Local DINOv2-B .pth or HF dir")
    extract.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit torch.hub facebookresearch/dinov2:dinov2_vitb14 (or HF facebook/dinov2-base)",
    )
    extract.add_argument("--device", default=None)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--fsq-ckpt", type=Path, default=None, help="Optional stage1_last.pt to emit codes.npy")
    extract.set_defaults(func=_cmd_extract)

    train = sub.add_parser(
        "train-fsq",
        help="Train fusion+FSQ on extract features (GPU). Writes codes.npy + xyz sidecar + checkpoints.",
    )
    train.add_argument("--features", type=Path, required=True, help="Directory from monarc extract")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", default=None)
    train.add_argument("--ckpt-every", type=int, default=50)
    train.add_argument("--keep-last", type=int, default=3)
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--levels",
        default="8,5,5,5",
        help="FSQ levels, comma-separated (default Mentzer 10-bit 8,5,5,5)",
    )
    train.add_argument("--smooth-weight", type=float, default=0.05)
    train.add_argument("--usage-weight", type=float, default=1.0)
    train.set_defaults(func=_cmd_train_fsq)

    aoi = sub.add_parser(
        "ingest-aoi",
        help=(
            "Golden-Morrison (or --center) NAIP+3DEP manifest. "
            "Default source is Planetary Computer (anonymous SAS, no ~/.aws)."
        ),
    )
    aoi.add_argument("--out", type=Path, required=True)
    aoi.add_argument(
        "--center",
        default=None,
        help="lat,lon (default: Golden-Morrison 39.725,-105.220)",
    )
    aoi.add_argument("--size-km", type=float, default=10.0)
    aoi.add_argument(
        "--source",
        default="planetary-computer",
        choices=["planetary-computer", "colorado-public-imagery", "naip-visualization"],
        help=(
            "planetary-computer: anonymous STAC+SAS (default). "
            "colorado-public-imagery: unsigned HTTPS list fallback. "
            "naip-visualization: explicit AWS requester-pays path; does not read ~/.aws."
        ),
    )
    aoi.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="Directory with naip_stac.json and tnm_products.json; skips live HTTP",
    )
    aoi.add_argument(
        "--chips",
        type=Path,
        default=None,
        help="Write range-read PNG chips + xyz sidecars for monarc extract (Runpod 4090)",
    )
    aoi.add_argument("--chip-size", type=int, default=224)
    aoi.add_argument("--chip-grid", type=int, default=8, help="Sample grid along each AOI axis")
    aoi.add_argument("--max-chips", type=int, default=64)
    aoi.add_argument(
        "--materialize-only",
        action="store_true",
        help="Range-read chip windows from an existing --out manifest (no catalog query)",
    )
    aoi.set_defaults(func=_cmd_ingest_aoi)

    ev = sub.add_parser(
        "eval-retrieve",
        help=(
            "Colorado retrieval track: bag-of-codes baseline plus frozen DINO "
            "cosine Recall@1/5 and xyz error on a spatial holdout of extract+fsq "
            "chips. CPU, no network."
        ),
    )
    ev.add_argument("--extract", type=Path, required=True, help="Directory with features.npy (and xyz.npy)")
    ev.add_argument("--fsq", type=Path, required=True, help="Directory with codes.npy and xyz.npy")
    ev.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path. Always printed to stdout.",
    )
    ev.add_argument(
        "--query-fraction",
        type=float,
        default=0.25,
        help="Fraction of unique positions on the split axis held out as queries",
    )
    ev.add_argument(
        "--axis",
        default="auto",
        choices=["auto", "east", "north"],
        help="Spatial box axis. auto: longer east/north span. High side is queries.",
    )
    ev.set_defaults(func=_cmd_eval_retrieve)

    place_ev = sub.add_parser(
        "eval-place-score",
        help="Same-place crop/overlap verification from local DINO grids and FSQ codes. CPU, no network.",
    )
    place_ev.add_argument("--extract", type=Path, required=True)
    place_ev.add_argument("--fsq", type=Path, required=True)
    place_ev.add_argument("--out", type=Path, required=True)
    place_ev.add_argument("--gsd-m", type=float, default=0.3)
    place_ev.add_argument("--query-fraction", type=float, default=0.25,
                          help="Far spatial-box fraction only")
    place_ev.add_argument("--axis", default="auto", choices=["auto", "east", "north"],
                          help="Far spatial-box axis only")
    place_ev.add_argument("--crop-margin", type=int, default=2, help="Crop margin in DINO patches")
    place_ev.add_argument("--top-k", type=int, default=5)
    place_ev.set_defaults(func=_cmd_eval_place_score)

    match_ev = sub.add_parser(
        "eval-match-pnp",
        help=(
            "Retrieve top-K Colorado chips, match frozen DINO patch grids, and "
            "run PnP/LM with a chip-center xy fallback. CPU, no network."
        ),
    )
    match_ev.add_argument("--extract", type=Path, required=True)
    match_ev.add_argument("--fsq", type=Path, required=True)
    match_ev.add_argument("--out", type=Path, default=None)
    match_ev.add_argument("--query-fraction", type=float, default=0.25)
    match_ev.add_argument("--axis", default="auto", choices=["auto", "east", "north"])
    match_ev.add_argument("--top-k", type=int, default=5)
    match_ev.add_argument(
        "--retrieve-mode",
        default="bag-of-codes",
        choices=["bag-of-codes", "dino-pooled-cosine", "dino-grid-cosine"],
    )
    match_ev.add_argument("--min-cosine", type=float, default=0.8)
    match_ev.set_defaults(func=_cmd_eval_match_pnp)

    fill_z = sub.add_parser(
        "fill-dsm-z",
        help="Fill coarse chip-center ENU Z from public 3DEP range-read samples",
    )
    fill_z.add_argument("--extract", type=Path, required=True)
    fill_z.add_argument("--fsq", type=Path, default=None)
    fill_z.add_argument("--chips", type=Path, default=None)
    fill_z.add_argument("--manifest", type=Path, required=True)
    fill_z.add_argument(
        "--href", default=None, help="Override records with one local or public GeoTIFF"
    )
    fill_z.add_argument(
        "--offline", action="store_true", help="Reject HTTP sources (use with local --href)"
    )
    fill_z.add_argument(
        "--patches", action="store_true",
        help="Also write patch_xyz.npy by sampling every DINO cell center",
    )
    fill_z.add_argument(
        "--gsd-m", type=float, default=0.3,
        help="Fallback chip ground sample distance (default: 0.3 m for NAIP rehearsal)",
    )
    fill_z.set_defaults(func=_cmd_fill_dsm_z)

    bench = sub.add_parser("bench-uav", help="List or parse a public UAV bench (University-1652)")
    bench.add_argument("--dataset", default="university1652")
    bench.add_argument("--root", type=Path, default=None)
    bench.add_argument("--list-only", action="store_true")
    bench.add_argument("--list-benches", action="store_true")
    bench.add_argument(
        "--download",
        action="store_true",
        help="Fetch a licensed zip/tar into --root (requires --download-url or MONARC_U1652_URL)",
    )
    bench.add_argument("--download-url", default=None)
    bench.set_defaults(func=_cmd_bench_uav)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "device", None) is None and args.cmd in {"extract", "train-fsq"}:
        args.device = default_torch_device()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
