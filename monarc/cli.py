"""Command-line entry: dry-run, AOI ingest, public-UAV bench listing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_dry_run(args: argparse.Namespace) -> int:
    from monarc.dryrun import run_dry_run

    report = run_dry_run(args.out, seed=args.seed, steps=args.steps, device="cpu")
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
    )
    sys.stdout.write(str(path) + "\n")
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
    bench = University1652(args.root)
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
            "MonARC first executable path: frozen-DINO stub tokens, tiny FSQ, "
            "code-xyz index, retrieve, matcher+PnP/LM. CPU dry-run. No Hunter."
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

    aoi = sub.add_parser(
        "ingest-aoi",
        help="Intersect Golden-Morrison (or --center) with NAIP visualization STAC and 3DEP inventory",
    )
    aoi.add_argument("--out", type=Path, required=True)
    aoi.add_argument(
        "--center",
        default=None,
        help="lat,lon (default: Golden-Morrison 39.725,-105.220)",
    )
    aoi.add_argument("--size-km", type=float, default=10.0)
    aoi.add_argument(
        "--offline",
        type=Path,
        default=None,
        help="Directory with naip_stac.json and tnm_products.json; skips live HTTP",
    )
    aoi.set_defaults(func=_cmd_ingest_aoi)

    bench = sub.add_parser("bench-uav", help="List or parse a public UAV bench (University-1652)")
    bench.add_argument("--dataset", default="university1652")
    bench.add_argument("--root", type=Path, default=None)
    bench.add_argument("--list-only", action="store_true")
    bench.add_argument("--list-benches", action="store_true")
    bench.set_defaults(func=_cmd_bench_uav)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
