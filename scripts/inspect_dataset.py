#!/usr/bin/env python3
"""Inspect one Lakeside dataset sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402
from lakeside.io import CAMERAS, COMPONENTS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Lakeside sequence and print resolved component folders, "
            "sensor counts, metrics path, and first/last filenames."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root containing {date}_{traverse}_{component} folders.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="YYMMDD recording date, for example 200803.",
    )
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type to inspect.",
    )
    parser.add_argument(
        "--use-downsampled-images",
        action="store_true",
        help="Prefer *_downsampled camera folders when available.",
    )
    return parser


def file_name(path: Path | None) -> str:
    return path.name if path is not None else "none"


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(
        args.root,
        date=args.date,
        traverse=args.traverse,
        use_downsampled_images=args.use_downsampled_images,
    )
    summary = sequence.summary()

    print("Lakeside sequence inspection")
    print(f"root: {summary['root'].resolve()}")
    print(f"date: {summary['date']}")
    print(f"traverse: {summary['traverse']}")
    print(f"use_downsampled_images: {summary['use_downsampled_images']}")
    print()

    print("resolved component folders:")
    for component in COMPONENTS:
        path = summary["component_paths"][component]
        status = "found" if path.exists() else "missing"
        print(f"  {component}: {path} ({status})")
    print()

    print("sensor files:")
    for sensor in ("lidar", *CAMERAS):
        sensor_summary = summary["sensors"][sensor]
        print(
            f"  {sensor}: count={sensor_summary.count}, "
            f"first={file_name(sensor_summary.first)}, "
            f"last={file_name(sensor_summary.last)}"
        )
    print()

    metrics_file = summary["metrics_file"]
    if metrics_file is None:
        print("metrics Imu.csv: not found")
    else:
        print(f"metrics Imu.csv: {metrics_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
