#!/usr/bin/env python3
"""List sensor frame files for one Lakeside sequence."""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the first N frame records for one Lakeside component, "
            "including index, timestamp or frame_id, and file path."
        )
    )
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", required=True, help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        required=True,
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=("lidar", "cam1", "cam2", "cam3"),
        help="Component stream to list.",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Prefer *_downsampled camera folders when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of frames to print. Defaults to 10.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(
        args.root,
        args.date,
        args.traverse,
        use_downsampled_images=args.downsampled,
    )
    limit = max(args.limit, 0)

    if args.component == "lidar":
        frames = sequence.iter_lidar()
        print("index\ttimestamp\tpath")
        for frame in islice(frames, limit):
            print(f"{frame.index}\t{frame.timestamp:.9f}\t{frame.path}")
    else:
        frames = sequence.iter_camera(args.component)
        print("index\tframe_id\tpath")
        for frame in islice(frames, limit):
            print(f"{frame.index}\t{frame.frame_id}\t{frame.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
