"""Minimal Lakeside devkit loading example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load one Lakeside sequence.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root.")
    parser.add_argument("--date", default="200803", help="YYMMDD recording date.")
    parser.add_argument(
        "--traverse",
        default="on-foot",
        choices=("on-foot", "scooter"),
        help="Traversal type.",
    )
    parser.add_argument(
        "--downsampled",
        action="store_true",
        help="Prefer *_downsampled camera folders.",
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

    print("summary:")
    print(sequence.summary())
    print()

    lidar_frame = next(sequence.iter_lidar(), None)
    print("first lidar frame:")
    print(lidar_frame if lidar_frame is not None else "not found")
    print()

    for camera in ("cam1", "cam2", "cam3"):
        camera_frame = next(sequence.iter_camera(camera), None)
        print(f"first {camera} frame:")
        print(camera_frame if camera_frame is not None else "not found")
    print()

    print("metrics summary:")
    print(sequence.metrics_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
