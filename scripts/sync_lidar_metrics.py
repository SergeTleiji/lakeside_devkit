#!/usr/bin/env python3
"""Print nearest-timestamp LiDAR-to-metrics matches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lakeside import LakesideSequence, match_lidar_frames_to_metrics  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match LiDAR frames to Imu.csv metrics rows by nearest Unix "
            "timestamp and print the first N matches."
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
        "--limit",
        type=int,
        default=20,
        help="Maximum number of LiDAR frames to match. Defaults to 20.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sequence = LakesideSequence(args.root, args.date, args.traverse)
    matches = match_lidar_frames_to_metrics(sequence, limit=args.limit)

    print("lidar_index\tlidar_timestamp\tmetrics_timestamp\tdelta_seconds\trtk_status")
    for match in matches:
        lidar_frame = match["lidar_frame"]
        metrics_row = match["metrics_row"]
        rtk_status = metrics_row.get("RTK_Status", "none")
        print(
            f"{lidar_frame.index}\t"
            f"{lidar_frame.timestamp:.9f}\t"
            f"{match['metrics_timestamp']:.9f}\t"
            f"{match['delta_seconds']:.9f}\t"
            f"{format_rtk_status(rtk_status)}"
        )
    return 0


def format_rtk_status(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
